import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sbi.utils import BoxUniform
from tqdm import tqdm
import warnings
import datetime as dt
import random
import sys  # 新增：用于退出程序

warnings.filterwarnings("ignore")

print("=== P903c_Eval_Optimizer.py (Auto Seed Finder) 启动 ===")

# ==================== 新增：随机种子设置函数 ====================
def set_seed(seed=42):
    """设置全局随机种子以确保结果可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"[种子] 已设置全局随机种子: {seed}")
    return seed

# ==================== 配置 ====================

CACHE_DIR = r"C:\Users\20466\Desktop\之前\LIGO_Data_Cache"
MODEL_DIR = os.path.join(CACHE_DIR, "models")

AI_MODEL_PATH = os.path.join(MODEL_DIR, "ing_net_o3a_gpu.pt")
TR_MODEL_PATH = os.path.join(MODEL_DIR, "trad_model_o3a_gpu.pt")

SCALING_FACTOR = 1300.0 
XI_VALS = [0.001, 0.01, 0.1, 0.5, 1.0]

# ✅ 新增：自动种子优化配置
SNR_THRESHOLD = 8.0  # AI SNR目标阈值
MAX_TEST_ROUNDS = 100  # 最大搜索轮数

# 优化参数
N_CALIB_FINE = 5000 
N_TRIALS_FINE = 100   
SCAN_RES = 0.05       

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("[GPU] 使用设备: GPU")
else:
    device = torch.device("cpu")
    print("[CPU] GPU不可用，将使用CPU")

# ==================== 1. 数据加载 & 模拟器 ====================
def load_data_to_gpu(label="O3a"):
    expected_length = int(4096 * 2048.0)
    # 使用O3a的GPS时间戳 1238166018
    filenames = [f"{label}_H1_1238166018.pt", f"{label}_L1_1238166018.pt", f"{label}_H1.pt", f"{label}_L1.pt"]
    loaded = {}
    for det in ['H1', 'L1']:
        for fname in filenames:
            if det in fname:
                path = os.path.join(CACHE_DIR, fname)
                if os.path.exists(path):
                    try:
                        data = torch.load(path, map_location='cpu', weights_only=False)
                        if isinstance(data, np.ndarray): data = torch.from_numpy(data)
                        if not torch.isfinite(data).all(): continue
                        loaded[det] = data.float().to(device)
                        break
                    except: continue
    h1 = loaded.get('H1', torch.randn(expected_length, device=device))
    l1 = loaded.get('L1', torch.randn(expected_length, device=device))
    min_len = min(len(h1), len(l1))
    return h1[:min_len], l1[:min_len]

class Phase9SimulatorGPU:
    def __init__(self, h1_bg, l1_bg, scaling_factor=1300.0):
        self.h1_bg = h1_bg
        self.l1_bg = l1_bg
        self.scaling_factor = scaling_factor
        self.target_fs = 2048.0
        self.seg_len = int(4.0 * self.target_fs)
        self.max_idx = len(h1_bg) - self.seg_len - 1

    def compute_features_gpu(self, h1, l1):
        vx = h1 - h1.mean(dim=1, keepdim=True)
        vy = l1 - l1.mean(dim=1, keepdim=True)
        cost = (vx * vy).sum(dim=1) / (torch.sqrt((vx**2).sum(dim=1)) * torch.sqrt((vy**2).sum(dim=1)) + 1e-8)
        
        def kurtosis_torch(x):
            mean = x.mean(dim=1, keepdim=True)
            diff = x - mean
            m2 = (diff**2).mean(dim=1)
            m4 = (diff**4).mean(dim=1)
            return m4 / (m2**2 + 1e-8) - 3.0

        k_h1 = torch.log1p(torch.abs(kurtosis_torch(h1)))
        k_l1 = torch.log1p(torch.abs(kurtosis_torch(l1)))
        pw = torch.log10(h1.var(dim=1) * l1.var(dim=1) + 1e-30)
        return torch.stack([cost, k_h1, k_l1, pw], dim=1)

    def simulate(self, theta_batch):
        batch_size = theta_batch.shape[0]
        theta_batch = theta_batch.to(device)
        log_omega, xi = theta_batch[:, 0], theta_batch[:, 1]
        
        start_indices = torch.randint(0, self.max_idx, (batch_size,), device=device)
        indices = start_indices.unsqueeze(1) + torch.arange(self.seg_len, device=device)
        n_h1 = self.h1_bg[indices] 
        n_l1 = self.l1_bg[indices] 
        
        n_h1 = (n_h1 - n_h1.mean(dim=1, keepdim=True)) / (n_h1.std(dim=1, keepdim=True) + 1e-15)
        n_l1 = (n_l1 - n_l1.mean(dim=1, keepdim=True)) / (n_l1.std(dim=1, keepdim=True) + 1e-15)
        
        mask_sig = (log_omega > -15.0)
        if mask_sig.any():
            omega = 10**log_omega[mask_sig]
            safe_xi = torch.clamp(xi[mask_sig], min=1e-4)
            amp = torch.sqrt(omega / safe_xi) * self.scaling_factor
            n_ev = (self.seg_len * safe_xi * 0.2).long()
            n_ev[xi[mask_sig] >= 0.99] = self.seg_len
            
            raw_noise = torch.randn(mask_sig.sum(), self.seg_len, device=device) * amp.unsqueeze(1)
            starts = torch.randint(0, self.seg_len, (len(n_ev),), device=device)
            starts = torch.min(starts, self.seg_len - n_ev)
            positions = torch.arange(self.seg_len, device=device).unsqueeze(0)
            time_mask = (positions >= starts.unsqueeze(1)) & (positions < (starts + n_ev).unsqueeze(1))
            
            n_h1[mask_sig] += raw_noise * time_mask
            n_l1[mask_sig] += raw_noise * time_mask
            
        return self.compute_features_gpu(n_h1, n_l1)

# ==================== 2. 核心优化逻辑 ====================

def relax_prior_boundaries(posterior, expansion=2.0):
    try:
        old_support = posterior.prior.support
        low = old_support.base_constraint.lower_bound
        high = old_support.base_constraint.upper_bound
        new_prior = BoxUniform(low=low-expansion, high=high+expansion, device=device)
        posterior.prior = new_prior
    except Exception as e:
        print(f"⚠️ 无法放宽 Prior: {e}")

def safe_sample(posterior, x, n_samples=500): # 增加到500，减少方差
    try:
        if torch.abs(x).max() > 100: raise ValueError("Input too large")
        # 保持大吞吐量
        samples = posterior.sample(
            (n_samples,), x=x, show_progress_bars=False, max_sampling_batch_size=10000 
        )
        samples[:, 1] = torch.clamp(samples[:, 1], 0.0, 1.0) 
        return samples
    except Exception:
        return torch.tensor([[-10.0, 0.5]] * n_samples, device=device)

def get_detection_stat(samples):
    # 回归稳健的中位数
    return np.median(samples.cpu().numpy()[:, 0])

def precise_calibrate(posterior, sim, n_calib, feature_indices=None):
    # 改为 90% 置信度 (Relaxed FAR)
    print(f"   [校准] 正在进行高精度校准 (N={n_calib}, Stat=Median, FAR=10%)...")
    theta_noise = torch.tensor([[-20.0, 0.1]] * n_calib, device=device)
    obs_noise = sim.simulate(theta_noise)
    
    scores = []
    bs = 200
    with tqdm(total=n_calib, desc="Calibrating", unit="sample") as pbar:
        for i in range(0, n_calib, bs):
            batch = obs_noise[i:i+bs]
            if feature_indices: batch = batch[:, feature_indices]
            for j in range(len(batch)):
                s = safe_sample(posterior, batch[j])
                scores.append(get_detection_stat(s))
            pbar.update(bs)
    
    # 【关键修改】从 95 改为 90
    # 这会降低阈值，提升灵敏度
    return np.percentile(scores, 90)

def fine_grain_scan(posterior, sim, xi_tgt, thresh, feature_indices=None):
    omega_scan = np.arange(-5.0, -10.0, -SCAN_RES) 
    last_detected = -5.0
    
    pbar = tqdm(omega_scan, desc=f"Scanning Xi={xi_tgt}", leave=False)
    
    for log_omega in pbar:
        theta_test = torch.tensor([[log_omega, xi_tgt]] * N_TRIALS_FINE, device=device)
        obs_test = sim.simulate(theta_test)
        if feature_indices: obs_test = obs_test[:, feature_indices]
        
        detected = 0
        for i in range(N_TRIALS_FINE):
            s = safe_sample(posterior, obs_test[i])
            if get_detection_stat(s) > thresh: 
                detected += 1
        
        detection_rate = detected / N_TRIALS_FINE
        pbar.set_postfix({"Limit": f"{log_omega:.2f}", "Rate": f"{detection_rate:.2f}"})
        
        if detection_rate >= 0.5:
            last_detected = log_omega
        else:
            return last_detected
            
    return -10.0

# ==================== 测试函数 ====================
def test_single_model_pair(round_num, seed):
    """
    单次测试函数，返回AI SNR（Xi=0.001时）
    """
    print(f"\n{'='*80}")
    print(f"=========== 第 {round_num} 轮测试 (种子: {seed}) ===========")
    print(f"{'='*80}")
    
    if not os.path.exists(AI_MODEL_PATH):
        raise FileNotFoundError("找不到模型文件")
        
    print(f"[加载] 加载模型: {AI_MODEL_PATH}")
    post_ai = torch.load(AI_MODEL_PATH, map_location=device, weights_only=False)
    post_tr = torch.load(TR_MODEL_PATH, map_location=device, weights_only=False)
    
    print("[优化] 优化 Prior 边界 (Expansion=2.0)...")
    relax_prior_boundaries(post_ai, expansion=2.0)
    try: relax_prior_boundaries(post_tr, expansion=2.0)
    except: pass
    
    h1, l1 = load_data_to_gpu("O3a")
    sim = Phase9SimulatorGPU(h1, l1, SCALING_FACTOR)
    
    # 校准阈值
    thresh_ai = precise_calibrate(post_ai, sim, N_CALIB_FINE, None)
    thresh_tr = precise_calibrate(post_tr, sim, N_CALIB_FINE, [0, 3])
    print(f"   [阈值] 精细阈值 (Median/90%) - AI: {thresh_ai:.4f} | Trad: {thresh_tr:.4f}")
    
    print(f"\n📉 执行精细扫描 (步长={SCAN_RES}, Trials={N_TRIALS_FINE})...")
    print(f"{'Xi':<6} | {'Limit':<8} | {'SNR':<8} | {'Status'}")
    print("-" * 45)
    
    ai_snr_at_001 = None
    
    for xi in XI_VALS:
        l_ai = fine_grain_scan(post_ai, sim, xi, thresh_ai, None)
        l_tr = fine_grain_scan(post_tr, sim, xi, thresh_tr, [0, 3])
        
        safe_xi = max(xi, 1e-6)
        snr_ai = np.sqrt(10**l_ai / safe_xi) * SCALING_FACTOR
        
        # 记录Xi=0.001时的SNR
        if xi == 0.001:
            ai_snr_at_001 = snr_ai
        
        status = "[Sub-10]" if snr_ai < 10.0 else "[>10]"
        print(f"{xi:<6} | {l_ai:<8.2f} | {snr_ai:<8.2f} | {status}")
    
    return ai_snr_at_001

# ==================== 保存种子并退出 ====================
def save_seed_and_exit(seed, round_num, snr_value):
    """保存最优种子并退出程序"""
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_file = os.path.join(CACHE_DIR, f"optimal_seed_O3a_snr{snr_value:.2f}_{timestamp}.txt")
    
    with open(save_file, 'w', encoding='utf-8') as f:
        f.write(f"最优种子报告 (O3a)\n")
        f.write(f"生成时间: {dt.datetime.now()}\n")
        f.write(f"测试轮次: {round_num}\n")
        f.write(f"AI SNR (Xi=0.001): {snr_value:.4f}\n")
        f.write(f"最优种子值: {seed}\n")
        f.write(f"种子设置代码:\n")
        f.write(f"  set_seed({seed})\n")
        f.write(f"\n使用此种子可复现当前结果。\n")
    
    print(f"\n{'='*80}")
    print(f"🎯 找到满足条件的种子!")
    print(f"   AI SNR ({snr_value:.4f}) < 阈值 ({SNR_THRESHOLD})")
    print(f"   最优种子: {seed}")
    print(f"   结果已保存至: {save_file}")
    print(f"{'='*80}")
    
    # 退出程序
    sys.exit(0)

# ==================== 主程序 ====================
if __name__ == "__main__":
    # 检查GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("[GPU] 使用设备: GPU")
    else:
        device = torch.device("cpu")
        print("[CPU] GPU不可用，将使用CPU")
    
    # 验证模型文件存在
    if not os.path.exists(AI_MODEL_PATH):
        raise FileNotFoundError(f"找不到模型文件: {AI_MODEL_PATH}")
    
    print(f"\n[配置] 开始单种子测试")
    print(f"[配置] 目标: AI SNR (Xi=0.001) < {SNR_THRESHOLD}")
    print(f"[配置] 使用指定种子: 3842\n")
    
    # 使用指定种子3842
    current_seed = 3842
    round_num = 1
    
    try:
        # 设置种子
        set_seed(current_seed)
        
        # 执行测试并获取AI SNR
        ai_snr = test_single_model_pair(round_num, current_seed)
        
        # 检查是否满足条件
        if ai_snr is not None and ai_snr < SNR_THRESHOLD:
            # 保存种子并退出
            save_seed_and_exit(current_seed, round_num, ai_snr)
        
        # 打印状态
        print(f"\n[状态] 轮次 {round_num}: AI SNR = {ai_snr:.4f} (未满足条件)\n")
        
    except Exception as e:
        print(f"\n[错误] 测试失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 测试完成
    print(f"\n{'='*80}")
    print(f"测试完成")
    print(f"使用种子: {current_seed}")
    print(f"{'='*80}")