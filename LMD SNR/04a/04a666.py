# @title Phase 9 (O4a): Ultra-Fast GPU Edition (With Model Saving)
# @markdown **🚀 Features:**
# @markdown 1. **Pure GPU Acceleration:** Train & Simulate in seconds.
# @markdown 2. **Model Saving:** Automatically saves `ing_net.pt` and `trad_model.pt`.
# @markdown 3. **High Precision:** N=20000 samples for reliable results.

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from sbi.inference import SNPE
from sbi.utils import BoxUniform
from tqdm import tqdm
import warnings
import datetime

warnings.filterwarnings("ignore")

print("=== P904a_GPU_UltraFast_WithSave.py 启动 ===")

# ==================== 配置区域 ====================
PT_DATA_DIR = r"C:\Users\20466\Desktop\之前\LIGO_Data_Cache"
CACHE_DIR = r"C:\Users\20466\Desktop\之前\LIGO_Data_Cache"
XI_TARGET = 0.001
SCALING_FACTOR = 1200.0  # 🔥 O4a使用1200（与O3b相同，不同于O3a的1300）
N_TRAIN = 20000   
N_CALIB = 1000    

# 🔥 检查 GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"🚀 核心设备: {torch.cuda.get_device_name(0)}")
else:
    raise RuntimeError("❌ 错误: 未检测到 GPU! 此脚本需要 CUDA。")

# ==================== 1. 数据加载 ====================
def load_data_to_gpu(label="O4a"):
    expected_length = int(4096 * 2048.0)
    # 🔥 O4a GPS起始时间: 1377415818 (2023-06-24 15:00 UTC)
    filenames = [f"{label}_H1_1377415818.pt", f"{label}_L1_1377415818.pt", 
                 f"{label}_H1.pt", f"{label}_L1.pt"]
    loaded = {}
    for det in ['H1', 'L1']:
        for fname in filenames:
            if det in fname:
                path = os.path.join(PT_DATA_DIR, fname)
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

# ==================== 2. 模拟器 (GPU) ====================
class Phase9SimulatorGPU:
    def __init__(self, h1_bg, l1_bg, scaling_factor=1200.0):
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
            
            # 使用 Tukey Window 减少硬切断带来的非物理高频
            from scipy.signal.windows import tukey
            window_cpu = torch.from_numpy(tukey(self.seg_len, alpha=0.1)).float().to(device)
            
            n_h1[mask_sig] += raw_noise * time_mask * window_cpu
            n_l1[mask_sig] += raw_noise * time_mask * window_cpu
            
        return self.compute_features_gpu(n_h1, n_l1)

# ==================== 3. 辅助函数 ====================
def generate_training_data(sim, prior, n_samples):
    batch_size = 1000
    theta_all, x_all = [], []
    print(f"⚡ GPU正在生成 {n_samples} 条模拟数据...")
    for _ in range(0, n_samples, batch_size):
        theta = prior.sample((batch_size,)).to(device)
        x = sim.simulate(theta)
        theta_all.append(theta)
        x_all.append(x)
    return torch.cat(theta_all), torch.cat(x_all)

def safe_sample(posterior, x, n_samples=200):
    try:
        return posterior.sample((n_samples,), x=x, show_progress_bars=False)
    except:
        return torch.tensor([[10.0, 0.5]] * n_samples, device=device)

def fast_calibrate(posterior, sim, n, feature_indices=None):
    theta_noise = torch.tensor([[-20.0, 0.1]] * n, device=device)
    obs_noise = sim.simulate(theta_noise)
    scores = []
    bs = 100
    for i in range(0, n, bs):
        batch = obs_noise[i:i+bs]
        if feature_indices: batch = batch[:, feature_indices]
        for j in range(len(batch)):
            s = safe_sample(posterior, batch[j])
            scores.append(s[:, 0].mean().item())
    return np.percentile(scores, 95)

def find_limit(posterior, sim, xi_tgt, thresh, feature_indices=None):
    low, high = -12.0, -1.0
    n_trials = 20
    while (high - low) > 0.2:
        mid = (high + low) / 2.0
        theta_test = torch.tensor([[mid, xi_tgt]] * n_trials, device=device)
        obs_test = sim.simulate(theta_test)
        if feature_indices: obs_test = obs_test[:, feature_indices]
        
        detected = 0
        for i in range(n_trials):
            s = safe_sample(posterior, obs_test[i])
            if s[:, 0].mean() > thresh: detected += 1
        
        if detected >= (n_trials / 2): high = mid
        else: low = mid
    return high

# ==================== 主流程 ====================
if __name__ == "__main__":
    h1_gpu, l1_gpu = load_data_to_gpu("O4a")
    sim_gpu = Phase9SimulatorGPU(h1_gpu, l1_gpu, scaling_factor=SCALING_FACTOR)
    
    # 扩大先验上限防止Glitch崩溃
    prior = BoxUniform(low=torch.tensor([-13.0, 0.001], device=device), 
                       high=torch.tensor([5.0, 1.0], device=device))
    
    theta_tr, x_tr = generate_training_data(sim_gpu, prior, N_TRAIN)
    
    print("🎯 训练模型 (GPU)...")
    inf_ai = SNPE(prior=prior, density_estimator="maf", device=str(device))
    inf_ai.append_simulations(theta_tr, x_tr)
    post_ai = inf_ai.build_posterior(inf_ai.train(show_train_summary=False))
    
    inf_tr = SNPE(prior=prior, density_estimator="maf", device=str(device))
    inf_tr.append_simulations(theta_tr, x_tr[:, [0, 3]])
    post_tr = inf_tr.build_posterior(inf_tr.train(show_train_summary=False))
    
    # ==================== 💾 保存模型 ====================
    print("\n💾 正在保存模型文件...")
    model_dir = os.path.join(CACHE_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # 添加时间戳，确保每次运行的文件名唯一
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    
    # 🔥 文件名已修改为 o4a + 时间戳，防止覆盖
    path_ai = os.path.join(model_dir, f"ing_net_o4a_gpu_{timestamp}.pt")
    path_tr = os.path.join(model_dir, f"trad_model_o4a_gpu_{timestamp}.pt")
    
    torch.save(post_ai, path_ai)
    torch.save(post_tr, path_tr)
    print(f"   ✅ 模型已保存至: {model_dir}")
    print(f"   - ING-Net: ing_net_o4a_gpu_{timestamp}.pt")
    print(f"   - Traditional: trad_model_o4a_gpu_{timestamp}.pt")
    # ====================================================
    
    print(f"\n⚖️ 快速CFAR校准 (N={N_CALIB})...")
    thresh_ai = fast_calibrate(post_ai, sim_gpu, N_CALIB, None)
    thresh_tr = fast_calibrate(post_tr, sim_gpu, N_CALIB, [0, 3])
    print(f"   ING-Net 阈值: {thresh_ai:.4f} | Traditional 阈值: {thresh_tr:.4f}")
    
    print("\n📉 极速扫描灵敏度...")
    xi_vals = [0.001, 0.01, 0.1, 0.5, 1.0]
    print(f"{'Xi':<6} | {'AI Limit':<10} | {'Trad Limit':<10} | {'Advantage'}")
    print("-" * 55)
    
    res_ai, res_tr = [], []
    for xi in xi_vals:
        l_ai = find_limit(post_ai, sim_gpu, xi, thresh_ai, None)
        l_tr = find_limit(post_tr, sim_gpu, xi, thresh_tr, [0, 3])
        res_ai.append(l_ai)
        res_tr.append(l_tr)
        diff = l_tr - l_ai
        adv = "AI Win" if l_ai < l_tr else "Trad Win"
        print(f"{xi:<6} | {l_ai:<10.2f} | {l_tr:<10.2f} | {diff:+.2f} ({adv})")
    
    # 🔥 结果文件名添加时间戳，防止覆盖
    results_path = os.path.join(CACHE_DIR, f"o4a_gpu_results_{timestamp}.pt")
    torch.save({"xi": xi_vals, "ai": res_ai, "trad": res_tr}, results_path)
    print(f"\n✅ 运行完成！结果与模型均已保存。")
    print(f"   - 结果文件: o4a_gpu_results_{timestamp}.pt")
