# @title Phase 9 (O3b): Ultra-Fast GPU Edition (With Model Saving)
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

print("=== P903b_GPU_UltraFast_WithSave.py 启动 ===")

# ==================== 配置区域 ====================
PT_DATA_DIR = r"C:\Users\20466\Desktop\新建文件夹 (6)\LIGO_Data_Cache"
CACHE_DIR = r"C:\Users\20466\Desktop\新建文件夹 (6)\LIGO_Data_Cache"
XI_TARGET = 0.001
SCALING_FACTOR = 1200.0 
N_TRAIN = 20000   
N_CALIB = 1000    
CUTOFF = 25.0  # ✅ 新增：高通滤波截止频率（Hz），O3b建议25~35

# 检查 GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"核心设备: {torch.cuda.get_device_name(0)}")
else:
    raise RuntimeError("错误: 未检测到 GPU! 此脚本需要 CUDA。")

# ==================== 1. 数据加载 ====================
def load_data_to_gpu(label="O3b"):
    expected_length = int(4096 * 2048.0)
    expected_length = int(4096 * 2048.0)
    filenames = [f"{label}_H1_1260834498_4.pt", f"{label}_L1_1260834498_4.pt", 
                 f"{label}_H1_1260834498_3.pt", f"{label}_L1_1260834498_3.pt"]
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

class Phase9SimulatorGPU:
    def __init__(self, h1_bg, l1_bg, scaling_factor=1300.0, cutoff=30.0):  # ✅ 新增cutoff参数
        self.h1_bg = h1_bg
        self.l1_bg = l1_bg
        self.scaling_factor = scaling_factor
        self.cutoff = cutoff  # ✅ 存储为实例变量
        self.target_fs = 2048.0
        self.seg_len = int(4.0 * self.target_fs)
        self.max_idx = len(h1_bg) - self.seg_len - 1

    # --- [新增] ---
    def apply_highpass_filter(self, x):  # ✅ 移除cutoff参数，使用self.cutoff
        """O3b 必须去除 <cutoff Hz 的噪声"""
        n = x.shape[-1]
        freq = torch.fft.rfftfreq(n, d=1/self.target_fs, device=device)
        fft_x = torch.fft.rfft(x, dim=-1)
        mask = (freq > self.cutoff).float()  # ✅ 使用实例变量
        return torch.fft.irfft(fft_x * mask, n=n, dim=-1)

    def compute_features_gpu(self, h1, l1):
        # 注意：这里的输入已经是经过滤波和归一化的了
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
        
        # log10(var) 对于 O3b 来说，可能需要更鲁棒的功率估计，但暂时保持原样
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
        
        # === 修改重点 1: 先滤波 ===
        n_h1 = self.apply_highpass_filter(n_h1)
        n_l1 = self.apply_highpass_filter(n_l1)

        # === 修改重点 2: 鲁棒归一化 ===
        # 使用 IQR (0.75 - 0.25 分位) 代替 std
        def robust_norm(x):
            q75 = torch.nanquantile(x, 0.75, dim=1, keepdim=True)
            q25 = torch.nanquantile(x, 0.25, dim=1, keepdim=True)
            iqr = q75 - q25
            median = torch.nanquantile(x, 0.5, dim=1, keepdim=True)
            return (x - median) / (iqr / 1.349 + 1e-15)

        n_h1 = robust_norm(n_h1)
        n_l1 = robust_norm(n_l1)
        
        mask_sig = (log_omega > -15.0)
        if mask_sig.any():
            omega = 10**log_omega[mask_sig]
            safe_xi = torch.clamp(xi[mask_sig], min=1e-4)
            # Scaling Factor 可能需要针对 O3b 微调，如果 SNR 依然低，尝试增加它
            amp = torch.sqrt(omega / safe_xi) * self.scaling_factor 
            
            n_ev = (self.seg_len * safe_xi * 0.2).long()
            n_ev[xi[mask_sig] >= 0.99] = self.seg_len
            
            # 生成信号噪声
            raw_noise = torch.randn(mask_sig.sum(), self.seg_len, device=device) * amp.unsqueeze(1)
            
            # 信号也需要经过同样的滤波！这是物理一致性
            # (虽然白噪声谱是平的，但为了匹配背景的处理方式，建议加上)
            raw_noise = self.apply_highpass_filter(raw_noise)

            starts = torch.randint(0, self.seg_len, (len(n_ev),), device=device)
            starts = torch.min(starts, self.seg_len - n_ev)
            
            positions = torch.arange(self.seg_len, device=device).unsqueeze(0)
            time_mask = (positions >= starts.unsqueeze(1)) & (positions < (starts + n_ev).unsqueeze(1))
            
            from scipy.signal.windows import tukey
            window_cpu = torch.from_numpy(tukey(self.seg_len, alpha=0.1)).float().to(device)
            
            n_h1[mask_sig] += raw_noise * time_mask * window_cpu
            n_l1[mask_sig] += raw_noise * time_mask * window_cpu
            
        return self.compute_features_gpu(n_h1, n_l1)


# ==================== 3. 辅助函数 ====================
def generate_training_data(sim, prior, n_samples):
    batch_size = 1000
    theta_all, x_all = [], []
    print(f"GPU正在生成 {n_samples} 条模拟数据...")
    # 添加进度条
    for i in tqdm(range(0, n_samples, batch_size), desc="[DEBUG] 生成训练数据", leave=True):
        batch_theta = prior.sample((batch_size,)).to(device)
        batch_x = sim.simulate(batch_theta)
        theta_all.append(batch_theta)
        x_all.append(batch_x)
        # 每10个批次打印一次进度
        if (i // batch_size) % 10 == 0:
            print(f"[DEBUG] 已生成 {i + batch_size} / {n_samples} 条数据")
    return torch.cat(theta_all), torch.cat(x_all)

def safe_sample(posterior, x, n_samples=200):
    try:
        return posterior.sample((n_samples,), x=x, show_progress_bars=False)
    except:
        return torch.tensor([[10.0, 0.5]] * n_samples, device=device)

def fast_calibrate(posterior, sim, n, feature_indices=None):
    bs = 100
    print(f"[DEBUG] 开始CFAR校准，n={n}，批大小={bs}...")
    theta_noise = torch.tensor([[-20.0, 0.1]] * n, device=device)
    print("[DEBUG] 正在生成噪声观测数据...")
    obs_noise = sim.simulate(theta_noise)
    scores = []
    total_batches = (n + bs - 1) // bs
    
    for batch_idx in tqdm(range(0, n, bs), desc="[DEBUG] CFAR校准进度", leave=True):
        batch = obs_noise[batch_idx:batch_idx+bs]
        if feature_indices: 
            batch = batch[:, feature_indices]
            print(f"[DEBUG] 应用特征索引: {feature_indices}")
        
        for sample_idx in range(len(batch)):
            s = safe_sample(posterior, batch[sample_idx])
            scores.append(s[:, 0].mean().item())
        
        # 每处理5个批次打印一次进度
        if (batch_idx // bs + 1) % 5 == 0:
            print(f"[DEBUG] 已完成 {batch_idx + bs} / {n} 个样本的校准")
    
    print("[DEBUG] 计算90百分位阈值...")
    return np.percentile(scores, 90)  # FAR=10%，阈值更低，SNR降低

def find_limit(posterior, sim, xi_tgt, thresh, feature_indices=None):
    print(f"[DEBUG] 开始寻找极限值，xi_tgt={xi_tgt}，阈值={thresh}...")
    low, high = -12.0, -1.0
    n_trials = 20
    iteration = 0
    
    while (high - low) > 0.2:
        iteration += 1
        mid = (high + low) / 2.0
        print(f"[DEBUG] 迭代 {iteration}: 测试值={mid:.4f}，当前范围 [{low:.4f}, {high:.4f}]")
        
        theta_test = torch.tensor([[mid, xi_tgt]] * n_trials, device=device)
        print(f"[DEBUG] 正在生成 {n_trials} 个测试观测...")
        obs_test = sim.simulate(theta_test)
        if feature_indices: 
            obs_test = obs_test[:, feature_indices]
            print(f"[DEBUG] 应用特征索引: {feature_indices}")
        
        detected = 0
        for i in range(n_trials):
            s = safe_sample(posterior, obs_test[i])
            if s[:, 0].mean() > thresh: 
                detected += 1
        
        print(f"[DEBUG] 检测到 {detected} / {n_trials} 个信号")
        if detected >= (n_trials / 2): 
            high = mid
            print(f"[DEBUG] 降低上限至 {high:.4f}")
        else: 
            low = mid
            print(f"[DEBUG] 提高下限至 {low:.4f}")
    
    print(f"[DEBUG] 寻找极限值完成，结果={high:.4f}")
    return high

# ==================== 主流程 ====================
if __name__ == "__main__":
    # 重复运行5次
    for run in range(1, 6):
        print(f"\n" + "="*100)
        print(f"==================== 第 {run} 次运行 ====================")
        print("="*100)
        print(f"[DEBUG] 开始执行第 {run} 次运行...")
        
        # 1. 数据加载
        print("[DEBUG] 正在加载数据...")
        h1_gpu, l1_gpu = load_data_to_gpu("O3b")
        print(f"[DEBUG] 数据加载完成，H1数据长度: {len(h1_gpu)}, L1数据长度: {len(l1_gpu)}")
        
        # 2. ✅ 修改：初始化模拟器时传入cutoff参数
        print("[DEBUG] 正在初始化模拟器...")
        sim_gpu = Phase9SimulatorGPU(h1_gpu, l1_gpu, scaling_factor=SCALING_FACTOR, cutoff=CUTOFF)
        print(f"[DEBUG] 模拟器初始化完成，cutoff={CUTOFF}Hz")
        
        # 3. 设置先验分布
        print("[DEBUG] 正在设置先验分布...")
        prior = BoxUniform(low=torch.tensor([-13.0, 0.001], device=device), 
                           high=torch.tensor([5.0, 1.0], device=device))
        print("[DEBUG] 先验分布设置完成")
        
        # 4. 生成训练数据
        print("[DEBUG] 正在生成训练数据...")
        theta_tr, x_tr = generate_training_data(sim_gpu, prior, N_TRAIN)
        print(f"[DEBUG] 训练数据生成完成，样本数: {len(theta_tr)}")
        
        # 5. 训练ING-Net模型
        print("[DEBUG] 正在训练ING-Net模型...")
        inf_ai = SNPE(prior=prior, density_estimator="maf", device=str(device))
        inf_ai.append_simulations(theta_tr, x_tr)
        print("[DEBUG] 正在执行ING-Net训练...")
        post_ai = inf_ai.build_posterior(inf_ai.train(show_train_summary=False))
        print("[DEBUG] ING-Net模型训练完成")
        
        # 6. 训练Traditional模型
        print("[DEBUG] 正在训练Traditional模型...")
        inf_tr = SNPE(prior=prior, density_estimator="maf", device=str(device))
        inf_tr.append_simulations(theta_tr, x_tr[:, [0, 3]])
        print("[DEBUG] 正在执行Traditional训练...")
        post_tr = inf_tr.build_posterior(inf_tr.train(show_train_summary=False))
        print("[DEBUG] Traditional模型训练完成")
        
        # 7. 保存模型
        print("\n[DEBUG] 正在保存模型文件...")
        model_dir = os.path.join(CACHE_DIR, "models")
        os.makedirs(model_dir, exist_ok=True)
        
        # 添加时间戳（精确到毫秒），防止覆盖原有文件
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        
        path_ai = os.path.join(model_dir, f"ing_net_o3b_gpu_{timestamp}.pt")
        path_tr = os.path.join(model_dir, f"trad_model_o3b_gpu_{timestamp}.pt")
        
        print(f"[DEBUG] 正在保存ING-Net模型至: {path_ai}")
        torch.save(post_ai, path_ai)
        print(f"[DEBUG] 正在保存Traditional模型至: {path_tr}")
        torch.save(post_tr, path_tr)
        print(f"[DEBUG] 模型已保存至: {model_dir}")
        print(f"[DEBUG] - ING-Net: ing_net_o3b_gpu_{timestamp}.pt")
        print(f"[DEBUG] - Traditional: trad_model_o3b_gpu_{timestamp}.pt")
        
        # 8. CFAR校准
        print(f"\n[DEBUG] 快速CFAR校准 (N={N_CALIB})...")
        print("[DEBUG] 正在校准ING-Net阈值...")
        thresh_ai = fast_calibrate(post_ai, sim_gpu, N_CALIB, None)
        print(f"[DEBUG] ING-Net阈值校准完成: {thresh_ai:.4f}")
        
        print("[DEBUG] 正在校准Traditional阈值...")
        thresh_tr = fast_calibrate(post_tr, sim_gpu, N_CALIB, [0, 3])
        print(f"[DEBUG] Traditional阈值校准完成: {thresh_tr:.4f}")
        
        print(f"[DEBUG] 阈值校准结果: ING-Net={thresh_ai:.4f} | Traditional={thresh_tr:.4f}")
        
        # 9. 灵敏度扫描
        print("\n[DEBUG] 开始扫描灵敏度...")
        xi_vals = [0.001, 0.01, 0.1, 0.5, 1.0]
        print(f"{'Xi':<6} | {'AI Limit':<10} | {'Trad Limit':<10} | {'Advantage'}")
        print("-" * 55)
        
        res_ai, res_tr = [], []
        for xi in tqdm(xi_vals, desc="[DEBUG] 灵敏度扫描进度"):
            print(f"[DEBUG] 正在处理Xi={xi}...")
            
            print(f"[DEBUG] 正在计算ING-Net极限值...")
            l_ai = find_limit(post_ai, sim_gpu, xi, thresh_ai, None)
            print(f"[DEBUG] ING-Net极限值计算完成: {l_ai:.2f}")
            
            print(f"[DEBUG] 正在计算Traditional极限值...")
            l_tr = find_limit(post_tr, sim_gpu, xi, thresh_tr, [0, 3])
            print(f"[DEBUG] Traditional极限值计算完成: {l_tr:.2f}")
            
            res_ai.append(l_ai)
            res_tr.append(l_tr)
            diff = l_tr - l_ai
            adv = "AI Win" if l_ai < l_tr else "Trad Win"
            print(f"{xi:<6} | {l_ai:<10.2f} | {l_tr:<10.2f} | {diff:+.2f} ({adv})")
        
        # 10. 保存结果
        print(f"\n[DEBUG] 正在保存结果文件...")
        results_path = os.path.join(CACHE_DIR, f"o3b_gpu_results_{timestamp}.pt")
        torch.save({"xi": xi_vals, "ai": res_ai, "trad": res_tr}, results_path)
        print(f"[DEBUG] 结果文件保存完成: {results_path}")
        
        print(f"\n[DEBUG] 第 {run} 次运行完成！结果与模型均已保存。")
        print(f"[DEBUG] - 结果文件: o3b_gpu_results_{timestamp}.pt")
        print(f"[DEBUG] - ING-Net模型: ing_net_o3b_gpu_{timestamp}.pt")
        print(f"[DEBUG] - Traditional模型: trad_model_o3b_gpu_{timestamp}.pt")
    
    print(f"\n" + "="*100)
    print("==================== 所有运行完成 ====================")
    print("="*100)
    print("[DEBUG] 5次运行已全部完成！所有结果与模型均已保存。")