# @title Phase 7 (Final): Segment-by-Segment Analysis (.pt Saving)
# @markdown **Updates:**
# @markdown 1. Saves all models as **.pt** files using `torch.save`.
# @markdown 2. Performs per-segment training and analysis.
# @markdown 3. Auto-skips invalid files.
# @markdown 4. Groups H1 and L1 files by GPS time.

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, pearsonr
from sbi.inference import SNPE, simulate_for_sbi
from sbi.utils import BoxUniform
from tqdm import tqdm
import os
import glob
import re

# ==========================================
# 0. Setup Paths
# ==========================================

# 创建新的结果目录
RESULTS_DIR = r"C:\Users\20466\Desktop\新建文件夹 (6)\ING_Net_Segment_Results_O3A"
os.makedirs(RESULTS_DIR, exist_ok=True)

LOCAL_DATA_PATH = r"C:\Users\20466\Desktop\新建文件夹 (6)\ligo_data\processed\03A数据"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device.upper()} | Saving .pt models to: {RESULTS_DIR}")

# ==========================================
# 1. Load Files
# ==========================================

# 只处理03A文件
data_files = [f for f in glob.glob(os.path.join(LOCAL_DATA_PATH, "*.pt")) if "O3a" in f or "03A" in f]
print(f"Found {len(data_files)} O3A data files.")

if not data_files:
    raise FileNotFoundError("No .pt files found in data directory!")

# ==========================================
# 2. Helper Functions
# ==========================================

def group_files_by_gps(data_files):
    """
    按GPS时间对H1和L1文件进行分组
    """
    gps_groups = {}
    for filepath in data_files:
        filename = os.path.basename(filepath)
        gps_match = re.search(r'(\d{9,10})', filename)
        if gps_match:
            gps_time = gps_match.group(1)
            if gps_time not in gps_groups:
                gps_groups[gps_time] = {'h1': None, 'l1': None}
            if 'H1' in filename:
                gps_groups[gps_time]['h1'] = filepath
            elif 'L1' in filename:
                gps_groups[gps_time]['l1'] = filepath
    return gps_groups

CURRENT_SEGMENT_DATA = None 

def get_current_segment_noise(seg_len):
    global CURRENT_SEGMENT_DATA
    total_len = CURRENT_SEGMENT_DATA.shape[1]
    if total_len <= seg_len: start_idx = 0
    else: start_idx = np.random.randint(0, total_len - seg_len)
        
    n_h1 = CURRENT_SEGMENT_DATA[0, start_idx : start_idx + seg_len]
    n_l1 = CURRENT_SEGMENT_DATA[1, start_idx : start_idx + seg_len]
    
    # Normalization
    n_h1 = (n_h1 - np.mean(n_h1)) / (np.std(n_h1) + 1e-30)
    n_l1 = (n_l1 - np.mean(n_l1)) / (np.std(n_l1) + 1e-30)
    return n_h1, n_l1

def fast_simulator(theta_batch):
    if isinstance(theta_batch, np.ndarray): theta_batch = torch.from_numpy(theta_batch)
    batch_stats = []
    
    for theta in theta_batch:
        log10_omega, xi = theta[0].item(), theta[1].item()
        seg_len = int(4.0 * 2048)
        
        n_h1, n_l1 = get_current_segment_noise(seg_len)
            
        omega = 10**log10_omega
        amp = np.sqrt(omega / max(xi, 1e-4)) * 3000.0
        n_events = int(seg_len * xi * 0.2)
        
        d_h1, d_l1 = n_h1.copy(), n_l1.copy()
        if n_events > 0:
            idx = np.random.randint(0, seg_len, n_events)
            burst = np.random.normal(0, amp, n_events)
            d_h1[idx] += burst
            d_l1[idx] += burst
            
        cc, _ = pearsonr(d_h1, d_l1)
        k_h1, k_l1 = kurtosis(d_h1), kurtosis(d_l1)
        p = np.log10(np.var(d_h1)*np.var(d_l1) + 1e-30)
        
        batch_stats.append(torch.tensor([cc, np.log1p(abs(k_h1)), np.log1p(abs(k_l1)), p], dtype=torch.float32))
        
    return torch.stack(batch_stats)

prior = BoxUniform(low=torch.tensor([-9.0, 0.001], device=device), high=torch.tensor([-5.0, 1.0], device=device))

# ==========================================
# 3. Main Loop
# ==========================================

# 按GPS时间分组文件
gps_groups = group_files_by_gps(data_files)
print(f"Found {len(gps_groups)} GPS time groups.")

# --- Main Loop ---
for group_idx, (gps_time, files) in enumerate(gps_groups.items()):
    total_groups = len(gps_groups)
    print(f"\n[{group_idx+1}/{total_groups}] Analyzing Segment: {gps_time} ...")
    
    try:
        # 检查是否同时有H1和L1文件
        if not files['h1'] or not files['l1']:
            print(f"  -> Skip (Missing H1 or L1 file)")
            continue
        
        # 加载H1和L1数据
        h1_data = torch.load(files['h1'])
        l1_data = torch.load(files['l1'])
        
        if isinstance(h1_data, torch.Tensor): h1_data = h1_data.numpy()
        if isinstance(l1_data, torch.Tensor): l1_data = l1_data.numpy()
        
        # 确保数据长度匹配
        min_len = min(len(h1_data), len(l1_data))
        if min_len < 8192:
            print(f"  -> Skip (Data too short)")
            continue
        
        # 组合成(2, N)的形状
        combined_data = np.vstack([h1_data[:min_len], l1_data[:min_len]])
        CURRENT_SEGMENT_DATA = combined_data
        
        print(f"  -> Loaded H1 and L1 data (length: {min_len})")
        
        # Simulate & Train
        print(f"  -> Simulating training data...")
        theta_train, x_train = simulate_for_sbi(fast_simulator, proposal=prior, num_simulations=10000) 
        
        # 1. ING-Net
        print(f"  -> Training ING-Net...")
        inf_ai = SNPE(prior=prior, density_estimator="maf", device=device)
        inf_ai.append_simulations(theta_train, x_train)
        de_ai = inf_ai.train(show_train_summary=False, training_batch_size=1000)
        # 使用MCMC采样方法
        post_ai = inf_ai.build_posterior(de_ai, sample_with='mcmc')
        
        # === 保存 ING-Net 为 .pt ===
        torch.save(post_ai, os.path.join(RESULTS_DIR, f"ingnet_{gps_time}.pt"))
        
        # 2. Traditional
        print(f"  -> Training Traditional Baseline...")
        x_train_trad = x_train[:, [0, 3]]
        inf_trad = SNPE(prior=prior, density_estimator="maf", device=device)
        inf_trad.append_simulations(theta_train, x_train_trad)
        de_trad = inf_trad.train(show_train_summary=False, training_batch_size=1000)
        # 使用MCMC采样方法
        post_trad = inf_trad.build_posterior(de_trad, sample_with='mcmc')
        
        # === 保存 Traditional 为 .pt ===
        torch.save(post_trad, os.path.join(RESULTS_DIR, f"trad_{gps_time}.pt"))
        
        # Evaluate & Plot
        print(f"  -> Evaluating...")
        ul_ai, ul_trad = [], []
        valid_evaluations = 0
        
        for _ in range(50):
            try:
                seg_len = int(4.0 * 2048)
                n_h1, n_l1 = get_current_segment_noise(seg_len)
                
                # 数据质量检查
                if np.std(n_h1) < 1e-6 or np.std(n_l1) < 1e-6:
                    continue
                
                cc, _ = pearsonr(n_h1, n_l1)
                # 处理NaN或无穷值
                if np.isnan(cc) or np.isinf(cc):
                    cc = 0.0
                
                k_h1, k_l1 = kurtosis(n_h1), kurtosis(n_l1)
                # 处理NaN或无穷值
                if np.isnan(k_h1) or np.isinf(k_h1):
                    k_h1 = 0.0
                if np.isnan(k_l1) or np.isinf(k_l1):
                    k_l1 = 0.0
                k_h1, k_l1 = np.log1p(abs(k_h1)), np.log1p(abs(k_l1))
                
                p = np.log10(np.var(n_h1)*np.var(n_l1) + 1e-30)
                
                obs = torch.tensor([cc, k_h1, k_l1, p], dtype=torch.float32, device=device)
                
                # 使用MCMC采样
                s_ai = post_ai.sample((1000,), x=obs, show_progress_bars=False)
                s_trad = post_trad.sample((1000,), x=obs[[0, 3]], show_progress_bars=False)
                
                ul_ai.append(np.percentile(s_ai[:,0].cpu().numpy(), 95))
                ul_trad.append(np.percentile(s_trad[:,0].cpu().numpy(), 95))
                valid_evaluations += 1
                
                if valid_evaluations >= 20:  # 只需要20个有效评估
                    break
                    
            except Exception as e:
                continue
        
        if valid_evaluations < 5:
            print(f"  -> Skip (Not enough valid evaluations)")
            continue
        
        avg_ul_ai = np.mean(ul_ai)
        avg_ul_trad = np.mean(ul_trad)
        
        plt.figure(figsize=(8, 5))
        plt.hist(s_ai[:, 0].cpu().numpy(), bins=30, density=True, alpha=0.5, color='blue', label='ING-Net')
        plt.hist(s_trad[:, 0].cpu().numpy(), bins=30, density=True, alpha=0.5, color='orange', label='Traditional')
        plt.axvline(avg_ul_ai, color='blue', linestyle='--')
        plt.axvline(avg_ul_trad, color='orange', linestyle='--')
        plt.title(f'Segment {gps_time}\nING-Net Limit: 10^{{avg_ul_ai:.2f}} (Trad: 10^{{avg_ul_trad:.2f}})')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"plot_{gps_time}.png"))
        plt.close()
        
        print(f"  -> Saved .pt models and plot for {gps_time}")
        
    except Exception as e:
        print(f"  -> Error: {e}")
        continue

print("\nAll done!")
