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
# 0. Setup Paths (路径设置)
# ==========================================

# 创建新的结果目录，用于保存训练的模型和图表
RESULTS_DIR = r"C:\Users\20466\Desktop\新建文件夹 (6)\ING_Net_Segment_Results_O3A"
os.makedirs(RESULTS_DIR, exist_ok=True)

# O3a数据集路径（根据记忆#1，O3a的SCALING_FACTOR应为1300）
LOCAL_DATA_PATH = r"C:\Users\20466\Desktop\新建文件夹 (6)\ligo_data\processed\03A数据"

# 自动检测GPU/CPU设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device.upper()} | Saving .pt models to: {RESULTS_DIR}")

# ==========================================
# 1. Load Files (加载数据文件)
# ==========================================

# 只处理O3a相关的.pt文件（根据记忆#1，优先处理O3a数据）
data_files = [f for f in glob.glob(os.path.join(LOCAL_DATA_PATH, "*.pt")) if "O3a" in f or "03A" in f]
print(f"Found {len(data_files)} O3A data files.")

# 如果没有找到文件则报错
if not data_files:
    raise FileNotFoundError("No .pt files found in data directory!")

# ==========================================
# 2. Helper Functions (辅助函数定义)
# ==========================================

def group_files_by_gps(data_files):
    """
    按GPS时间对H1和L1文件进行分组
    参数:
        data_files: 数据文件路径列表
    返回:
        gps_groups: 字典，键为GPS时间，值为包含'h1'和'l1'文件路径的字典
    """
    gps_groups = {}
    for filepath in data_files:
        filename = os.path.basename(filepath)
        # 使用正则表达式提取GPS时间戳（9-10位数字）
        gps_match = re.search(r'(\d{9,10})', filename)
        if gps_match:
            gps_time = gps_match.group(1)
            if gps_time not in gps_groups:
                gps_groups[gps_time] = {'h1': None, 'l1': None}
            # 根据文件名中的探测器标识分类
            if 'H1' in filename:
                gps_groups[gps_time]['h1'] = filepath
            elif 'L1' in filename:
                gps_groups[gps_time]['l1'] = filepath
    return gps_groups

# 全局变量，存储当前处理的数据段，供模拟器使用
CURRENT_SEGMENT_DATA = None 

def get_current_segment_noise(seg_len):
    """
    从当前数据段中随机提取指定长度的噪声样本
    参数:
        seg_len: 需要的样本长度（采样点数）
    返回:
        n_h1, n_l1: 归一化后的H1和L1探测器噪声数据
    """
    global CURRENT_SEGMENT_DATA
    total_len = CURRENT_SEGMENT_DATA.shape[1]
    
    # 随机选择起始索引，确保不超出数据范围
    if total_len <= seg_len: 
        start_idx = 0
    else: 
        start_idx = np.random.randint(0, total_len - seg_len)
        
    # 提取数据
    n_h1 = CURRENT_SEGMENT_DATA[0, start_idx : start_idx + seg_len]
    n_l1 = CURRENT_SEGMENT_DATA[1, start_idx : start_idx + seg_len]
    
    # Z-score归一化：减去均值，除以标准差（防止除零加极小值）
    n_h1 = (n_h1 - np.mean(n_h1)) / (np.std(n_h1) + 1e-30)
    n_l1 = (n_l1 - np.mean(n_l1)) / (np.std(n_l1) + 1e-30)
    return n_h1, n_l1

def fast_simulator(theta_batch):
    """
    SBI快速模拟器：根据参数theta生成模拟数据统计特征
    模拟引力波爆发信号叠加在噪声上的过程
    
    参数:
        theta_batch: 批次参数 [log10(omega), xi]，omega为能量密度，xi为事件率
    返回:
        统计特征张量 [相关系数, log(1+|H1峰度|), log(1+|L1峰度|), log(功率积)]
    """
    # 确保输入是PyTorch张量
    if isinstance(theta_batch, np.ndarray): 
        theta_batch = torch.from_numpy(theta_batch)
    
    batch_stats = []
    
    # 遍历批次中的每个参数集
    for theta in theta_batch:
        log10_omega = theta[0].item()  # 对数能量密度
        xi = theta[1].item()           # 爆发事件率
        
        # 计算段长度：4秒 * 2048Hz采样率 = 8192个采样点
        seg_len = int(4.0 * 2048)
        
        # 获取当前段的噪声
        n_h1, n_l1 = get_current_segment_noise(seg_len)
            
        # 计算爆发信号幅度（根据omega和xi的物理关系）
        omega = 10**log10_omega
        amp = np.sqrt(omega / max(xi, 1e-4)) * 3000.0
        n_events = int(seg_len * xi * 0.2)  # 段内爆发事件数
        
        # 复制噪声，准备添加信号
        d_h1, d_l1 = n_h1.copy(), n_l1.copy()
        
        # 在随机位置添加高斯爆发信号（H1和L1添加相同信号，模拟真实引力波）
        if n_events > 0:
            idx = np.random.randint(0, seg_len, n_events)
            burst = np.random.normal(0, amp, n_events)
            d_h1[idx] += burst
            d_l1[idx] += burst
            
        # 计算统计特征
        cc, _ = pearsonr(d_h1, d_l1)  # 皮尔逊相关系数
        k_h1 = kurtosis(d_h1)         # H1峰度（衡量尾部厚度）
        k_l1 = kurtosis(d_l1)         # L1峰度
        p = np.log10(np.var(d_h1)*np.var(d_l1) + 1e-30)  # 对数功率积
        
        # 将特征组合为张量（对峰度取log1p压缩动态范围）
        stats = torch.tensor([cc, np.log1p(abs(k_h1)), np.log1p(abs(k_l1)), p], dtype=torch.float32)
        batch_stats.append(stats)
        
    return torch.stack(batch_stats)

# 定义参数先验分布：log10(omega) ∈ [-9, -5], xi ∈ [0.001, 1.0]
prior = BoxUniform(
    low=torch.tensor([-9.0, 0.001], device=device), 
    high=torch.tensor([-5.0, 1.0], device=device)
)

# ==========================================
# 3. Main Loop (主循环)
# ==========================================

# 按GPS时间分组文件，确保H1和L1配对
gps_groups = group_files_by_gps(data_files)
print(f"Found {len(gps_groups)} GPS time groups.")

# --- 主循环：遍历每个GPS时间组 ---
for group_idx, (gps_time, files) in enumerate(gps_groups.items()):
    total_groups = len(gps_groups)
    print(f"\n[{group_idx+1}/{total_groups}] Analyzing Segment: {gps_time} ...")
    
    try:
        # 检查是否同时存在H1和L1文件，缺少任何一个则跳过
        if not files['h1'] or not files['l1']:
            print(f"  -> Skip (Missing H1 or L1 file)")
            continue
        
        # 加载H1和L1的.pt数据文件（根据记忆#1，使用.pt作为数据源）
        h1_data = torch.load(files['h1'])
        l1_data = torch.load(files['l1'])
        
        # 如果是张量则转为numpy数组
        if isinstance(h1_data, torch.Tensor): 
            h1_data = h1_data.numpy()
        if isinstance(l1_data, torch.Tensor): 
            l1_data = l1_data.numpy()
        
        # 确保两个探测器数据长度一致，取最小长度
        min_len = min(len(h1_data), len(l1_data))
        if min_len < 8192:  # 数据太短则跳过（至少需要4秒数据）
            print(f"  -> Skip (Data too short)")
            continue
        
        # 组合成(2, N)形状：第0行为H1，第1行为L1
        combined_data = np.vstack([h1_data[:min_len], l1_data[:min_len]])
        CURRENT_SEGMENT_DATA = combined_data  # 更新全局变量供模拟器使用
        
        print(f"  -> Loaded H1 and L1 data (length: {min_len})")
        
        # 步骤1: 使用模拟器生成训练数据（10000次模拟）
        print(f"  -> Simulating training data...")
        theta_train, x_train = simulate_for_sbi(
            fast_simulator, 
            proposal=prior, 
            num_simulations=10000
        ) 
        
        # 步骤2: 训练ING-Net（使用完整统计特征：CC + 峰度 + 功率）
        print(f"  -> Training ING-Net...")
        inf_ai = SNPE(prior=prior, density_estimator="maf", device=device)
        inf_ai.append_simulations(theta_train, x_train)
        de_ai = inf_ai.train(show_train_summary=False, training_batch_size=1000)
        # 使用MCMC采样构建后验（比直接采样更稳定）
        post_ai = inf_ai.build_posterior(de_ai, sample_with='mcmc')
        
        # 保存ING-Net模型为.pt文件（根据记忆#1要求）
        torch.save(post_ai, os.path.join(RESULTS_DIR, f"ingnet_{gps_time}.pt"))
        
        # 步骤3: 训练传统基线（仅使用CC和功率，不使用峰度信息）
        print(f"  -> Training Traditional Baseline...")
        x_train_trad = x_train[:, [0, 3]]  # 只取第0列(CC)和第3列(功率)
        inf_trad = SNPE(prior=prior, density_estimator="maf", device=device)
        inf_trad.append_simulations(theta_train, x_train_trad)
        de_trad = inf_trad.train(show_train_summary=False, training_batch_size=1000)
        post_trad = inf_trad.build_posterior(de_trad, sample_with='mcmc')
        
        # 保存传统模型为.pt文件
        torch.save(post_trad, os.path.join(RESULTS_DIR, f"trad_{gps_time}.pt"))
        
        # 步骤4: 评估模型性能
        print(f"  -> Evaluating...")
        ul_ai, ul_trad = [], []  # 存储上限值列表
        valid_evaluations = 0
        
        # 进行多次评估（最多50次尝试，直到获得20个有效结果）
        for _ in range(50):
            try:
                seg_len = int(4.0 * 2048)
                n_h1, n_l1 = get_current_segment_noise(seg_len)
                
                # 数据质量检查：标准差过小视为无效数据
                if np.std(n_h1) < 1e-6 or np.std(n_l1) < 1e-6:
                    continue
                
                # 计算观测统计量
                cc, _ = pearsonr(n_h1, n_l1)
                if np.isnan(cc) or np.isinf(cc):  # 处理异常值
                    cc = 0.0
                
                k_h1, k_l1 = kurtosis(n_h1), kurtosis(n_l1)
                if np.isnan(k_h1) or np.isinf(k_h1):
                    k_h1 = 0.0
                if np.isnan(k_l1) or np.isinf(k_l1):
                    k_l1 = 0.0
                k_h1, k_l1 = np.log1p(abs(k_h1)), np.log1p(abs(k_l1))
                
                p = np.log10(np.var(n_h1)*np.var(n_l1) + 1e-30)
                
                # 构建观测向量
                obs = torch.tensor([cc, k_h1, k_l1, p], dtype=torch.float32, device=device)
                
                # 从两个后验中采样（各1000个样本）
                s_ai = post_ai.sample((1000,), x=obs, show_progress_bars=False)
                s_trad = post_trad.sample((1000,), x=obs[[0, 3]], show_progress_bars=False)
                
                # 计算95%置信上限（能量密度的对数值）
                ul_ai.append(np.percentile(s_ai[:,0].cpu().numpy(), 95))
                ul_trad.append(np.percentile(s_trad[:,0].cpu().numpy(), 95))
                valid_evaluations += 1
                
                if valid_evaluations >= 20:  # 达到20个有效评估则停止
                    break
                    
            except Exception as e:
                continue  # 单个评估失败则继续下一次尝试
        
        # 如果有效评估少于5次，跳过该段
        if valid_evaluations < 5:
            print(f"  -> Skip (Not enough valid evaluations)")
            continue
        
        # 计算平均上限值
        avg_ul_ai = np.mean(ul_ai)
        avg_ul_trad = np.mean(ul_trad)
        
        # 绘制后验分布对比图
        plt.figure(figsize=(8, 5))
        plt.hist(s_ai[:, 0].cpu().numpy(), bins=30, density=True, alpha=0.5, color='blue', label='ING-Net')
        plt.hist(s_trad[:, 0].cpu().numpy(), bins=30, density=True, alpha=0.5, color='orange', label='Traditional')
        plt.axvline(avg_ul_ai, color='blue', linestyle='--', label=f'ING-Net 95% UL: {avg_ul_ai:.2f}')
        plt.axvline(avg_ul_trad, color='orange', linestyle='--', label=f'Trad 95% UL: {avg_ul_trad:.2f}')
        plt.title(f'Segment {gps_time}\nING-Net Limit: 10^{avg_ul_ai:.2f} (Trad: 10^{avg_ul_trad:.2f})')
        plt.xlabel('log10(Omega)')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"plot_{gps_time}.png"))
        plt.close()
        
        print(f"  -> Saved .pt models and plot for {gps_time}")
        
    except Exception as e:
        print(f"  -> Error: {e}")
        continue  # 当前组出错则继续下一组

print("\nAll done!")