##为你逐段详细注释代码，解释每个部分的原理和目的：
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
# 为什么要单独创建目录：便于管理和查找结果，避免与原始数据混淆
RESULTS_DIR = r"C:\Users\20466\Desktop\新建文件夹 (6)\ING_Net_Segment_Results_O3A"
os.makedirs(RESULTS_DIR, exist_ok=True)  # exist_ok=True：如果目录已存在不报错

# O3a数据集路径（根据记忆#1，O3a的SCALING_FACTOR应为1300）
# 注意：这里SCALING_FACTOR=3000是代码中的默认值，但记忆提示O3a应该用1300
LOCAL_DATA_PATH = r"C:\Users\20466\Desktop\新建文件夹 (6)\ligo_data\processed\03A数据"

# 自动检测GPU/CPU设备
# 为什么要检测：GPU可以大幅加速神经网络训练，但如果不可用也能在CPU上运行
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device.upper()} | Saving .pt models to: {RESULTS_DIR}")

# ==========================================
# 1. Load Files (加载数据文件)
# ==========================================

# 只处理O3a相关的.pt文件（根据记忆#1，优先处理O3a数据）
# 为什么要过滤：避免处理其他数据集（如O3b、O4a）的文件，确保一致性
data_files = [f for f in glob.glob(os.path.join(LOCAL_DATA_PATH, "*.pt")) if "O3a" in f or "03A" in f]
print(f"Found {len(data_files)} O3A data files.")

# 如果没有找到文件则报错
# 提前检查可以避免后续代码出现难以调试的错误
if not data_files:
    raise FileNotFoundError("No .pt files found in data directory!")

# ==========================================
# 2. Helper Functions (辅助函数定义)
# ==========================================

def group_files_by_gps(data_files):
    """
    按GPS时间对H1和L1文件进行分组
    
    为什么需要分组：
    - LIGO有两个探测器（H1在汉福德，L1在利文斯顿）
    - 引力波信号会先后到达两个探测器（有时间差）
    - 只有同一GPS时间的数据才能配对分析
    - GPS时间是引力波事件的标准时间戳（秒级精度）
    """
    gps_groups = {}
    for filepath in data_files:
        filename = os.path.basename(filepath)
        # 使用正则表达式提取GPS时间戳（9-10位数字）
        # LIGO的GPS时间是从1980年1月6日开始的秒数，约为9-10位数字
        gps_match = re.search(r'(\d{9,10})', filename)
        if gps_match:
            gps_time = gps_match.group(1)
            if gps_time not in gps_groups:
                gps_groups[gps_time] = {'h1': None, 'l1': None}
            # 根据文件名中的探测器标识分类
            # H1 = Hanford（汉福德，华盛顿州）
            # L1 = Livingston（利文斯顿，路易斯安那州）
            if 'H1' in filename:
                gps_groups[gps_time]['h1'] = filepath
            elif 'L1' in filename:
                gps_groups[gps_time]['l1'] = filepath
    return gps_groups

# 全局变量，存储当前处理的数据段，供模拟器使用
# 为什么用全局变量：simulate_for_sbi要求模拟器是全局可访问的函数
# 替代方案是用类，但这里为了简洁使用全局变量
CURRENT_SEGMENT_DATA = None 

def get_current_segment_noise(seg_len):
    """
    从当前数据段中随机提取指定长度的噪声样本
    
    参数:
        seg_len: 需要的样本长度（采样点数），4秒*2048Hz=8192点
        
    为什么要随机提取：
    - LIGO数据很长（几个小时），可以提取很多段
    - 随机提取模拟真实观测中的不确定性
    - 增加训练数据的多样性，防止过拟合
    
    为什么要归一化：
    - 不同时间段噪声水平不同（白天/黑夜，天气影响）
    - 归一化使数据具有零均值和单位方差，便于神经网络学习
    - 防止数值过大导致梯度爆炸
    """
    global CURRENT_SEGMENT_DATA
    total_len = CURRENT_SEGMENT_DATA.shape[1]
    
    # 随机选择起始索引，确保不超出数据范围
    # np.random.randint确保每次调用得到不同的段，增加数据多样性
    if total_len <= seg_len: 
        start_idx = 0
    else: 
        start_idx = np.random.randint(0, total_len - seg_len)
        
    # 提取数据：[0,:]是H1，[1,:]是L1
    n_h1 = CURRENT_SEGMENT_DATA[0, start_idx : start_idx + seg_len]
    n_l1 = CURRENT_SEGMENT_DATA[1, start_idx : start_idx + seg_len]
    
    # Z-score归一化：减去均值，除以标准差（防止除零加极小值）
    # 公式：z = (x - μ) / σ
    # 为什么加1e-30：防止标准差为0（虽然罕见，但理论上可能）
    n_h1 = (n_h1 - np.mean(n_h1)) / (np.std(n_h1) + 1e-30)
    n_l1 = (n_l1 - np.mean(n_l1)) / (np.std(n_l1) + 1e-30)
    return n_h1, n_l1

def fast_simulator(theta_batch):
    """
    SBI快速模拟器：根据参数theta生成模拟数据统计特征
    
    核心物理原理：
    - 引力波背景（SGWB）由大量小爆发组成（类似 popcorn）
    - 每个爆发持续时间很短，在时域表现为脉冲
    - 两个探测器接收到的信号相同（只是有时间延迟）
    - 噪声是探测器本地的（不相关），信号是共有的（强相关）
    
    为什么要用统计特征而非原始波形：
    - 原始波形维度太高（8192维），需要超大神经网络
    - 统计特征（相关系数、峰度、功率）包含了关键信息
    - 这是"摘要统计量"（Summary Statistics），SBI的标准做法
    
    参数:
        theta_batch: 批次参数 [log10(omega), xi]
            - omega: 能量密度参数，描述引力波背景的强度
            - xi: 爆发事件率，描述单位时间内爆发事件的数量
    返回:
        统计特征张量 [相关系数, log(1+|H1峰度|), log(1+|L1峰度|), log(功率积)]
    """
    # 确保输入是PyTorch张量
    # SBI框架内部使用PyTorch，需要类型一致性
    if isinstance(theta_batch, np.ndarray): 
        theta_batch = torch.from_numpy(theta_batch)
    
    batch_stats = []
    
    # 遍历批次中的每个参数集
    # 为什么不向量化：每个样本需要独立的随机数（爆发位置和幅度）
    for theta in theta_batch:
        log10_omega = theta[0].item()  # .item()将张量转为Python标量
        xi = theta[1].item()           # xi（希腊字母ξ）是事件率参数
        
        # 计算段长度：4秒 * 2048Hz采样率 = 8192个采样点
        # 4秒是LIGO分析的标准长度，平衡了频率分辨率和计算效率
        seg_len = int(4.0 * 2048)
        
        # 获取当前段的噪声
        # 噪声是真实的LIGO数据，包含探测器噪声和干扰
        n_h1, n_l1 = get_current_segment_noise(seg_len)
            
        # 计算爆发信号幅度（关键物理公式）
        # 公式推导：
        # - 信号能量 ∝ omega（能量密度）
        # - 信号幅度 ∝ sqrt(能量) ∝ sqrt(omega)
        # - 爆发越频繁(xi大)，每个爆发的能量越小，幅度 ∝ 1/sqrt(xi)
        # - 3000.0是经验缩放因子，将物理单位转为ADC计数
        omega = 10**log10_omega  # 从对数转回线性
        amp = np.sqrt(omega / max(xi, 1e-4)) * 3000.0  # 1e-4防止除零
        
        # 计算段内爆发事件数
        # 0.2是效率因子：假设只有20%的爆发被有效探测
        # 这是经验值，考虑了信号叠加和探测效率
        n_events = int(seg_len * xi * 0.2)
        
        # 复制噪声，准备添加信号
        # 必须复制！否则原始噪声会被修改，影响后续使用
        d_h1, d_l1 = n_h1.copy(), n_l1.copy()
        
        # 在随机位置添加高斯爆发信号
        # 为什么是随机位置：模拟真实的随机爆发时间
        # 为什么H1和L1加相同信号：引力波是共有的，两个探测器看到相同波形（除时间延迟）
        # 为什么是正态分布：中心极限定理，大量小爆发叠加接近高斯
        if n_events > 0:
            idx = np.random.randint(0, seg_len, n_events)  # 随机位置
            burst = np.random.normal(0, amp, n_events)     # 随机幅度（均值为0，方差为amp^2）
            d_h1[idx] += burst
            d_l1[idx] += burst  # 相同信号！
            
        # 计算统计特征
        # 1. 皮尔逊相关系数：衡量H1和L1的线性相关性
        #    引力波信号是相关的，噪声是不相关的 → CC是探测关键指标
        cc, _ = pearsonr(d_h1, d_l1)
        
        # 2. 峰度（Kurtosis）：衡量分布的"尖峰"程度
        #    高斯噪声峰度≈3，有信号叠加时峰度会增大（更多异常值）
        #    对非高斯性敏感，是ING-Net的关键优势
        k_h1 = kurtosis(d_h1)  # scipy默认计算超额峰度（减去了3）
        k_l1 = kurtosis(d_l1)
        
        # 3. 对数功率积：H1和L1功率（方差）的乘积
        #    反映信号总能量，对强信号敏感
        p = np.log10(np.var(d_h1)*np.var(n_l1) + 1e-30)
        
        # 对峰度取log1p：压缩动态范围，处理极大值
        # log1p(x) = log(1+x)，当x很大时≈log(x)，当x很小时比log更稳定
        stats = torch.tensor([cc, np.log1p(abs(k_h1)), np.log1p(abs(k_l1)), p], dtype=torch.float32)
        batch_stats.append(stats)
        
    return torch.stack(batch_stats)  # 堆叠成[batch_size, 4]的张量

# 定义参数先验分布：log10(omega) ∈ [-9, -5], xi ∈ [0.001, 1.0]
# 为什么用均匀分布：表示我们对参数的真实值没有先验偏好（无知先验）
# 为什么omega范围是[-9, -5]：基于LIGO的灵敏度估计，更弱的信号探测不到，更强的应该已被发现
# 为什么xi范围是[0.001, 1.0]：从稀有爆发（0.1%占空比）到连续信号（100%占空比）
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
        # 检查配对完整性
        # 为什么要跳过：缺少一个探测器就无法计算相关系数（CC）
        if not files['h1'] or not files['l1']:
            print(f"  -> Skip (Missing H1 or L1 file)")
            continue
        
        # 加载.pt文件（PyTorch原生格式，保存了张量数据）
        # 为什么用torch.load：高效，保持数据类型，支持GPU直接加载
        h1_data = torch.load(files['h1'])
        l1_data = torch.load(files['l1'])
        
        # 类型转换：如果是张量转numpy，方便后续处理
        # 为什么转numpy：scipy的统计函数（pearsonr, kurtosis）需要numpy数组
        if isinstance(h1_data, torch.Tensor): 
            h1_data = h1_data.numpy()
        if isinstance(l1_data, torch.Tensor): 
            l1_data = l1_data.numpy()
        
        # 长度对齐：取最小长度，避免索引越界
        # 为什么可能不同：两个探测器的录制时间可能有微小差异
        min_len = min(len(h1_data), len(l1_data))
        if min_len < 8192:  # 8192 = 4秒 * 2048Hz，最小分析长度
            print(f"  -> Skip (Data too short)")
            continue
        
        # vstack垂直堆叠：[2, N]形状，第0行H1，第1行L1
        combined_data = np.vstack([h1_data[:min_len], l1_data[:min_len]])
        CURRENT_SEGMENT_DATA = combined_data  # 更新全局变量
        
        print(f"  -> Loaded H1 and L1 data (length: {min_len})")
        
        # 步骤1: 生成训练数据（10000次模拟）
        # simulate_for_sbi：SBI包的核心函数，并行化模拟
        # 为什么10000次：平衡计算时间和统计精度（神经网络需要足够数据）
        print(f"  -> Simulating training data...")
        theta_train, x_train = simulate_for_sbi(
            fast_simulator, 
            proposal=prior, 
            num_simulations=10000
        ) 
        
        # 步骤2: 训练ING-Net
        # SNPE = Sequential Neural Posterior Estimation（序列神经后验估计）
        # 这是SBI的一种算法，用神经网络学习从统计量到参数的后验分布
        print(f"  -> Training ING-Net...")
        inf_ai = SNPE(prior=prior, density_estimator="maf", device=device)
        # density_estimator="maf"：使用Masked Autoregressive Flow（掩码自回归流）
        # MAF是一种灵活的密度估计模型，可以学习复杂的后验分布形状
        
        inf_ai.append_simulations(theta_train, x_train)
        # 将模拟数据添加到推断对象
        
        de_ai = inf_ai.train(show_train_summary=False, training_batch_size=1000)
        # training_batch_size=1000：大batch加速训练，但需要足够显存
        
        post_ai = inf_ai.build_posterior(de_ai, sample_with='mcmc')
        # build_posterior：从训练好的密度估计器构建后验对象
        # sample_with='mcmc'：使用MCMC（马尔可夫链蒙特卡洛）采样
        # 为什么用MCMC：比直接采样更精确，特别是在高维或复杂后验情况下
        
        # 保存模型为.pt文件（PyTorch标准格式）
        # 保存整个后验对象，包括神经网络的权重和结构
        torch.save(post_ai, os.path.join(RESULTS_DIR, f"ingnet_{gps_time}.pt"))
        
        # 步骤3: 训练传统基线
        # 为什么只选CC和功率：这是传统引力波数据分析的标准统计量
        # 传统方法不使用峰度，因为我们想展示ING-Net利用峰度信息的优势
        print(f"  -> Training Traditional Baseline...")
        x_train_trad = x_train[:, [0, 3]]  # 第0列=CC，第3列=功率
        
        inf_trad = SNPE(prior=prior, density_estimator="maf", device=device)
        inf_trad.append_simulations(theta_train, x_train_trad)
        de_trad = inf_trad.train(show_train_summary=False, training_batch_size=1000)
        post_trad = inf_trad.build_posterior(de_trad, sample_with='mcmc')
        
        torch.save(post_trad, os.path.join(RESULTS_DIR, f"trad_{gps_time}.pt"))
        
        # 步骤4: 评估模型性能
        print(f"  -> Evaluating...")
        ul_ai, ul_trad = [], []  # 存储上限值列表（UL = Upper Limit）
        
        # 为什么要多次评估（20次）：
        # - 每次评估用不同的噪声段（随机提取）
        # - 减少随机性，得到更稳定的极限估计
        # - 类似于实验中的多次测量取平均
        valid_evaluations = 0
        
        for _ in range(50):  # 最多尝试50次，直到获得20个有效结果
            try:
                seg_len = int(4.0 * 2048)
                n_h1, n_l1 = get_current_segment_noise(seg_len)
                
                # 数据质量检查：标准差过小可能是恒零数据或损坏
                if np.std(n_h1) < 1e-6 or np.std(n_l1) < 1e-6:
                    continue
                
                # 计算观测统计量（与模拟器中相同的计算）
                cc, _ = pearsonr(n_h1, n_l1)
                if np.isnan(cc) or np.isinf(cc):  # 处理异常值
                    cc = 0.0
                
                k_h1, k_l1 = kurtosis(n_h1), kurtosis(n_l1)
                # 清理异常值（偶尔会出现数值溢出）
                if np.isnan(k_h1) or np.isinf(k_h1): k_h1 = 0.0
                if np.isnan(k_l1) or np.isinf(k_l1): k_l1 = 0.0
                k_h1, k_l1 = np.log1p(abs(k_h1)), np.log1p(abs(k_l1))
                
                p = np.log10(np.var(n_h1)*np.var(n_l1) + 1e-30)
                
                # 构建观测向量（与训练时相同的格式）
                obs = torch.tensor([cc, k_h1, k_l1, p], dtype=torch.float32, device=device)
                
                # ===== 核心：从后验分布中采样 =====
                # post_ai.sample((1000,), x=obs, ...)：
                # - 给定观测数据obs，从神经网络学习的后验分布中采样1000个参数值
                # - 这些样本代表了"给定观测，参数可能是什么"的概率分布
                # - 这是贝叶斯推断的核心：p(θ|x) ∝ p(x|θ) * p(θ)
                
                s_ai = post_ai.sample((1000,), x=obs, show_progress_bars=False)
                s_trad = post_trad.sample((1000,), x=obs[[0, 3]], show_progress_bars=False)
                
                # 计算95%置信上限
                # 含义：我们有95%的置信度认为真实参数小于这个值
                # 这是引力波天文学报告探测极限的标准方式
                ul_ai.append(np.percentile(s_ai[:,0].cpu().numpy(), 95))
                ul_trad.append(np.percentile(s_trad[:,0].cpu().numpy(), 95))
                valid_evaluations += 1
                
                if valid_evaluations >= 20:  # 达到20次有效评估就停止
                    break
                    
            except Exception as e:
                continue  # 单个评估失败不终止整个流程
        
        # 如果有效评估太少，该数据段不可靠，跳过
        if valid_evaluations < 5:
            print(f"  -> Skip (Not enough valid evaluations)")
            continue
        
        # 计算平均上限值
        avg_ul_ai = np.mean(ul_ai)
        avg_ul_trad = np.mean(ul_trad)
        
        # 绘制后验分布对比图
        plt.figure(figsize=(8, 5))
        
        # 绘制两个方法的后验分布直方图
        # 横轴是log10(omega)，纵轴是概率密度
        plt.hist(s_ai[:, 0].cpu().numpy(), bins=30, density=True, alpha=0.5, 
                 color='blue', label='ING-Net')
        plt.hist(s_trad[:, 0].cpu().numpy(), bins=30, density=True, alpha=0.5, 
                 color='orange', label='Traditional')
        
        # 画出95%上限线
        # 如果ING-Net的线在左边，说明它的极限更严格（更好）
        plt.axvline(avg_ul_ai, color='blue', linestyle='--', 
                   label=f'ING-Net 95% UL: {avg_ul_ai:.2f}')
        plt.axvline(avg_ul_trad, color='orange', linestyle='--', 
                   label=f'Trad 95% UL: {avg_ul_trad:.2f}')
        
        # 标题显示关键信息：GPS时间和两个极限值的对比
        plt.title(f'Segment {gps_time}\n'
                 f'ING-Net Limit: 10^{avg_ul_ai:.2f} (Trad: 10^{avg_ul_trad:.2f})')
        plt.xlabel('log10(Omega)')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"plot_{gps_time}.png"))
        plt.close()  # 关闭图，释放内存
        
        print(f"  -> Saved .pt models and plot for {gps_time}")
        
    except Exception as e:
        # 捕获所有异常，确保一个组失败不影响其他组
        print(f"  -> Error: {e}")
        continue

print("\nAll done!")
```

## 关键概念总结

# SBI (Simulation-Based Inference)
# 当似然函数p(x|θ)未知但可以从模拟器采样时，用神经网络学习后验

# SNPE (Sequential Neural Posterior Estimation)
# 序列神经后验估计，SBI的一种算法，通过多轮迭代改进后验估计

# MAF (Masked Autoregressive Flow)
# 掩码自回归流，一种生成模型，用于学习复杂的概率分布

# MCMC采样 (Markov Chain Monte Carlo)
# 马尔可夫链蒙特卡洛，从复杂分布中采样的数值方法

# 95%置信上限 (95% Credible Upper Limit)
# 贝叶斯统计中表示"有95%概率真实值小于此"的极限

# 峰度 (Kurtosis)
# 衡量分布尾部厚度的统计量，对非高斯信号敏感

# 皮尔逊相关系数 (Pearson Correlation Coefficient)
# 衡量两个变量线性相关程度的指标，引力波探测的核心
