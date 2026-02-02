#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LIGO 数据批量检测与预处理工具
检测是否已白化/归一化，未处理则自动处理并输出到新文件夹
"""

import numpy as np
import torch
from scipy import signal
from scipy.stats import kurtosis, normaltest
import os
import glob
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt

class DataInspector:
    """数据质量检测器"""
    
    def __init__(self):
        self.thresholds = {
            'mean_abs': 0.5,      # 均值绝对值阈值
            'std_min': 0.3,       # 标准差下限
            'std_max': 5.0,       # 标准差上限  
            'max_val': 50,        # 最大绝对值阈值（原始strain通常>1e-19，处理后<10）
            'min_val': -50,
            'normality_p': 0.01   # 正态检验p值
        }
    
    def diagnose(self, data, filename="Unknown"):
        """
        全面诊断数据状态
        返回: (is_preprocessed, report_dict)
        """
        if isinstance(data, torch.Tensor):
            data = data.numpy()
        
        # 基础统计
        report = {
            'filename': filename,
            'length': len(data),
            'mean': float(np.mean(data)),
            'std': float(np.std(data)),
            'min': float(np.min(data)),
            'max': float(np.max(data)),
            'max_abs': float(np.max(np.abs(data))),
            'has_nan': bool(np.isnan(data).any()),
            'has_inf': bool(np.isinf(data).any()),
            'kurtosis': float(kurtosis(data)),
            'normality_p': float(normaltest(data)[1]) if len(data) > 20 else 1.0
        }
        
        issues = []
        
        # 关键检测指标
        if report['has_nan']:
            issues.append("包含NaN值")
        if report['has_inf']:
            issues.append("包含Inf值")
        if abs(report['mean']) > self.thresholds['mean_abs']:
            issues.append(f"未去直流偏置 (mean={report['mean']:.2e})")
        if report['std'] < self.thresholds['std_min']:
            issues.append(f"标准差过小 ({report['std']:.3f})，可能已压缩或恒定")
        if report['std'] > self.thresholds['std_max']:
            issues.append(f"标准差过大 ({report['std']:.2f})，未归一化")
        if report['max_abs'] > self.thresholds['max_val']:
            issues.append(f"数值范围过大 ({report['max_abs']:.2e})，疑似原始应变数据")
        if report['normality_p'] < self.thresholds['normality_p']:
            issues.append(f"显著偏离高斯分布 (p={report['normality_p']:.2e})")
            
        is_preprocessed = len(issues) == 0
        
        return is_preprocessed, report, issues

class DataProcessor:
    """数据预处理器"""
    
    def __init__(self, target_fs=4096, low_freq=20.0, high_freq=1800.0):
        self.target_fs = target_fs
        self.low_freq = low_freq
        self.high_freq = high_freq
        
    def process(self, data, original_fs=16384):
        """
        完整预处理流程
        """
        if isinstance(data, torch.Tensor):
            data = data.numpy()
            
        print(f"    开始预处理 ({len(data)} 点)...")
        processed = data.copy().astype(np.float64)
        
        # 1. 清洗异常值
        processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)
        
        # 2. 去直流偏置和高阶趋势
        processed = processed - np.mean(processed)
        
        # 3. 去趋势（多项式）
        time_idx = np.arange(len(processed))
        for order in [3, 2, 1]:  # 先尝试高阶，失败则降级
            try:
                coeffs = np.polyfit(time_idx, processed, order)
                trend = np.polyval(coeffs, time_idx)
                processed = processed - trend
                break
            except:
                continue
        
        # 4. 重采样（如果需要）
        if original_fs != self.target_fs and len(processed) > 1000:
            resample_ratio = self.target_fs / original_fs
            new_len = int(len(processed) * resample_ratio)
            processed = signal.resample(processed, new_len)
            print(f"      重采样: {original_fs}Hz → {self.target_fs}Hz")
        
        # 5. 带通滤波（去除地震噪声和高频噪声）
        sos_high = signal.butter(4, self.low_freq, btype='high', 
                                fs=self.target_fs, output='sos')
        sos_low = signal.butter(4, self.high_freq, btype='low', 
                               fs=self.target_fs, output='sos')
        processed = signal.sosfilt(sos_high, processed)
        processed = signal.sosfilt(sos_low, processed)
        
        # 6. 频域白化（分段处理避免内存溢出）
        processed = self._whiten(processed)
        
        # 7. 最终归一化（零均值单位方差）
        processed = (processed - np.mean(processed)) / (np.std(processed) + 1e-30)
        
        # 限制范围防止极端值
        processed = np.clip(processed, -20, 20)
        
        return torch.from_numpy(processed.astype(np.float32))
    
    def _whiten(self, data, segment_duration=4.0):
        """鲁棒频域白化"""
        seg_samples = int(segment_duration * self.target_fs)
        if seg_samples > len(data):
            seg_samples = len(data) // 4
            
        n_segs = len(data) // seg_samples
        if n_segs == 0:
            return data
            
        whitened = []
        for i in range(n_segs):
            seg = data[i*seg_samples : (i+1)*seg_samples]
            
            # Welch法估计PSD
            f, psd = signal.welch(seg, fs=self.target_fs, nperseg=seg_samples//4)
            psd_smooth = signal.medfilt(psd, kernel_size=5)
            psd_smooth = np.maximum(psd_smooth, 1e-30)
            
            # 频域白化
            fft = np.fft.rfft(seg)
            freqs = np.fft.rfftfreq(len(seg), 1/self.target_fs)
            psd_interp = np.interp(freqs, f, psd_smooth, 
                                  left=psd_smooth[0], right=psd_smooth[-1])
            
            white_fft = fft / (np.sqrt(psd_interp) + 1e-30)
            white_fft = np.clip(white_fft, -1e8, 1e8)  # 防止数值爆炸
            
            white_seg = np.fft.irfft(white_fft, n=len(seg))
            whitened.append(white_seg)
            
        return np.concatenate(whitened)

def plot_comparison(original, processed, filename, save_path):
    """可视化对比原始和处理后的数据"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 时域对比
    time_orig = np.arange(len(original)) / 16384  # 假设原始16KHz
    time_proc = np.arange(len(processed)) / 4096   # 处理后4KHz
    
    axes[0, 0].plot(time_orig[:1000], original[:1000], 'b-', alpha=0.7, label='Original')
    axes[0, 0].set_title(f'{filename}\nOriginal (First 1000 samples)')
    axes[0, 0].set_xlabel('Time (s)')
    axes[0, 0].set_ylabel('Strain')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].plot(time_proc[:1000], processed.numpy()[:1000], 'r-', alpha=0.7, label='Processed')
    axes[0, 1].set_title('Processed (First 1000 samples)')
    axes[0, 1].set_xlabel('Time (s)')
    axes[0, 1].set_ylabel('Whitened Strain')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 直方图对比
    axes[1, 0].hist(original, bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[1, 0].set_title(f'Original Distribution\nMean: {np.mean(original):.2e}, Std: {np.std(original):.2e}')
    axes[1, 0].set_xlabel('Value')
    axes[1, 0].set_ylabel('Count')
    
    axes[1, 1].hist(processed.numpy(), bins=50, alpha=0.7, color='red', edgecolor='black')
    axes[1, 1].set_title(f'Processed Distribution\nMean: {np.mean(processed.numpy()):.2e}, Std: {np.std(processed.numpy()):.2f}')
    axes[1, 1].set_xlabel('Value')
    axes[1, 1].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

# ==========================================
# 主程序
# ==========================================

def main():
    # 配置路径
    INPUT_DIR = r"C:\Users\20466\Desktop\新建文件夹 (6)\ligo_data"
    OUTPUT_DIR = os.path.join(INPUT_DIR, "processed")
    PLOT_DIR = os.path.join(INPUT_DIR, "diagnostic_plots")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)
    
    print(f"扫描目录: {INPUT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("="*60)
    
    # 查找所有.pt文件
    pt_files = glob.glob(os.path.join(INPUT_DIR, "*.pt"))
    if not pt_files:
        print("未找到 .pt 文件!")
        return
    
    print(f"找到 {len(pt_files)} 个数据文件")
    
    inspector = DataInspector()
    processor = DataProcessor(target_fs=4096)
    
    stats = {
        'total': 0,
        'preprocessed': 0,
        'needs_processing': 0,
        'processed': 0,
        'failed': 0
    }
    
    # 处理每个文件
    for filepath in tqdm(pt_files, desc="检测与处理中"):
        try:
            filename = os.path.basename(filepath)
            stats['total'] += 1
            
            print(f"\n[{stats['total']}/{len(pt_files)}] {filename}")
            
            # 加载数据
            data = torch.load(filepath, map_location='cpu')
            
            # 检测
            is_preprocessed, report, issues = inspector.diagnose(data, filename)
            
            print(f"  状态: {'✅ 已预处理' if is_preprocessed else '⚠️  需要处理'}")
            print(f"  统计: Mean={report['mean']:.2e}, Std={report['std']:.2f}, MaxAbs={report['max_abs']:.2e}")
            
            if issues:
                print(f"  问题: {', '.join(issues)}")
            
            # 如果已预处理，复制到输出目录（保持一致性）
            if is_preprocessed:
                stats['preprocessed'] += 1
                output_path = os.path.join(OUTPUT_DIR, filename)
                # 如果输出目录中没有该文件，则复制
                if not os.path.exists(output_path):
                    shutil.copy2(filepath, output_path)
                    print(f"  操作: 已复制到 processed/")
                else:
                    print(f"  操作: 已存在，跳过")
            else:
                stats['needs_processing'] += 1
                print(f"  操作: 正在预处理...")
                
                # 执行预处理
                processed_data = processor.process(data, original_fs=16384)  # 假设原始为16KHz
                
                # 验证预处理结果
                is_ok, new_report, _ = inspector.diagnose(processed_data, filename)
                if not is_ok:
                    print(f"  ⚠️  预处理后仍有问题，请检查参数")
                
                # 保存处理后的数据
                output_path = os.path.join(OUTPUT_DIR, filename)
                torch.save(processed_data, output_path)
                stats['processed'] += 1
                print(f"  ✅ 已保存: {output_path}")
                
                # 生成对比图
                plot_path = os.path.join(PLOT_DIR, f"{filename.replace('.pt', '.png')}")
                plot_comparison(data.numpy() if torch.is_tensor(data) else data, 
                              processed_data, filename, plot_path)
                
        except Exception as e:
            print(f"  ❌ 错误: {str(e)}")
            stats['failed'] += 1
            continue
    
    # 最终报告
    print("\n" + "="*60)
    print("处理完成报告")
    print("="*60)
    print(f"总文件数:     {stats['total']}")
    print(f"已预处理:     {stats['preprocessed']} (直接复制)")
    print(f"需要处理:     {stats['needs_processing']}")
    print(f"成功处理:     {stats['processed']}")
    print(f"处理失败:     {stats['failed']}")
    print(f"\n输出位置: {OUTPUT_DIR}")
    print(f"诊断图表: {PLOT_DIR}")

if __name__ == "__main__":
    main()