#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LIGO 数据自动下载脚本
- 随机选择 GPS 时间（基于 O3a/O3b/O4a 大致时段）
- 循环下载 4 个批次
- 严格校验 H1/L1 时间匹配，不匹配则重试
- 保存为 .pt 格式（与您的图片命名一致）
"""

import numpy as np
import torch
from gwpy.timeseries import TimeSeries
import os
import time

# ==================== 配置 ====================
OUTPUT_DIR = "./ligo_data"
DURATION = 4096       # 数据长度（秒），约 1.1 小时，与您的文件大小(~64MB)匹配
SAMPLE_RATE = 4096    # 采样率 4KHz（如需 16KHz 可改为 16384，但文件会更大）

# 各观测段的中心 GPS 时间和合理范围（基于您的图片）
DATASET_RANGES = {
    'O3a': {
        'center_gps': 1238166018,  # 2019-05-07
        'start_gps': 1238166018 - 30*24*3600,  # ±30天
        'end_gps': 1238166018 + 30*24*3600,
        'description': 'O3a (Apr-Oct 2019)'
    },
    'O3b': {
        'center_gps': 1260834498,  # 2019-11-08
        'start_gps': 1260834498 - 30*24*3600,
        'end_gps': 1260834498 + 30*24*3600,
        'description': 'O3b (Nov 2019-Mar 2020)'
    },
    'O4a': {
        'center_gps': 1377415818,  # 2023-09-14
        'start_gps': 1377415818 - 60*24*3600,  # O4a 范围更大
        'end_gps': 1377415818 + 60*24*3600,
        'description': 'O4a (May 2023-Jan 2024)'
    }
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_segment(detector, gps_start, duration, sample_rate):
    """
    下载单个探测器数据，带错误处理
    """
    try:
        print(f"    下载 {detector} @ GPS {int(gps_start)}...")
        data = TimeSeries.fetch_open_data(
            detector, 
            gps_start, 
            gps_start + duration, 
            sample_rate=sample_rate,
            format='hdf5'
        )
        return data
    except Exception as e:
        print(f"    ⚠️ {detector} 下载失败: {str(e)[:60]}")
        return None

def validate_and_save(h1_data, l1_data, dataset, gps_start, batch_num):
    """
    验证 H1/L1 时间匹配，保存为 .pt
    """
    if h1_data is None or l1_data is None:
        return False
    
    # 严格校验时间对齐（起始时间差 < 1 秒）
    h1_start = float(h1_data.t0.value)
    l1_start = float(l1_data.t0.value)
    time_diff = abs(h1_start - l1_start)
    
    if time_diff > 1.0:
        print(f"  ❌ 时间不匹配！H1:{h1_start:.0f}, L1:{l1_start:.0f}, 差值:{time_diff:.1f}s")
        return False
    
    # 校验数据长度一致
    if len(h1_data) != len(l1_data):
        print(f"  ❌ 长度不匹配！H1:{len(h1_data)}, L1:{len(l1_data)}")
        return False
    
    # 转换为 Tensor
    h1_tensor = torch.from_numpy(h1_data.value).float()
    l1_tensor = torch.from_numpy(l1_data.value).float()
    
    # 文件名格式：Dataset_H1_GPS.pt / Dataset_L1_GPS.pt
    # 如果 batch_num > 0，添加后缀如 _4（与您图片中的 O3b_..._4 一致）
    suffix = f"_{batch_num}" if batch_num > 0 else ""
    
    h1_filename = f"{dataset}_H1_{int(gps_start)}{suffix}.pt"
    l1_filename = f"{dataset}_L1_{int(gps_start)}{suffix}.pt"
    
    h1_path = os.path.join(OUTPUT_DIR, h1_filename)
    l1_path = os.path.join(OUTPUT_DIR, l1_filename)
    
    torch.save(h1_tensor, h1_path)
    torch.save(l1_tensor, l1_path)
    
    print(f"  ✅ 成功保存: {h1_filename} ({len(h1_tensor)/SAMPLE_RATE/3600:.2f}h)")
    print(f"           {l1_filename}")
    return True

def get_random_gps(dataset):
    """在有效范围内生成随机 GPS 时间"""
    info = DATASET_RANGES[dataset]
    return np.random.randint(info['start_gps'], info['end_gps'])

def download_dataset_batches(dataset, n_batches=4, max_retries=20):
    """
    为某个数据集下载 n 个批次，确保 H1/L1 对应
    """
    print(f"\n{'='*60}")
    print(f"开始下载 {DATASET_RANGES[dataset]['description']}")
    print(f"目标: {n_batches} 个匹配批次 (H1+L1)")
    print(f"{'='*60}")
    
    successful_batches = 0
    attempts = 0
    
    while successful_batches < n_batches and attempts < max_retries:
        attempts += 1
        
        # 生成随机 GPS 时间（确保在数据段内且避开边缘）
        gps_start = get_random_gps(dataset)
        
        print(f"\n[尝试 {attempts}/{max_retries}] {dataset} 批次 {successful_batches+1}/{n_batches}")
        print(f"  GPS 时间: {gps_start} ({time.strftime('%Y-%m-%d %H:%M', time.gmtime(1238166018 + (gps_start-1238166018)))})")
        
        # 下载 H1 和 L1
        h1 = download_segment('H1', gps_start, DURATION, SAMPLE_RATE)
        time.sleep(0.5)  # 避免请求过于频繁
        l1 = download_segment('L1', gps_start, DURATION, SAMPLE_RATE)
        
        # 验证并保存
        if validate_and_save(h1, l1, dataset, gps_start, batch_num=successful_batches+1):
            successful_batches += 1
            time.sleep(1)  # 成功下载后短暂休息
        else:
            print(f"  🔄 该批次无效，重新选择 GPS 时间...")
            time.sleep(0.5)
    
    if successful_batches < n_batches:
        print(f"⚠️ 警告: {dataset} 仅完成 {successful_batches}/{n_batches} 批次")
    else:
        print(f"✅ {dataset} 全部 {n_batches} 批次下载完成！")
    
    return successful_batches

# ==================== 主程序 ====================

if __name__ == "__main__":
    # 检查依赖
    try:
        import gwpy
    except ImportError:
        print("请先安装 gwpy: pip install gwpy")
        exit(1)
    
    print("LIGO 数据自动下载工具")
    print(f"数据保存目录: {os.path.abspath(OUTPUT_DIR)}")
    print(f"每段时长: {DURATION/3600:.1f} 小时, 采样率: {SAMPLE_RATE}Hz")
    
    # 只下载 O3a 数据集，6个批次
    all_datasets = ['O3a']
    
    total_stats = {}
    for ds in all_datasets:
        count = download_dataset_batches(ds, n_batches=6, max_retries=30)
        total_stats[ds] = count
    
    # 最终统计
    print(f"\n{'='*60}")
    print("下载完成统计:")
    for ds, count in total_stats.items():
        status = "✅ 完成" if count == 4 else "⚠️ 部分"
        print(f"  {ds}: {count}/4 批次 {status}")
    
    print(f"\n文件列表:")
    files = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.pt')])
    for f in files:
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024 / 1024
        print(f"  {f} ({size:.1f} MB)")