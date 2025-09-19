#!/usr/bin/env python3
"""
内存清理和监控脚本
用于在训练前清理GPU内存
"""

import torch
import gc
import os
import subprocess
import time

def clear_gpu_memory():
    """清理GPU内存"""
    if torch.cuda.is_available():
        print("清理GPU内存...")
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        print(f"GPU内存已清理，当前使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    else:
        print("CUDA不可用")

def get_gpu_memory_info():
    """获取GPU内存信息"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for i, line in enumerate(lines):
                used, total = map(int, line.split(', '))
                print(f"GPU {i}: {used}MB / {total}MB ({used/total*100:.1f}%)")
        else:
            print("无法获取GPU内存信息")
    except FileNotFoundError:
        print("nvidia-smi 未找到")

def set_memory_limits():
    """设置内存限制"""
    # 设置更严格的内存分配
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:8'
    
    # 设置垃圾回收阈值
    gc.set_threshold(50, 5, 5)
    
    print("内存限制已设置")

def main():
    print("=== GPU内存清理和监控 ===")
    
    # 显示清理前状态
    print("\n清理前状态:")
    get_gpu_memory_info()
    
    # 设置内存限制
    set_memory_limits()
    
    # 清理内存
    clear_gpu_memory()
    
    # 显示清理后状态
    print("\n清理后状态:")
    get_gpu_memory_info()
    
    print("\n内存清理完成！")

if __name__ == "__main__":
    main()

