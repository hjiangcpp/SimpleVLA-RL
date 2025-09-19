# SimpleVLA-RL 显存优化指南

## 问题描述
您的GPU只有23.49 GiB总容量，但模型需要更多显存。本指南提供了多种优化方案。

## 优化方案（按显存需求从高到低）

### 1. 标准LoRA版本
**文件**: `run_openvla_oft_rl_lora.sh`
- **LoRA rank**: 16
- **批次大小**: 32
- **预期显存节省**: 70-80%
- **适用场景**: 显存相对充足

### 2. 超低内存LoRA版本
**文件**: `run_openvla_oft_rl_lora_ultra_low.sh`
- **LoRA rank**: 4
- **批次大小**: 16
- **预期显存节省**: 85-90%
- **适用场景**: 显存严重不足

### 3. 极端低内存LoRA版本
**文件**: `run_openvla_oft_rl_lora_extreme_low.sh`
- **LoRA rank**: 2
- **批次大小**: 8
- **预期显存节省**: 90-95%
- **适用场景**: 显存极度不足

### 4. 最小LoRA版本
**文件**: `run_openvla_oft_rl_lora_minimal.sh`
- **LoRA rank**: 1
- **批次大小**: 4
- **预期显存节省**: 95-98%
- **适用场景**: 显存几乎耗尽

### 5. CPU卸载LoRA版本
**文件**: `run_openvla_oft_rl_lora_cpu_offload.sh`
- **LoRA rank**: 1
- **批次大小**: 2
- **预期显存节省**: 98%+
- **适用场景**: 显存完全不足（但会很慢）

## 其他优化技术

### 已启用的优化
- ✅ 梯度检查点 (Gradient Checkpointing)
- ✅ 参数卸载 (Parameter Offloading)
- ✅ 梯度卸载 (Gradient Offloading)
- ✅ 优化器卸载 (Optimizer Offloading)
- ✅ 动态批次大小 (Dynamic Batch Size)
- ✅ 填充移除 (Remove Padding)
- ✅ 量化引擎 (Quantized Engine)
- ✅ 严格内存管理

### 系统级优化
```bash
# 清理GPU内存
python3 -c "import torch; torch.cuda.empty_cache()"

# 监控显存使用
nvidia-smi -l 1
```

## 使用建议

### 1. 按顺序尝试
1. 先尝试 `run_openvla_oft_rl_lora.sh`
2. 如果失败，尝试 `run_openvla_oft_rl_lora_ultra_low.sh`
3. 继续尝试更激进的版本

### 2. 监控显存使用
```bash
# 实时监控
watch -n 1 nvidia-smi

# 或者
nvidia-smi -l 1
```

### 3. 如果所有方案都失败
- 考虑使用更小的模型
- 使用CPU训练（虽然很慢）
- 升级硬件
- 使用云端GPU

## 预期效果

| 版本 | LoRA Rank | 批次大小 | 显存节省 | 训练速度 | 模型性能 |
|------|-----------|----------|----------|----------|----------|
| 标准 | 16 | 32 | 70-80% | 快 | 好 |
| 超低 | 4 | 16 | 85-90% | 中等 | 中等 |
| 极端 | 2 | 8 | 90-95% | 慢 | 较差 |
| 最小 | 1 | 4 | 95-98% | 很慢 | 差 |
| CPU卸载 | 1 | 2 | 98%+ | 极慢 | 很差 |

## 故障排除

### 如果仍然出现OOM错误
1. 进一步减少批次大小到1
2. 减少序列长度到16或8
3. 使用rank=1的LoRA
4. 考虑使用CPU训练

### 如果训练太慢
1. 适当增加LoRA rank
2. 增加批次大小
3. 减少梯度检查点频率

## 总结

LoRA是解决显存不足问题的最佳方案，配合其他优化技术可以显著减少显存需求。建议从标准版本开始尝试，逐步降低参数直到能够运行。

