# TWO
# Dis-NeRF-VQA Advanced (THREE) 使用说明

这是 Dis-NeRF-VQA 的进阶版本 (v3)，包含了您设计的高级特性：**自适应特征融合**、**多任务失真感知头**、**多尺度随机裁剪**以及**对比学习策略**。

## 1. 核心特性 (Innovation)

*   **Adaptive Feature Fusion (自适应特征融合)**: 使用 Gated Attention 机制，自动学习内容特征和失真特征的权重，而不是简单的拼接。
*   **Multi-task Distortion Head (多任务失真感知头)**: 利用辅助任务（预测模糊、伪影等子分数）来增强模型对特定失真的敏感度。
*   **Multi-scale Random Cropping (多尺度随机裁剪)**: 训练时随机缩放和裁剪，模拟不同视距，增强鲁棒性。
*   **Contrastive Learning Strategy (对比学习策略)**: 引入特征正则化，优化特征空间分布。

## 2. 如何运行

### 2.1 准备数据
确保您已经运行了 `convert_excel_to_json.py` 生成了 `mos_advanced.json`，其中包含子分数。

### 2.2 启动训练 (带子分数)
如果您有子分数数据（如本数据集），请加上 `--use_subscores` 参数：

```bash
python train_advanced.py --root_dir "../renders" --mos_file "mos_advanced.json" --use_subscores --gpu 0
```

### 2.3 启动训练 (不带子分数 - 通用模式)
如果在**其他数据集**上运行（没有模糊/伪影等子标签），**不要加** `--use_subscores` 参数即可。模型会自动忽略辅助任务，只预测主 MOS 分数。

```bash
python train_advanced.py --root_dir "/path/to/other/dataset" --mos_file "mos_other.json" --gpu 0
```

## 3. 参数说明

*   `--use_subscores`: 启用辅助任务（需要 JSON 中包含 `sub_scores` 字段）。
# Dis-NeRF-VQA Advanced (THREE) 使用说明

这是 Dis-NeRF-VQA 的进阶版本 (v3)，包含了您设计的高级特性：**自适应特征融合**、**多任务失真感知头**、**多尺度随机裁剪**以及**对比学习策略**。

## 1. 核心特性 (Innovation)

*   **Adaptive Feature Fusion (自适应特征融合)**: 使用 Gated Attention 机制，自动学习内容特征和失真特征的权重，而不是简单的拼接。
*   **Multi-task Distortion Head (多任务失真感知头)**: 利用辅助任务（预测模糊、伪影等子分数）来增强模型对特定失真的敏感度。
*   **Multi-scale Random Cropping (多尺度随机裁剪)**: 训练时随机缩放和裁剪，模拟不同视距，增强鲁棒性。
*   **Contrastive Learning Strategy (对比学习策略)**: 引入特征正则化，优化特征空间分布。

## 2. 如何运行

### 2.1 准备数据
确保您已经运行了 `convert_excel_to_json.py` 生成了 `mos_advanced.json`，其中包含子分数。

### 2.2 启动训练 (带子分数)
如果您有子分数数据（如本数据集），请加上 `--use_subscores` 参数：

```bash
python train_advanced.py --root_dir "../renders" --mos_file "mos_advanced.json" --use_subscores --gpu 0
```

### 2.3 启动训练 (不带子分数 - 通用模式)
如果在**其他数据集**上运行（没有模糊/伪影等子标签），**不要加** `--use_subscores` 参数即可。模型会自动忽略辅助任务，只预测主 MOS 分数。

```bash
python train_advanced.py --root_dir "/path/to/other/dataset" --mos_file "mos_other.json" --gpu 0
```

## 3. 参数说明

*   `--use_subscores`: 启用辅助任务（需要 JSON 中包含 `sub_scores` 字段）。
*   `--lambda_sub`: 子分数损失的权重 (默认 0.5)。
*   `--lambda_cont`: 对比学习损失的权重 (默认 0.1)。
*   `--lambda_mi`: 互信息损失权重 (默认 0.1)。

## 4. 模型保存
训练过程中，最好的模型会自动保存到 `checkpoints_advanced/` 目录下，文件名包含时间戳，例如 `best_model_20251128_1030.pth`。

## 5. 模型评估
使用 `evaluate_advanced.py` 评估训练好的模型。

```bash
# 评估命令

# 1. Full Model (完整模型)
python K:\NVS-SQA\TWO\evaluate_advanced.py --checkpoint "K:\NVS-SQA\TWO\checkpoints_advanced\best_model_20251128_223410.pth" --gpu 0 --note "Full Model"

# 2. w/o Fusion (无融合) -> 注意这里必须加 --no_fusion
python K:\NVS-SQA\TWO\evaluate_advanced.py --checkpoint "K:\NVS-SQA\TWO\checkpoints_advanced\best_model_20251128_224859.pth" --gpu 0 --no_fusion --note "Ablation: No Fusion"

# 3. w/o Multi-task (无多任务)
python K:\NVS-SQA\TWO\evaluate_advanced.py --checkpoint "K:\NVS-SQA\TWO\checkpoints_advanced\best_model_20251128_225840.pth" --gpu 0 --note "Ablation: Multi-task"

# 4. w/o Multi-scale (无多尺度)
python K:\NVS-SQA\TWO\evaluate_advanced.py --checkpoint "K:\NVS-SQA\TWO\checkpoints_advanced\best_model_20251128_231241.pth" --gpu 0 --note "Ablation: Multi-scale"

```


python K:\NVS-SQA\TWO\evaluate_advanced.py --checkpoint "K:\NVS-SQA\TWO\checkpoints_advanced\best_model_20251128_223410.pth" --gpu 0 --note "Full Model"


python K:\NVS-SQA\TWO\evaluate_advanced.py --checkpoint "K:\NVS-SQA\TWO\checkpoints_advanced\best_model_20251128_224859.pth" --gpu 0 --no_fusion --note "Ablation: No Fusion"


python K:\NVS-SQA\TWO\evaluate_advanced.py --checkpoint "K:\NVS-SQA\TWO\checkpoints_advanced\best_model_20251128_225840.pth" --gpu 0 --note "Ablation: Multi-task"



**参数说明：**
*   `--checkpoint`: 模型权重文件的路径。
*   `--use_subscores`: 如果想评估子分数（如模糊、伪影），加上此参数。
*   `--save_results`: 结果保存路径 (默认 `results_advanced.json`)。

**输出示例：**
```text
==============================
EVALUATION RESULTS (Advanced)
==============================
SRCC: 0.9281
PLCC: 0.9516
RMSE: 0.0823
```

消融实验可视化工具
为了方便您对比不同变体的效果，我为您写了一个可视化脚本 K:\NVS-SQA\TWO\plot_ablation_results.py。

功能： 它会自动读取 K:\NVS-SQA\TWO\eval_results\ 下的所有 JSON 结果文件，并画出一张柱状图（SRCC/PLCC）和折线图（RMSE），直观展示各个模块的贡献。

使用方法： 等您跑完所有消融实验后，运行：

bash
python K:\NVS-SQA\TWO\plot_ablation_results.py
它会生成 ablation_comparison.png 图片。

建议： 在评估时，请务必使用 --note 参数标记实验名称（如 "Full Model", "No Fusion" 等），脚本会自动识别这些标记来命名图表中的图例。
