# 消融实验指南 (Ablation Study Guide)

本文档指导您如何运行消融实验，以验证各个创新模块的有效性。

## 1. 实验设计

我们将对比以下几种变体：

| 实验名称 | 描述 | 命令行参数 |
| :--- | :--- | :--- |
| **Full Model** | 完整模型 (包含所有特性) | `--use_subscores` |
| **w/o Adaptive Fusion** | 去掉自适应融合 (使用简单拼接) | `--use_subscores --no_fusion` |
| **w/o Multi-task** | 去掉多任务辅助头 | (不加 `--use_subscores`) |
| **w/o Multi-scale** | 去掉多尺度随机裁剪 | `--use_subscores --no_multiscale` |
| **w/o Contrastive** | 去掉对比学习/正则化 | `--use_subscores --lambda_cont 0` |
| **w/o MI Loss** | 去掉互信息解耦约束 | `--use_subscores --lambda_mi 0` |

## 2. 运行命令示例

请在终端中依次运行以下命令（建议每个实验跑 30-50 轮，观察 Best SRCC）：

### 2.1 完整模型 (Baseline for Ablation)
```bash
python K:\NVS-SQA\TWO\train_advanced.py --root_dir "K:\NVS-SQA\renders" --mos_file "K:\NVS-SQA\TWO\mos_advanced.json" --use_subscores --epochs 32 --gpu 0 --no_wandb & \\
python K:\NVS-SQA\TWO\train_advanced.py --root_dir "K:\NVS-SQA\renders" --mos_file "K:\NVS-SQA\TWO\mos_advanced.json" --use_subscores --epochs 32 --no_fusion --gpu 0  --no_wandb & \\
python K:\NVS-SQA\TWO\train_advanced.py --root_dir "K:\NVS-SQA\renders" --mos_file "K:\NVS-SQA\TWO\mos_advanced.json" --gpu 0 --epochs 32 --no_wandb \\ &
python K:\NVS-SQA\TWO\train_advanced.py --root_dir "K:\NVS-SQA\renders" --mos_file "K:\NVS-SQA\TWO\mos_advanced.json" --use_subscores --no_multiscale --gpu 0 --no_wandb 


--wandb_name "Full_Model"
```

python /media/abc/One\ Touch/NVS-SQA/TWO/train_advanced1.py --root_dir "/media/abc/One Touch/NVS-SQA/renders" --mos_file "/media/abc/One Touch/NVS-SQA/TWO/mos_advanced.json" --use_subscores --epochs 32 --gpu 0 --no_wandb
python "/media/abc/One Touch/NVS-SQA/TWO/train_advanced1.py" --root_dir "/media/abc/One Touch/NVS-SQA/renders" --mos_file "/media/abc/One Touch/NVS-SQA/TWO/mos_advanced.json" --use_subscores --epochs 32 --gpu 0 --no_wandb

### 2.2 去掉自适应融合 (w/o Fusion)
```bash
python K:\NVS-SQA\TWO\train_advanced.py --root_dir "K:\NVS-SQA\renders" --mos_file "K:\NVS-SQA\TWO\mos_advanced.json" --use_subscores --epochs 32 --no_fusion --gpu 0 --wandb_name "No_Fusion"
```

### 2.3 去掉多任务 (w/o Multi-task)
```bash
python K:\NVS-SQA\TWO\train_advanced.py --root_dir "K:\NVS-SQA\renders" --mos_file "K:\NVS-SQA\TWO\mos_advanced.json" --gpu 0 --wandb_name "No_Multitask"
```
*(注意：这里去掉了 `--use_subscores`)*

### 2.4 去掉多尺度 (w/o Multi-scale)
```bash
python K:\NVS-SQA\TWO\train_advanced.py --root_dir "K:\NVS-SQA\renders" --mos_file "K:\NVS-SQA\TWO\mos_advanced.json" --use_subscores --no_multiscale --gpu 0 --wandb_name "No_Multiscale"
```

## 3. 关于训练轮数 (Epochs) 与 Early Stopping

我在代码中加入了 **Early Stopping (早停)** 机制。

*   **默认 Patience**: 10 轮。如果验证集 SRCC 连续 10 轮没有提升，训练会自动停止。
*   **建议设置**:
    *   `--epochs 100`: 设置一个较大的上限。
    *   `--patience 15`: 给模型多一点耐心。
    
**如何找到最优轮数？**
您不需要手动找。程序会自动保存 `best_model_timestamp.pth`。
只要您设置了 Early Stopping，程序停下来的时候，最好的模型就已经被保存了。

**推荐命令 (自动找最优):**
```bash
python K:\NVS-SQA\TWO\train_advanced.py --root_dir "K:\NVS-SQA\renders" --mos_file "K:\NVS-SQA\TWO\mos_advanced.json" --use_subscores --epochs 200 --patience 15 --gpu 0 --no_wandb
```
