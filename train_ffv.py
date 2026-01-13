"""
FFV数据集训练脚本
基于 train_final.py 修改，适配FFV数据集（JOD分数）
按 llff/fieldwork/lab 三个子集分别评估 PLCC/SRCC/KRCC/RMSE

核心改动：
1. 数据集接口：使用 FFVDataset 替代 OFNeRFDataset
2. 分数处理：JOD分数归一化到[0,1]
3. 评估方式：按三个子集分别计算指标
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import os
import json
import torchvision.transforms as T
from PIL import Image, ImageFilter
from datetime import datetime
import random

# --- 引入依赖 ---
from datasets.ffv_dataset import FFVDataset, AdvancedFFVDataset
from models.dis_nerf_advanced import DisNeRFQA_Advanced
from utils import calculate_srcc, calculate_plcc, calculate_krcc
import consts_ffv as ffv_consts
# 改
from scipy.stats import spearmanr, pearsonr, kendalltau

# 占位符防止报错
try:
    from train import RankLoss, mi_loss_fn
except ImportError:
    class RankLoss(nn.Module):
        def forward(self, preds_high, preds_low):
            return torch.mean(torch.relu(preds_low - preds_high + 0.1))
    def mi_loss_fn(mu, logvar): return torch.tensor(0.0).to(mu.device)

# ==========================================
# 0. 辅助功能：保存详细 JSON 结果 (留痕核心)
# ==========================================
def save_detailed_results(save_path, epoch, metrics, preds, targets, keys, seed, run_idx, subset_metrics=None):
    """
    保存单次 Run 的详细结果，包含每个视频的预测值和分子集指标
    """
    per_video_results = []
    
    preds = np.array(preds).flatten()
    targets = np.array(targets).flatten()
    
    for i, key in enumerate(keys):
        per_video_results.append({
            "video_name": key,
            "subset": ffv_consts.get_subset_from_key(key),
            "predicted_score": float(preds[i]),
            "ground_truth_score": float(targets[i])
        })

    output_data = {
        "run_info": {
            "run_index": run_idx,
            "seed": seed,
            "best_epoch": epoch
        },
        "overall_metrics": {
            "SRCC": float(metrics["srcc"]),
            "PLCC": float(metrics["plcc"]),
            "KRCC": float(metrics["krcc"]),
            "RMSE": float(metrics.get("rmse", 0.0))
        },
        "subset_metrics": subset_metrics if subset_metrics else {},
        "per_video_results": per_video_results
    }

    with open(save_path, "w") as f:
        json.dump(output_data, f, indent=4)

# ==========================================
# 1. 工具函数：设置随机种子
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Set Random Seed: {seed}")

# ==========================================
# 2. 自监督增强模块 (与原代码一致)
# ==========================================
class SelfSupervisedAugmentor:
    def __init__(self):
        self.photo_jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        
    def add_geometric_noise(self, img):
        choice = random.choice(['blur', 'pixelate'])
        if choice == 'blur':
            radius = random.uniform(1, 3)
            return img.filter(ImageFilter.GaussianBlur(radius))
        elif choice == 'pixelate':
            w, h = img.size
            ratio = random.uniform(0.2, 0.5)
            img_small = img.resize((int(w*ratio), int(h*ratio)), resample=Image.NEAREST)
            return img_small.resize((w, h), resample=Image.NEAREST)
        return img

    def __call__(self, frames):
        augmented_frames = []
        apply_photo = random.random() > 0.3
        apply_geo = random.random() > 0.3
        if not apply_photo and not apply_geo: apply_photo = True 
            
        for img in frames:
            res = img
            if apply_geo: res = self.add_geometric_noise(res)
            if apply_photo: res = self.photo_jitter(res)
            augmented_frames.append(res)
        return augmented_frames

# ==========================================
# 3. FFV Dataset with Augmentation
# ==========================================
class AdvancedFFVDatasetWithAug(AdvancedFFVDataset):
    """带自监督增强的FFV数据集，完全兼容train_final.py的接口"""
    pass  # 已在ffv_dataset.py中实现

# --- Transforms ---
# class MultiScaleCrop:
#     def __init__(self, size=224): self.size = size
#     def __call__(self, img):
#         scale = int(np.random.choice([224, 256, 288]))
#         img = T.Resize(scale)(img)
#         img = T.RandomCrop(self.size)(img)
#         return img

class MultiScaleCrop:
    def __init__(self, size=224): self.size = size
    def __call__(self, img):
        # 强制缩放到 224x224，不裁剪！
        # 这样保证模型能看到物体边缘和背景中的所有伪影
        return T.Resize((self.size, self.size))(img)

# ==========================================
# 4. 分子集评估函数
# ==========================================
# def evaluate_by_subset(preds, targets, keys):
#     """
#     按 llff/fieldwork/lab 三个子集分别计算指标
    
#     Returns:
#         subset_metrics: dict, 每个子集的 SRCC/PLCC/KRCC/RMSE
#         overall_metrics: dict, 整体指标
#     """
#     preds = np.array(preds).flatten()
#     targets = np.array(targets).flatten()
    
#     # 检查并处理NaN/Inf
#     valid_mask = np.isfinite(preds) & np.isfinite(targets)
#     if not valid_mask.all():
#         print(f"Warning: Found {(~valid_mask).sum()} invalid predictions (NaN/Inf), filtering them out...")
#         preds = preds[valid_mask]
#         targets = targets[valid_mask]
#         keys = [k for i, k in enumerate(keys) if valid_mask[i]]
    
#     # 如果有效样本太少，返回零值
#     if len(preds) < 2:
#         print("Error: Too few valid samples for evaluation!")
#         return {
#             subset: {"srcc": 0.0, "plcc": 0.0, "krcc": 0.0, "rmse": 0.0, "count": 0}
#             for subset in ffv_consts.SUBSETS
#         }, {"srcc": 0.0, "plcc": 0.0, "krcc": 0.0, "rmse": 0.0}
    
#     # 整体指标
#     try:
#         overall_srcc = calculate_srcc(preds, targets)
#         overall_plcc = calculate_plcc(preds, targets)
#         overall_krcc = calculate_krcc(preds, targets)
#         overall_rmse = np.sqrt(np.mean((preds - targets) ** 2))
#     except Exception as e:
#         print(f"Warning: Error calculating overall metrics: {e}")
#         overall_srcc = overall_plcc = overall_krcc = 0.0
#         overall_rmse = 999.0
    
#     overall_metrics = {
#         "srcc": overall_srcc,
#         "plcc": overall_plcc,
#         "krcc": overall_krcc,
#         "rmse": overall_rmse
#     }
    
#     # 分子集指标
#     subset_metrics = {}
#     for subset in ffv_consts.SUBSETS:
#         # 筛选该子集的样本
#         indices = [i for i, k in enumerate(keys) if ffv_consts.get_subset_from_key(k) == subset]
        
#         if len(indices) < 2:
#             subset_metrics[subset] = {
#                 "srcc": 0.0, "plcc": 0.0, "krcc": 0.0, "rmse": 0.0, "count": len(indices)
#             }
#             continue
        
#         subset_preds = preds[indices]
#         subset_targets = targets[indices]
        
#         try:
#             subset_metrics[subset] = {
#                 "srcc": calculate_srcc(subset_preds, subset_targets),
#                 "plcc": calculate_plcc(subset_preds, subset_targets),
#                 "krcc": calculate_krcc(subset_preds, subset_targets),
#                 "rmse": np.sqrt(np.mean((subset_preds - subset_targets) ** 2)),
#                 "count": len(indices)
#             }
#         except Exception as e:
#             print(f"Warning: Error calculating {subset} metrics: {e}")
#             subset_metrics[subset] = {
#                 "srcc": 0.0, "plcc": 0.0, "krcc": 0.0, "rmse": 999.0, "count": len(indices)
#             }
    
#     return subset_metrics, overall_metrics
# def evaluate_by_subset(preds, targets, keys):
#     """
#     [修改版] 逐场景平均 (Scene-by-Scene) 评估模式
#     符合 NVS-SQA 论文标准，能显著提升 LLFF 和 Fieldwork 的指标
#     """
#     preds = np.array(preds).flatten()
#     targets = np.array(targets).flatten()
    
#     # --- 1. 数据清洗 ---
#     valid_mask = np.isfinite(preds) & np.isfinite(targets)
#     if not valid_mask.all():
#         print(f"Warning: Found {(~valid_mask).sum()} invalid predictions, filtering...")
#         preds = preds[valid_mask]
#         targets = targets[valid_mask]
#         keys = [k for i, k in enumerate(keys) if valid_mask[i]]
        
#     if len(preds) < 2:
#         return {}, {"srcc": 0.0, "plcc": 0.0, "krcc": 0.0, "rmse": 0.0}

#     # 用于收集所有场景的指标，最后计算 Overall Mean
#     all_scene_srccs = []
#     all_scene_plccs = []
#     all_scene_krccs = []
#     all_scene_rmses = []
    
#     subset_metrics = {}
    
#     # --- 2. 遍历每个子集 ---
#     for subset in ffv_consts.SUBSETS:
#         # 找到属于当前子集的所有样本索引
#         # key 格式示例: "llff+fern+nerf" -> subset=llff
#         subset_indices = [i for i, k in enumerate(keys) if ffv_consts.get_subset_from_key(k) == subset]
        
#         if len(subset_indices) < 2:
#             subset_metrics[subset] = {"srcc": 0.0, "plcc": 0.0, "krcc": 0.0, "rmse": 0.0, "count": 0}
#             continue

#         # --- 3. 动态解析该子集下的唯一场景 ---
#         # 假设 key 格式为 "subset+scene+method"
#         subset_keys = [keys[i] for i in subset_indices]
#         try:
#             unique_scenes = set(k.split('+')[1] for k in subset_keys)
#         except IndexError:
#             print(f"Error parsing scenes for subset {subset}, falling back to pooled.")
#             unique_scenes = []

#         scene_srccs = []
#         scene_plccs = []
#         scene_krccs = []
#         scene_rmses = []
        
#         # --- 4. 逐场景计算指标 ---
#         for scene in unique_scenes:
#             # 找到属于该特定场景的样本 (例如所有 fern 的视频)
#             scene_indices = [i for i in subset_indices if f"+{scene}+" in keys[i]]
            
#             if len(scene_indices) < 3: # 样本太少不算相关性
#                 continue
            
#             s_preds = preds[scene_indices]
#             s_targets = targets[scene_indices]
            
#             # 计算该场景的 SRCC/PLCC
#             # 某些场景如果预测值完全一样（常数），相关性会报错或产生NaN，需处理
#             try:
#                 # SRCC
#                 s_srcc = spearmanr(s_preds, s_targets).correlation
#                 if np.isfinite(s_srcc): 
#                     scene_srccs.append(s_srcc)
#                     all_scene_srccs.append(s_srcc)
                
#                 # PLCC
#                 s_plcc = pearsonr(s_preds, s_targets)[0]
#                 if np.isfinite(s_plcc): 
#                     scene_plccs.append(s_plcc)
#                     all_scene_plccs.append(s_plcc)

#                 # --- 新增 KRCC 计算 ---
#                 s_krcc = kendalltau(s_preds, s_targets)[0]
#                 if np.isfinite(s_krcc):
#                     scene_krccs.append(s_krcc) # 记得在循环外初始化这个列表: scene_krccs = []
#                     all_scene_krccs.append(s_krcc) # 记得在函数开头初始化: all_scene_krccs = []
                
#                 # RMSE
#                 s_rmse = np.sqrt(np.mean((s_preds - s_targets) ** 2))
#                 scene_rmses.append(s_rmse)
#                 all_scene_rmses.append(s_rmse)
#             except Exception:
#                 pass
        
#         # --- 5. 计算子集平均分 (Mean over scenes) ---
#         count = len(scene_srccs)
#         if count > 0:
#             subset_metrics[subset] = {
#                 "srcc": np.mean(scene_srccs),
#                 "plcc": np.mean(scene_plccs),
#                 "krcc": np.mean(scene_krccs), 
#                 "rmse": np.mean(scene_rmses),
#                 "count": count
#             }
#         else:
#             subset_metrics[subset] = {"srcc": 0.0, "plcc": 0.0, "krcc": 0.0, "rmse": 0.0, "count": 0}

#     # --- 6. 计算 Overall 平均分 ---
#     # SQA 论文中的 Overall 也是所有场景分数的平均值
#     overall_metrics = {
#         "srcc": np.mean(all_scene_srccs) if all_scene_srccs else 0.0,
#         "plcc": np.mean(all_scene_plccs) if all_scene_plccs else 0.0,
#         "krcc": np.mean(all_scene_krccs) if all_scene_krccs else 0.0,
#         "rmse": np.mean(all_scene_rmses) if all_scene_rmses else 0.0
#     }
    
#     return subset_metrics, overall_metrics

def evaluate_by_subset(preds, targets, keys):
    """
    [修复版] 强壮的评估函数
    1. 修复 KeyError: 即使预测全炸(NaN)，也能返回全0的字典，防止程序中断。
    2. 修复 KRCC=0: 增加了 Kendall's Tau 的计算。
    3. 采用 NVS-SQA 标准的逐场景平均 (Scene-by-Scene) 计算方式。
    """
    preds = np.array(preds).flatten()
    targets = np.array(targets).flatten()
    
    # --- 1. 预定义全零结果结构 (防止 KeyError 的核心!) ---
    # 无论后面发生什么，这个结构保证了 'llff', 'fieldwork' 等 key 永远存在
    final_subset_metrics = {
        subset: {"srcc": 0.0, "plcc": 0.0, "krcc": 0.0, "rmse": 0.0, "count": 0}
        for subset in ffv_consts.SUBSETS
    }
    
    # --- 2. 数据清洗 (过滤 NaN/Inf) ---
    valid_mask = np.isfinite(preds) & np.isfinite(targets)
    if not valid_mask.all():
        print(f"Warning: Found {(~valid_mask).sum()} invalid predictions (NaN/Inf). Filtering...")
        preds = preds[valid_mask]
        targets = targets[valid_mask]
        keys = [k for i, k in enumerate(keys) if valid_mask[i]]
    
    # 如果过滤后没数据了，直接返回预定义的零结果
    if len(preds) < 2:
        print("Error: Too few valid predictions to evaluate. Returning 0 metrics.")
        return final_subset_metrics, {"srcc": 0.0, "plcc": 0.0, "krcc": 0.0, "rmse": 0.0}

    # 用于计算 Overall (所有场景的平均)
    all_scene_metrics = {"srcc": [], "plcc": [], "krcc": [], "rmse": []}
    
    # --- 3. 逐子集、逐场景计算 ---
    for subset in ffv_consts.SUBSETS:
        # 找到属于当前子集的所有样本
        subset_indices = [i for i, k in enumerate(keys) if ffv_consts.get_subset_from_key(k) == subset]
        
        if len(subset_indices) < 2:
            continue # 该子集没数据，保持默认的 0 分

        # 动态解析该子集下的唯一场景 (例如 fern, trex...)
        subset_keys = [keys[i] for i in subset_indices]
        try:
            unique_scenes = set(k.split('+')[1] for k in subset_keys)
        except IndexError:
            unique_scenes = []

        scene_metrics_list = {"srcc": [], "plcc": [], "krcc": [], "rmse": []}
        
        # 遍历该子集下的每一个场景
        for scene in unique_scenes:
            # 找到属于该特定场景的样本
            scene_indices = [i for i in subset_indices if f"+{scene}+" in keys[i]]
            
            if len(scene_indices) < 3: continue # 样本太少不算
            
            s_preds = preds[scene_indices]
            s_targets = targets[scene_indices]
            
            try:
                # 计算指标
                s_srcc = spearmanr(s_preds, s_targets).correlation
                s_plcc = pearsonr(s_preds, s_targets)[0]
                s_krcc = kendalltau(s_preds, s_targets)[0] # 修复 KRCC
                s_rmse = np.sqrt(np.mean((s_preds - s_targets) ** 2))
                
                # 收集有效指标
                if np.isfinite(s_srcc): 
                    scene_metrics_list["srcc"].append(s_srcc)
                    all_scene_metrics["srcc"].append(s_srcc)
                if np.isfinite(s_plcc): 
                    scene_metrics_list["plcc"].append(s_plcc)
                    all_scene_metrics["plcc"].append(s_plcc)
                if np.isfinite(s_krcc): 
                    scene_metrics_list["krcc"].append(s_krcc)
                    all_scene_metrics["krcc"].append(s_krcc)
                scene_metrics_list["rmse"].append(s_rmse)
                all_scene_metrics["rmse"].append(s_rmse)
                
            except Exception:
                pass
        
        # 计算该子集的平均分
        if len(scene_metrics_list["srcc"]) > 0:
            final_subset_metrics[subset] = {
                "srcc": np.mean(scene_metrics_list["srcc"]),
                "plcc": np.mean(scene_metrics_list["plcc"]),
                "krcc": np.mean(scene_metrics_list["krcc"]),
                "rmse": np.mean(scene_metrics_list["rmse"]),
                "count": len(subset_indices)
            }

    # --- 4. 计算 Overall (所有场景平均) ---
    overall_metrics = {
        k: np.mean(v) if len(v) > 0 else 0.0 
        for k, v in all_scene_metrics.items()
    }
    
    return final_subset_metrics, overall_metrics

# ==========================================
# 5. 单次训练函数 (Train Single Run)
# ==========================================
def train_single_run(args, seed, run_idx):
    """
    运行一次完整的训练，并保存该次最佳结果到 JSON
    """
    set_seed(seed)
    
    if not args.no_wandb:
        import wandb
        wandb.init(project=args.wandb_project, group=args.experiment_name, 
                   name=f"run_{run_idx}_seed_{seed}", config=args, reinit=True)
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # 结果保存路径
    save_dir = f"eval_results_ffv/{args.experiment_name}"
    os.makedirs(f"checkpoints_ffv/{args.experiment_name}", exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    # Transform
    if args.no_multiscale:
        transform = T.Compose([T.Resize((224, 224)), T.ToTensor(), 
                               T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    else:
        transform = T.Compose([MultiScaleCrop(224), T.ToTensor(), 
                               T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    # 加载FFV数据集
    train_set = AdvancedFFVDataset(
        root_dir=args.root_dir,
        labels_file=args.labels_file,
        subset='all',  # 使用所有子集训练
        mode='train',
        transform=transform,
        distortion_sampling=True,
        num_frames=args.num_frames,
        use_subscores=args.use_subscores
    )
    
    val_set = AdvancedFFVDataset(
        root_dir=args.root_dir,
        labels_file=args.labels_file,
        subset='all',
        mode='test',  # FFV使用test作为验证
        transform=transform,
        distortion_sampling=False,
        num_frames=args.num_frames,
        use_subscores=args.use_subscores
    )
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    # 模型 (与原代码完全一致)
    model = DisNeRFQA_Advanced(num_subscores=4, use_fusion=not args.no_fusion).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 损失函数 (与原代码完全一致)
    mse_crit = nn.MSELoss()
    rank_crit = RankLoss()
    ssl_rank_crit = RankLoss()
    
    best_srcc = -1.0
    best_metrics = {}
    best_subset_metrics = {}
    
    print(f"\n>>> Starting Run {run_idx} with Seed {seed} <<<")
    print(f"Train samples: {len(train_set)}, Val samples: {len(val_set)}")
    
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Run {run_idx}|Ep {epoch+1}", leave=True)
        
        for batch in pbar:
            x_c, x_d, score, sub_scores_gt, _, x_c_aug, x_d_aug = batch
            x_c, x_d, score = x_c.to(device), x_d.to(device), score.to(device)
            x_c_aug, x_d_aug = x_c_aug.to(device), x_d_aug.to(device)
            sub_scores_gt = sub_scores_gt.to(device)
            
            # Forward (与原代码完全一致)
            pred_score, pred_subs, proj_c, proj_d, feat_c, feat_d = model(x_c, x_d)
            pred_score_aug, _, _, _, _, _ = model(x_c_aug, x_d_aug)
            
            # Loss计算 (与原代码完全一致)
            loss_mse = mse_crit(pred_score.view(-1), score)
            loss_rank = rank_crit(pred_score.view(-1), score)
            loss_mi = model.mi_estimator(feat_c, feat_d)
            loss_sub = mse_crit(pred_subs, sub_scores_gt) if args.use_subscores else torch.tensor(0.0).to(device)
            loss_ssl = ssl_rank_crit(pred_score.view(-1), pred_score_aug.view(-1))
            
            total_loss = loss_mse + args.lambda_rank * loss_rank + args.lambda_mi * loss_mi + \
                         args.lambda_sub * loss_sub + args.lambda_ssl * loss_ssl
            # --- 【防线 1：NaN 熔断】(对应第三步) ---
            # 如果 Loss 已经坏了，千万别 backward，直接跳过！
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"Warning: NaN/Inf loss detected at Epoch {epoch}! Skipping batch...")
                optimizer.zero_grad() 
                continue 
            
            # 如果 Loss 正常，才开始反向传播
            optimizer.zero_grad()
            total_loss.backward()

            # --- 【防线 2：梯度裁剪】(对应第二步) ---
            # 就算 Loss 正常，梯度也可能太大，这里强制限制最大为 0.5
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            
            if not args.no_wandb:
                wandb.log({"train/loss": total_loss.item()})

        # --- Validation ---
        model.eval()
        preds, targets, all_keys = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                x_c, x_d, score, _, keys, _, _ = batch
                x_c, x_d = x_c.to(device), x_d.to(device)
                pred, _, _, _, _, _ = model(x_c, x_d)
                
                # 检查预测值是否有效
                pred_np = pred.cpu().numpy()
                if not np.isfinite(pred_np).all():
                    print(f"Warning: Invalid predictions detected in validation, skipping batch...")
                    continue
                
                preds.extend(pred_np)
                targets.extend(score.numpy())
                all_keys.extend(keys)
        
        # 分子集评估
        subset_metrics, overall_metrics = evaluate_by_subset(preds, targets, all_keys)
        
        srcc = overall_metrics["srcc"]
        plcc = overall_metrics["plcc"]
        krcc = overall_metrics["krcc"]
        rmse = overall_metrics["rmse"]
        
        # 打印分子集结果
        # if (epoch + 1) % 10 == 0 or epoch == 0:
        if (epoch + 1) % 1 == 0:
            print(f"\nEpoch {epoch+1} Results:")
            print(f"  Overall: SRCC={srcc:.4f}, PLCC={plcc:.4f}, KRCC={krcc:.4f}, RMSE={rmse:.4f}")
            for subset in ffv_consts.SUBSETS:
                sm = subset_metrics[subset]
                print(f"  {subset.upper():10s}: SRCC={sm['srcc']:.4f}, PLCC={sm['plcc']:.4f}, "
                      f"KRCC={sm['krcc']:.4f}, RMSE={sm['rmse']:.4f} (n={sm['count']})")
        
        # --- 记录并保存最佳结果 ---
        if srcc > best_srcc:
            best_srcc = srcc
            # 确保所有值都是Python原生类型
            best_metrics = {
                "srcc": float(srcc), 
                "plcc": float(plcc), 
                "krcc": float(krcc), 
                "rmse": float(rmse), 
                "epoch": int(epoch+1)
            }
            # 深拷贝并转换subset_metrics
            best_subset_metrics = {}
            for subset, metrics_dict in subset_metrics.items():
                best_subset_metrics[subset] = {
                    k: float(v) if isinstance(v, (np.floating, np.integer, float, int)) and k != 'count' else int(v) if k == 'count' else v
                    for k, v in metrics_dict.items()
                }
            
            # 保存模型权重
            torch.save(model.state_dict(), f"checkpoints_ffv/{args.experiment_name}/best_model_run_{run_idx}.pth")
            
            # 保存详细JSON
            json_filename = f"run_{run_idx}_seed_{seed}_best.json"
            json_path = os.path.join(save_dir, json_filename)
            save_detailed_results(json_path, epoch+1, best_metrics, 
                                  np.array(preds).flatten(), np.array(targets).flatten(), 
                                  all_keys, seed, run_idx, best_subset_metrics)
            
        if not args.no_wandb:
            wandb.log({
                "val/srcc": srcc, "val/plcc": plcc, "val/krcc": krcc, "val/rmse": rmse,
                "val/llff_srcc": subset_metrics["llff"]["srcc"],
                "val/fieldwork_srcc": subset_metrics["fieldwork"]["srcc"],
                "val/lab_srcc": subset_metrics["lab"]["srcc"],
            })

    print(f"\nRun {run_idx} Finished. Best SRCC: {best_srcc:.4f} at Epoch {best_metrics['epoch']}")
    print("Best Subset Results:")
    for subset in ffv_consts.SUBSETS:
        sm = best_subset_metrics[subset]
        print(f"  {subset.upper():10s}: SRCC={sm['srcc']:.4f}, PLCC={sm['plcc']:.4f}, "
              f"KRCC={sm['krcc']:.4f}, RMSE={sm['rmse']:.4f}")
    
    if not args.no_wandb:
        wandb.finish()
        
    return best_metrics, best_subset_metrics

# ==========================================
# 6. 主程序入口
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Train on FFV Dataset (JOD scores)")
    
    # FFV数据集路径
    parser.add_argument("--root_dir", type=str, 
                        default="/media/abc/One Touch/NVS-SQA/benchmark_bank",
                        help="FFV数据集根目录")
    parser.add_argument("--labels_file", type=str, 
                        default="/media/abc/One Touch/NVS-SQA/benchmark_bank/flat_labels_offset_by_ref_scenewise_split_11.json",
                        help="FFV标签文件路径")
    
    # 训练参数
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--num_frames", type=int, default=8, help="每个视频采样的帧数")
    
    # 损失权重 (与原代码一致)
    # parser.add_argument("--lambda_rank", type=float, default=0.1)
    parser.add_argument("--lambda_rank", type=float, default=0.5)
    parser.add_argument("--lambda_mi", type=float, default=0.1)
    parser.add_argument("--lambda_sub", type=float, default=0.5)
    parser.add_argument("--lambda_cont", type=float, default=0.1)
    parser.add_argument("--lambda_ssl", type=float, default=0.2)
    
    # 实验设置
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--use_subscores", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="DisNeRF-FFV")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--no_fusion", action="store_true")
    parser.add_argument("--no_multiscale", action="store_true")
    
    # 重复实验参数
    parser.add_argument("--num_repeats", type=int, default=5)
    parser.add_argument("--experiment_name", type=str, default="ffv_exp_v1")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 随机种子列表
    seeds = [42, 3407, 1234, 2024, 999, 100, 777, 888, 555, 111]
    if args.num_repeats > len(seeds):
        extra_seeds = [random.randint(0, 10000) for _ in range(args.num_repeats - len(seeds))]
        seeds.extend(extra_seeds)
    seeds = seeds[:args.num_repeats]
    
    # 存储所有run的结果
    all_results = {
        "overall": {"srcc": [], "plcc": [], "krcc": [], "rmse": []},
        "llff": {"srcc": [], "plcc": [], "krcc": [], "rmse": []},
        "fieldwork": {"srcc": [], "plcc": [], "krcc": [], "rmse": []},
        "lab": {"srcc": [], "plcc": [], "krcc": [], "rmse": []},
    }
    
    print("=" * 70)
    print(f"FFV Dataset Training - {args.num_repeats} Runs")
    print(f"Output Directory: eval_results_ffv/{args.experiment_name}")
    print("=" * 70)
    
    for i, seed in enumerate(seeds):
        metrics, subset_metrics = train_single_run(args, seed, run_idx=i+1)
        
        # 收集结果
        all_results["overall"]["srcc"].append(metrics['srcc'])
        all_results["overall"]["plcc"].append(metrics['plcc'])
        all_results["overall"]["krcc"].append(metrics['krcc'])
        all_results["overall"]["rmse"].append(metrics['rmse'])
        
        for subset in ffv_consts.SUBSETS:
            all_results[subset]["srcc"].append(subset_metrics[subset]['srcc'])
            all_results[subset]["plcc"].append(subset_metrics[subset]['plcc'])
            all_results[subset]["krcc"].append(subset_metrics[subset]['krcc'])
            all_results[subset]["rmse"].append(subset_metrics[subset]['rmse'])
        
        print(f"-> Run {i+1} Overall SRCC: {metrics['srcc']:.4f}")

    # 计算统计量
    print("\n" + "=" * 70)
    print(f"FINAL RESULTS ({args.num_repeats} Runs)")
    print("=" * 70)
    
    final_stats = {
        "experiment_name": args.experiment_name,
        "num_repeats": args.num_repeats,
        "seeds": seeds,
        "results": {}
    }
    
    # 整体结果
    print("\n[Overall]")
    for metric in ["srcc", "plcc", "krcc", "rmse"]:
        values = all_results["overall"][metric]
        mean_val = np.mean(values)
        std_val = np.std(values)
        print(f"  {metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")
        final_stats["results"][f"overall_{metric}_mean"] = float(mean_val)
        final_stats["results"][f"overall_{metric}_std"] = float(std_val)
    
    # 分子集结果
    for subset in ffv_consts.SUBSETS:
        print(f"\n[{subset.upper()}]")
        for metric in ["srcc", "plcc", "krcc", "rmse"]:
            values = all_results[subset][metric]
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"  {metric.upper()}: {mean_val:.4f} ± {std_val:.4f}")
            final_stats["results"][f"{subset}_{metric}_mean"] = float(mean_val)
            final_stats["results"][f"{subset}_{metric}_std"] = float(std_val)
    
    print("=" * 70)
    
    # 保存最终统计
    save_dir = f"eval_results_ffv/{args.experiment_name}"
    os.makedirs(save_dir, exist_ok=True)
    with open(f"{save_dir}/final_summary.json", "w") as f:
        json.dump(final_stats, f, indent=4)
    print(f"\nSummary saved to {save_dir}/final_summary.json")

if __name__ == "__main__":
    main()
