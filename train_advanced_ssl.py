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
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
from datetime import datetime
import random

# 引入你的自定义模块
# 假设目录结构没有变，确保这些模块能被 Python 找到
from datasets.of_nerf import OFNeRFDataset
from models.dis_nerf_advanced import DisNeRFQA_Advanced
from utils import calculate_srcc, calculate_plcc, calculate_krcc

# 防止导入报错的占位符
try:
    from train import RankLoss, mi_loss_fn
except ImportError:
    print("Warning: Could not import RankLoss/mi_loss_fn from train.py. Using placeholders.")
    # 一个简单的 RankLoss 实现，用于自监督排序
    class RankLoss(nn.Module):
        def forward(self, preds_high, preds_low):
            # 目标：preds_high > preds_low
            # Loss = max(0, preds_low - preds_high + margin)
            return torch.mean(torch.relu(preds_low - preds_high + 0.1))
            
    def mi_loss_fn(mu, logvar): return torch.tensor(0.0).to(mu.device)

# --- 增强工具类：实现 PDF 中的混合失真模拟 ---
class SelfSupervisedAugmentor:
    def __init__(self):
        # 光度失真：亮度、对比度、饱和度、色相
        self.photo_jitter = T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1)
        
    def add_geometric_noise(self, img):
        """注入几何伪影（模糊、像素化）"""
        choice = random.choice(['blur', 'pixelate'])
        if choice == 'blur':
            # 高斯模糊
            radius = random.uniform(1, 3)
            return img.filter(ImageFilter.GaussianBlur(radius))
        elif choice == 'pixelate':
            # 像素化：缩小再放大
            w, h = img.size
            ratio = random.uniform(0.2, 0.5)
            img_small = img.resize((int(w*ratio), int(h*ratio)), resample=Image.NEAREST)
            return img_small.resize((w, h), resample=Image.NEAREST)
        return img

    def add_photometric_noise(self, img):
        """注入光度失真"""
        return self.photo_jitter(img)

    def __call__(self, frames):
        """
        对视频帧序列应用一致的增强
        frames: list of PIL Images
        """
        # 随机决定应用哪种失真，或者混合应用
        # PDF 提到：在几何流中学习对光度变化的不变性，在光度流中学习对几何变化的敏感性
        # 这里为了构建负样本（质量更差），我们随机应用一种或多种
        
        augmented_frames = []
        
        # 随机种子，保证同一视频序列的增强也是连贯的（可选）
        apply_photo = random.random() > 0.3
        apply_geo = random.random() > 0.3
        if not apply_photo and not apply_geo:
            apply_photo = True # 至少应用一种
            
        # 针对序列中的每一帧
        for img in frames:
            res = img
            if apply_geo:
                res = self.add_geometric_noise(res)
            if apply_photo:
                res = self.add_photometric_noise(res)
            augmented_frames.append(res)
            
        return augmented_frames

# --- 1. 修改 Dataset：增加自监督样本构建 ---
class AdvancedOFNeRFDataset(OFNeRFDataset):
    def __init__(self, root_dir, mos_file, mode='train', transform=None, distortion_sampling=False, num_frames=8, use_subscores=False):
        super().__init__(root_dir, mos_file, mode, transform, distortion_sampling, num_frames)
        self.use_subscores = use_subscores
        
        # 初始化自监督增强器
        self.augmentor = SelfSupervisedAugmentor()

    def __getitem__(self, idx):
        folder_path = self.valid_samples[idx]
        key = self._get_key_from_path(folder_path)
        
        # --- 获取标签 ---
        entry = self.mos_labels[key]
        if isinstance(entry, dict):
            score = entry['mos'] / 100.0
            sub_data = entry.get('sub_scores', {})
        else:
            score = entry / 100.0
            sub_data = {}
        
        score_tensor = torch.tensor(score, dtype=torch.float32)

        sub_scores_tensor = torch.zeros(4, dtype=torch.float32)
        if self.use_subscores:
            sub_scores_tensor = torch.tensor([
                sub_data.get("discomfort", 0),
                sub_data.get("blur", 0),
                sub_data.get("lighting", 0),
                sub_data.get("artifacts", 0)
            ], dtype=torch.float32) / 5.0
        
        # --- 加载原始帧 (PIL Image list) ---
        # 注意：这里我们修改 _load_frames 让它返回 PIL 列表而不是 Tensor，
        # 因为我们需要先做增强，再做 Transform
        frames_pil = self._load_frames_pil(folder_path)
        
        # --- 生成自监督增强样本 (SSL Sample) ---
        # 仅在训练模式下生成增强样本，验证模式不需要（或者是可选的）
        if self.mode == 'train':
            frames_aug_pil = self.augmentor(frames_pil)
        else:
            frames_aug_pil = frames_pil # 占位，验证集不用
        
        # --- 应用 Transform (ToTensor, Normalize, MultiScaleCrop) ---
        # 原始流
        content_input = self._apply_transform(frames_pil)
        # 增强流
        content_input_aug = self._apply_transform(frames_aug_pil)

        # --- 失真采样 (Distortion Sampling / Grid Mini-Patch) ---
        if self.distortion_sampling:
            distortion_input = self._grid_mini_patch_sampling(content_input)
            distortion_input_aug = self._grid_mini_patch_sampling(content_input_aug)
        else:
            distortion_input = content_input
            distortion_input_aug = content_input_aug
            
        # 返回：原始Content, 原始Distortion, 分数, 子分数, Key, 增强Content, 增强Distortion
        return content_input, distortion_input, score_tensor, sub_scores_tensor, key, content_input_aug, distortion_input_aug

    def _load_frames_pil(self, folder_path):
        """
        辅助函数：只加载 PIL 图片，不转 Tensor，方便做增强
        """
        all_frames = sorted(list(folder_path.glob("frame_*.png")))
        if not all_frames:
            # 兼容 jpg
            all_frames = sorted(list(folder_path.glob("frame_*.jpg")))
        if not all_frames:
             # 如果找不到，尝试直接列出所有图片
            all_frames = sorted([f for f in folder_path.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
            
        if not all_frames:
             raise ValueError(f"No frames found in {folder_path}")

        # 均匀采样
        indices = torch.linspace(0, len(all_frames)-1, self.num_frames).long()
        selected_frames = [all_frames[i] for i in indices]
        
        imgs = []
        for p in selected_frames:
            img = Image.open(p).convert('RGB')
            imgs.append(img)
        return imgs

    def _apply_transform(self, pil_list):
        """对帧列表应用 transform 并 stack"""
        t_imgs = []
        for img in pil_list:
            if self.transform:
                res = self.transform(img)
                t_imgs.append(res)
            else:
                t_imgs.append(T.ToTensor()(img))
        return torch.stack(t_imgs) # [T, C, H, W]

# --- Transforms ---
class MultiScaleCrop:
    def __init__(self, size=224):
        self.size = size
    def __call__(self, img):
        # 随机选择尺度，模拟 PDF 中的 "Multi-scale Random Crop"
        scale = int(np.random.choice([224, 256, 288]))
        img = T.Resize(scale)(img)
        img = T.RandomCrop(self.size)(img)
        return img

# --- Main Script ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train Dis-NeRF-VQA Advanced with SSL")
    parser.add_argument("--root_dir", type=str, default="../renders")
    parser.add_argument("--mos_file", type=str, default="mos_advanced.json")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    # 损失权重
    parser.add_argument("--lambda_rank", type=float, default=0.1)
    parser.add_argument("--lambda_mi", type=float, default=0.1)
    parser.add_argument("--lambda_sub", type=float, default=0.5)
    parser.add_argument("--lambda_cont", type=float, default=0.1)
    # 新增：自监督损失权重
    parser.add_argument("--lambda_ssl", type=float, default=0.2, help="Weight for Self-Supervised Ranking Loss")
    
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--use_subscores", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Dis-NeRF-VQA-Adv-SSL")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--no_fusion", action="store_true")
    parser.add_argument("--no_multiscale", action="store_true")
    return parser.parse_args()

def save_detailed_results(save_path, epoch, metrics, preds, targets, keys, timestamp):
    per_video_results = []
    test_videos_list = []
    for i, key in enumerate(keys):
        test_videos_list.append(key)
        per_video_results.append({
            "video_name": key,
            "predicted_mos": float(preds[i] * 100), 
            "ground_truth_mos": float(targets[i] * 100)
        })

    output_data = {
        "evaluation_info": {
            "timestamp": timestamp,
            "checkpoint": f"epoch_{epoch}",
            "num_test_videos": len(keys)
        },
        "metrics": {
            "SRCC": float(metrics["srcc"]),
            "PLCC": float(metrics["plcc"]),
            "KRCC": float(metrics["krcc"]),
            "RMSE": float(metrics["rmse"]) 
        },
        "test_videos": test_videos_list,
        "per_video_results": per_video_results
    }

    with open(save_path, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"Results JSON saved to: {save_path}")

def train(args):
    if not args.no_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=args)
        
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    save_dir = "eval_results_ssl"
    os.makedirs("checkpoints_advanced_ssl", exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Transforms
    if args.no_multiscale:
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = T.Compose([
            MultiScaleCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Datasets
    # 训练集开启 distortion_sampling 和 自监督增强
    train_set = AdvancedOFNeRFDataset(args.root_dir, args.mos_file, mode='train', transform=transform, distortion_sampling=True, use_subscores=args.use_subscores)
    # 验证集通常不需要 distortion_sampling (根据你的原始逻辑)，也不需要增强
    val_set = AdvancedOFNeRFDataset(args.root_dir, args.mos_file, mode='val', transform=transform, distortion_sampling=False, use_subscores=args.use_subscores)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    model = DisNeRFQA_Advanced(num_subscores=4, use_fusion=not args.no_fusion).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    mse_crit = nn.MSELoss()
    rank_crit = RankLoss()
    ssl_rank_crit = RankLoss() # 自监督专用的排序 Loss
    
    best_srcc = -1.0
    
    for epoch in range(args.epochs):
        # --- Training ---
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for x_c, x_d, score, sub_scores_gt, _, x_c_aug, x_d_aug in pbar:
            # 数据迁移到 GPU
            x_c, x_d, score = x_c.to(device), x_d.to(device), score.to(device)
            x_c_aug, x_d_aug = x_c_aug.to(device), x_d_aug.to(device)
            sub_scores_gt = sub_scores_gt.to(device)
            
            # 1. 原始数据的 Forward
            pred_score, pred_subs, proj_c, proj_d, feat_c, feat_d = model(x_c, x_d)
            
            # 2. 增强数据的 Forward (用于自监督)
            # 我们不需要增强数据的子分数或特征做其他 Loss，只需要预测分数进行排序对比
            # 注意：增强后的样本应当被认为质量更差 (Lower Quality)
            pred_score_aug, _, _, _, _, _ = model(x_c_aug, x_d_aug)
            
            # --- Losses Calculation ---
            
            # (1) 主任务 MSE Loss
            loss_mse = mse_crit(pred_score.view(-1), score)
            
            # (2) 原始数据的 Ranking Loss (如果有 Batch 内的 Pair)
            loss_rank = rank_crit(pred_score.view(-1), score)
            
            # (3) MI Loss
            loss_mi = model.mi_estimator(feat_c, feat_d)
            
            # (4) Sub-score Loss
            loss_sub = torch.tensor(0.0).to(device)
            if args.use_subscores:
                loss_sub = mse_crit(pred_subs, sub_scores_gt)
                
            # (5) [NEW] 自监督排序 Loss (SSL Loss)
            # 逻辑：原始图像质量 (pred_score) 应 > 增强图像质量 (pred_score_aug)
            # 使用 rank_crit 计算，目标是 pred_score > pred_score_aug
            loss_ssl = ssl_rank_crit(pred_score.view(-1), pred_score_aug.view(-1))
            
            # 总 Loss
            total_loss = loss_mse + \
                         args.lambda_rank * loss_rank + \
                         args.lambda_mi * loss_mi + \
                         args.lambda_sub * loss_sub + \
                         args.lambda_ssl * loss_ssl  # 加入自监督 Loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            pbar.set_postfix({
                'loss': total_loss.item(), 
                'ssl': loss_ssl.item()
            })
            if not args.no_wandb:
                wandb.log({
                    "train/loss": total_loss.item(),
                    "train/ssl_loss": loss_ssl.item()
                })
        
        # --- Validation (保持不变) ---
        model.eval()
        preds, targets, all_keys = [], [], []
        with torch.no_grad():
            for x_c, x_d, score, _, keys, _, _ in val_loader: # 注意 unpack 增加的占位符
                x_c, x_d = x_c.to(device), x_d.to(device)
                pred, _, _, _, _, _ = model(x_c, x_d)
                
                preds.extend(pred.cpu().numpy())
                targets.extend(score.numpy())
                all_keys.extend(keys)
        
        preds_arr = np.array(preds).flatten()
        targets_arr = np.array(targets).flatten()

        srcc = calculate_srcc(preds_arr, targets_arr)
        plcc = calculate_plcc(preds_arr, targets_arr)
        krcc = calculate_krcc(preds_arr, targets_arr)
        rmse = np.sqrt(np.mean((preds_arr*100 - targets_arr*100)**2))

        print(f"Epoch {epoch+1} -> SRCC: {srcc:.4f} | PLCC: {plcc:.4f} | KRCC: {krcc:.4f}")
        
        if not args.no_wandb:
            wandb.log({"val/srcc": srcc, "val/plcc": plcc, "val/krcc": krcc, "epoch": epoch + 1})

        # Save Best
        metrics = {"srcc": srcc, "plcc": plcc, "krcc": krcc, "rmse": rmse}
        
        if srcc > best_srcc:
            best_srcc = srcc
            print(f"*** New Best SRCC: {best_srcc:.4f} ***")
            torch.save(model.state_dict(), f"checkpoints_advanced_ssl/best_model_{run_timestamp}.pth")
            
            save_path = os.path.join(save_dir, f"eval_best_model_{run_timestamp}.json")
            save_detailed_results(save_path, epoch+1, metrics, preds_arr, targets_arr, all_keys, run_timestamp)

        if epoch + 1 == args.epochs:
            final_save_path = os.path.join(save_dir, f"eval_final_epoch_{epoch+1}_{run_timestamp}.json")
            save_detailed_results(final_save_path, epoch+1, metrics, preds_arr, targets_arr, all_keys, run_timestamp)

if __name__ == "__main__":
    args = parse_args()
    train(args)