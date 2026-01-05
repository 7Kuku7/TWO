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
from PIL import Image
from datetime import datetime

# 引入你的自定义模块
from datasets.of_nerf import OFNeRFDataset
from models.dis_nerf_advanced import DisNeRFQA_Advanced
from utils import calculate_srcc, calculate_plcc, calculate_krcc

# 防止导入报错的占位符
try:
    from train import RankLoss, mi_loss_fn
except ImportError:
    print("Warning: Could not import RankLoss/mi_loss_fn from train.py. Using placeholders.")
    class RankLoss(nn.Module):
        def forward(self, preds, target): return torch.tensor(0.0).to(preds.device)
    def mi_loss_fn(mu, logvar): return torch.tensor(0.0).to(mu.device)

# --- 1. 修改 Dataset：必须返回视频名称 (key) ---
class AdvancedOFNeRFDataset(OFNeRFDataset):
    def __init__(self, root_dir, mos_file, mode='train', transform=None, distortion_sampling=False, num_frames=8, use_subscores=False):
        super().__init__(root_dir, mos_file, mode, transform, distortion_sampling, num_frames)
        self.use_subscores = use_subscores

    def __getitem__(self, idx):
        folder_path = self.valid_samples[idx]
        
        # 获取视频的名称 key (例如 "raf_furnishedroom_nerf_baseline_path2")
        key = self._get_key_from_path(folder_path)
        
        entry = self.mos_labels[key]
        
        # 解析 MOS 和 Sub-scores
        if isinstance(entry, dict):
            score = entry['mos'] / 100.0
            sub_data = entry.get('sub_scores', {})
        else:
            score = entry / 100.0
            sub_data = {}
        
        score = torch.tensor(score, dtype=torch.float32)

        sub_scores = torch.zeros(4, dtype=torch.float32)
        if self.use_subscores:
            sub_scores = torch.tensor([
                sub_data.get("discomfort", 0),
                sub_data.get("blur", 0),
                sub_data.get("lighting", 0),
                sub_data.get("artifacts", 0)
            ], dtype=torch.float32) / 5.0
        
        frames = self._load_frames(folder_path)
        
        content_input = frames
        if self.distortion_sampling:
            distortion_input = self._grid_mini_patch_sampling(frames)
        else:
            distortion_input = frames
            
        # !!! 关键修改：返回 key (视频名) !!!
        return content_input, distortion_input, score, sub_scores, key

# --- Transforms ---
class MultiScaleCrop:
    def __init__(self, size=224):
        self.size = size
    def __call__(self, img):
        scale = int(np.random.choice([224, 256, 288]))
        img = T.Resize(scale)(img)
        img = T.RandomCrop(self.size)(img)
        return img

# --- Main Script ---
def parse_args():
    parser = argparse.ArgumentParser(description="Train Dis-NeRF-VQA Advanced")
    parser.add_argument("--root_dir", type=str, default="../renders")
    parser.add_argument("--mos_file", type=str, default="mos_advanced.json")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lambda_rank", type=float, default=0.1)
    parser.add_argument("--lambda_mi", type=float, default=0.1)
    parser.add_argument("--lambda_sub", type=float, default=0.5)
    parser.add_argument("--lambda_cont", type=float, default=0.1)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--use_subscores", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Dis-NeRF-VQA-Adv")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--no_fusion", action="store_true")
    parser.add_argument("--no_multiscale", action="store_true")
    return parser.parse_args()

# --- 2. 新增：保存详细 JSON 的函数（完美复刻截图格式）---
def save_detailed_results(save_path, epoch, metrics, preds, targets, keys, timestamp):
    """
    保存与你截图格式完全一致的 JSON 文件
    """
    # 构造 per_video_results 列表
    per_video_results = []
    test_videos_list = []
    
    for i, key in enumerate(keys):
        test_videos_list.append(key)
        per_video_results.append({
            "video_name": key,
            # 强制转换为原生 float，防止 json 报错
            "predicted_mos": float(preds[i] * 100), 
            "ground_truth_mos": float(targets[i] * 100)
        })

    # 构造最终的大字典
    output_data = {
        "evaluation_info": {
            "timestamp": timestamp,
            "checkpoint": f"epoch_{epoch}",
            "num_test_videos": len(keys)
        },
        "metrics": {
            # !!! 关键修改：在这里加 float() 转换 !!!
            "SRCC": float(metrics["srcc"]),
            "PLCC": float(metrics["plcc"]),
            "KRCC": float(metrics["krcc"]),
            "RMSE": float(metrics["rmse"]) 
        },
        "test_videos": test_videos_list,
        "per_video_results": per_video_results
    }

    # 写入文件
    with open(save_path, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"Results JSON saved to: {save_path}")

def train(args):
    if not args.no_wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_name, config=args)
        
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 使用 eval_results 文件夹，和你截图里的一致
    save_dir = "eval_results"
    os.makedirs("checkpoints_advanced", exist_ok=True)
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
    train_set = AdvancedOFNeRFDataset(args.root_dir, args.mos_file, mode='train', transform=transform, distortion_sampling=True, use_subscores=args.use_subscores)
    val_set = AdvancedOFNeRFDataset(args.root_dir, args.mos_file, mode='val', transform=transform, distortion_sampling=False, use_subscores=args.use_subscores)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    model = DisNeRFQA_Advanced(num_subscores=4, use_fusion=not args.no_fusion).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    mse_crit = nn.MSELoss()
    rank_crit = RankLoss()
    
    best_srcc = -1.0
    
    for epoch in range(args.epochs):
        # --- Training ---
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        # 注意：这里我们不需要 key，用 _ 忽略
        for x_c, x_d, score, sub_scores_gt, _ in pbar:
            x_c, x_d, score = x_c.to(device), x_d.to(device), score.to(device)
            sub_scores_gt = sub_scores_gt.to(device)
            
            pred_score, pred_subs, proj_c, proj_d, feat_c, feat_d = model(x_c, x_d)
            
            loss_mse = mse_crit(pred_score.view(-1), score)
            loss_rank = rank_crit(pred_score.view(-1), score)
            # 改
            # mu, logvar = model.mi_estimator(feat_c, feat_d)
            # loss_mi = mi_loss_fn(mu, logvar)
            loss_mi = model.mi_estimator(feat_c, feat_d)
            
            loss_sub = torch.tensor(0.0).to(device)
            if args.use_subscores:
                loss_sub = mse_crit(pred_subs, sub_scores_gt)
                
            total_loss = loss_mse + args.lambda_rank * loss_rank + \
                         args.lambda_mi * loss_mi + \
                         args.lambda_sub * loss_sub
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            pbar.set_postfix({'loss': total_loss.item()})
            if not args.no_wandb:
                wandb.log({"train/loss": total_loss.item()})
        
        # --- Validation (With Detailed Recording) ---
        model.eval()
        preds, targets, all_keys = [], [], [] # 新增 all_keys
        with torch.no_grad():
            # 这里的 keys 是当前 batch 的视频名列表
            for x_c, x_d, score, _, keys in val_loader:
                x_c, x_d = x_c.to(device), x_d.to(device)
                pred, _, _, _, _, _ = model(x_c, x_d)
                
                preds.extend(pred.cpu().numpy())
                targets.extend(score.numpy())
                all_keys.extend(keys) # 收集所有的视频名
        
        # 计算指标
        preds_arr = np.array(preds).flatten()
        targets_arr = np.array(targets).flatten()

        srcc = calculate_srcc(preds_arr, targets_arr)
        plcc = calculate_plcc(preds_arr, targets_arr)
        krcc = calculate_krcc(preds_arr, targets_arr)
        rmse = np.sqrt(np.mean((preds_arr*100 - targets_arr*100)**2)) # 估算 RMSE (0-100 scale)

        print(f"Epoch {epoch+1} -> SRCC: {srcc:.4f} | PLCC: {plcc:.4f} | KRCC: {krcc:.4f}")
        
        if not args.no_wandb:
            wandb.log({"val/srcc": srcc, "val/plcc": plcc, "val/krcc": krcc, "epoch": epoch + 1})

        # --- 保存逻辑 (Save Logic) ---
        metrics = {"srcc": srcc, "plcc": plcc, "krcc": krcc, "rmse": rmse}
        
        # 1. 保存 "Best" 结果 (如果打破记录)
        if srcc > best_srcc:
            best_srcc = srcc
            print(f"*** New Best SRCC: {best_srcc:.4f} ***")
            torch.save(model.state_dict(), f"checkpoints_advanced/best_model_{run_timestamp}.pth")
            
            # 保存对应的最佳 JSON
            save_path = os.path.join(save_dir, f"eval_best_model_{run_timestamp}.json")
            save_detailed_results(save_path, epoch+1, metrics, preds_arr, targets_arr, all_keys, run_timestamp)

        # 2. 强制保存最后一轮的结果 (无论好坏，防止跑完找不到数据)
        if epoch + 1 == args.epochs:
            final_save_path = os.path.join(save_dir, f"eval_final_epoch_{epoch+1}_{run_timestamp}.json")
            print("Saving Final Epoch Results...")
            save_detailed_results(final_save_path, epoch+1, metrics, preds_arr, targets_arr, all_keys, run_timestamp)

if __name__ == "__main__":
    args = parse_args()
    train(args)