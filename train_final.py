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
# 请确保目录结构正确，能够导入这些模块
from datasets.of_nerf import OFNeRFDataset
from models.dis_nerf_advanced import DisNeRFQA_Advanced
from utils import calculate_srcc, calculate_plcc, calculate_krcc

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
def save_detailed_results(save_path, epoch, metrics, preds, targets, keys, seed, run_idx):
    """
    保存单次 Run 的详细结果，包含每个视频的预测值
    """
    per_video_results = []
    test_videos_list = []
    
    # 将 numpy 类型转为 python 原生类型，防止 json 报错
    preds = np.array(preds).flatten()
    targets = np.array(targets).flatten()
    
    for i, key in enumerate(keys):
        test_videos_list.append(key)
        per_video_results.append({
            "video_name": key,
            "predicted_mos": float(preds[i] * 100), # 恢复到 0-100 分制方便查看
            "ground_truth_mos": float(targets[i] * 100)
        })

    output_data = {
        "run_info": {
            "run_index": run_idx,
            "seed": seed,
            "best_epoch": epoch
        },
        "metrics": {
            "SRCC": float(metrics["srcc"]),
            "PLCC": float(metrics["plcc"]),
            "KRCC": float(metrics["krcc"]),
            "RMSE": float(metrics.get("rmse", 0.0)) 
        },
        "per_video_results": per_video_results
    }

    with open(save_path, "w") as f:
        json.dump(output_data, f, indent=4)
    # print(f"  -> Detail saved: {os.path.basename(save_path)}") # 减少刷屏，可注释

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
# 2. 自监督增强模块
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
# 3. Dataset
# ==========================================
class AdvancedOFNeRFDataset(OFNeRFDataset):   
    # [修改 1] __init__ 增加 enable_ssl 参数，默认为 True
    def __init__(self, root_dir, mos_file, mode='train', transform=None, distortion_sampling=False, num_frames=8, use_subscores=False, enable_ssl=True):
        super().__init__(root_dir, mos_file, mode, transform, distortion_sampling, num_frames)

        self.mode = mode
        self.use_subscores = use_subscores
        self.enable_ssl = enable_ssl # [新增] 保存状态
        self.augmentor = SelfSupervisedAugmentor()

    def __getitem__(self, idx):
        folder_path = self.valid_samples[idx]
        key = self._get_key_from_path(folder_path)
        
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
                sub_data.get("discomfort", 0), sub_data.get("blur", 0),
                sub_data.get("lighting", 0), sub_data.get("artifacts", 0)
            ], dtype=torch.float32) / 5.0
        
        frames_pil = self._load_frames_pil(folder_path)
        
        # 修改后：必须是训练模式，并且开启了 SSL 开关，才做增强
        if self.mode == 'train' and self.enable_ssl:
            frames_aug_pil = self.augmentor(frames_pil)
        else:
            frames_aug_pil = frames_pil
        
        content_input = self._apply_transform(frames_pil)
        content_input_aug = self._apply_transform(frames_aug_pil)

        if self.distortion_sampling:
            distortion_input = self._grid_mini_patch_sampling(content_input)
            distortion_input_aug = self._grid_mini_patch_sampling(content_input_aug)
        else:
            distortion_input = content_input
            distortion_input_aug = content_input_aug
            
        return content_input, distortion_input, score_tensor, sub_scores_tensor, key, content_input_aug, distortion_input_aug

    def _load_frames_pil(self, folder_path):
        all_frames = sorted(list(folder_path.glob("frame_*.png")))
        if not all_frames: all_frames = sorted(list(folder_path.glob("frame_*.jpg")))
        if not all_frames: all_frames = sorted([f for f in folder_path.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']])
        if not all_frames: raise ValueError(f"No frames found in {folder_path}")

        indices = torch.linspace(0, len(all_frames)-1, self.num_frames).long()
        selected_frames = [all_frames[i] for i in indices]
        return [Image.open(p).convert('RGB') for p in selected_frames]

    def _apply_transform(self, pil_list):
        t_imgs = []
        for img in pil_list:
            t_imgs.append(self.transform(img) if self.transform else T.ToTensor()(img))
        return torch.stack(t_imgs)

# --- Transforms ---
class MultiScaleCrop:
    def __init__(self, size=224): self.size = size
    def __call__(self, img):
        scale = int(np.random.choice([224, 256, 288]))
        img = T.Resize(scale)(img)
        img = T.RandomCrop(self.size)(img)
        return img

# ==========================================
# 4. 单次训练函数 (Train Single Run)
# ==========================================
def train_single_run(args, seed, run_idx):
    """
    运行一次完整的训练，并保存该次最佳结果到 JSON
    """
    set_seed(seed)
    
    if not args.no_wandb:
        import wandb
        wandb.init(project=args.wandb_project, group=args.experiment_name, name=f"run_{run_idx}_seed_{seed}", config=args, reinit=True)
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    
    # 结果保存路径
    save_dir = f"eval_results_repeated/{args.experiment_name}"
    os.makedirs(f"checkpoints_repeated/{args.experiment_name}", exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # ============ [新增/修改 START] 消融实验逻辑控制 ============
    # 1. 消融 SSL (自监督): 如果 args.no_ssl 为 True，则 dataset 不增强，且 lambda_ssl = 0
    current_enable_ssl = not args.no_ssl
    current_lambda_ssl = 0.0 if args.no_ssl else args.lambda_ssl
    # 2. 消融 Decoupling (解耦): 如果 args.no_decouple 为 True，则 lambda_mi = 0
    current_lambda_mi = 0.0 if args.no_decouple else args.lambda_mi
    # 3. 消融 Multi-task (多任务): 如果 args.no_multitask 为 True，则 lambda_sub = 0，且不使用子分数
    current_use_subscores = False if args.no_multitask else args.use_subscores
    current_lambda_sub = 0.0 if args.no_multitask else args.lambda_sub
    
    # ============ [新增/修改 END] ============
    if args.no_multiscale:
        transform = T.Compose([T.Resize((224, 224)), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    else:
        transform = T.Compose([MultiScaleCrop(224), T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
   # ============ [修改 Dataset 初始化] ============
    # 传入上面计算好的 current_use_subscores 和 current_enable_ssl
    train_set = AdvancedOFNeRFDataset(
        args.root_dir, args.mos_file, mode='train', transform=transform, 
        distortion_sampling=True, 
        use_subscores=current_use_subscores, # [修改] 使用动态变量
        enable_ssl=current_enable_ssl        # [修改] 使用动态变量
    )
    val_set = AdvancedOFNeRFDataset(
        args.root_dir, args.mos_file, mode='val', transform=transform, 
        distortion_sampling=False, 
        use_subscores=current_use_subscores, # [修改] 使用动态变量
        enable_ssl=False                     # Val 永远不需要 SSL 增强
    )
    # ===============================================
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    model = DisNeRFQA_Advanced(num_subscores=4, use_fusion=not args.no_fusion).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    mse_crit = nn.MSELoss()
    rank_crit = RankLoss()
    ssl_rank_crit = RankLoss()
    
    best_srcc = -1.0
    best_metrics = {}
    
    print(f"\n>>> Starting Run {run_idx} with Seed {seed} <<<")
    
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Run {run_idx}|Ep {epoch+1}", leave=False)
        
        for x_c, x_d, score, sub_scores_gt, _, x_c_aug, x_d_aug in pbar:
            x_c, x_d, score = x_c.to(device), x_d.to(device), score.to(device)
            x_c_aug, x_d_aug = x_c_aug.to(device), x_d_aug.to(device)
            sub_scores_gt = sub_scores_gt.to(device)
            
            # 1. 正常的前向传播
            pred_score, pred_subs, proj_c, proj_d, feat_c, feat_d = model(x_c, x_d)
            
            # ============ [修改 START] SSL 计算逻辑 ============
            loss_ssl = torch.tensor(0.0).to(device)
            # 只有在开启 SSL 时，才计算增强数据的 Forward 和 Loss，节省计算资源
            if current_enable_ssl:
                pred_score_aug, _, _, _, _, _ = model(x_c_aug, x_d_aug)
                loss_ssl = ssl_rank_crit(pred_score.view(-1), pred_score_aug.view(-1))
            # ============ [修改 END] ============
            
            loss_mse = mse_crit(pred_score.view(-1), score)
            loss_rank = rank_crit(pred_score.view(-1), score)
            
            # Loss MI (解耦损失)
            loss_mi = model.mi_estimator(feat_c, feat_d)
            
            # ============ [修改 START] 多任务 Loss 逻辑 ============
            loss_sub = torch.tensor(0.0).to(device)
            if current_use_subscores: # 使用动态变量判断
                loss_sub = mse_crit(pred_subs, sub_scores_gt)
            # ============ [修改 END] ============
            
            # ============ [修改 START] Total Loss 权重替换 ============
            # 注意这里全部换成了 current_lambda_xxx
            total_loss = loss_mse + \
                         args.lambda_rank * loss_rank + \
                         current_lambda_mi * loss_mi + \
                         current_lambda_sub * loss_sub + \
                         current_lambda_ssl * loss_ssl
            # ============ [修改 END] ============
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            if not args.no_wandb:
                wandb.log({"train/loss": total_loss.item()})

        # --- Validation ---
        model.eval()
        preds, targets, all_keys = [], [], []
        with torch.no_grad():
            # [修改] 必须接收 keys，因为 Dataset 返回了它
            # unpack 顺序: content, distortion, score, sub_score, KEY, aug_content, aug_distortion
            for x_c, x_d, score, _, keys, _, _ in val_loader:
                x_c, x_d = x_c.to(device), x_d.to(device)
                pred, _, _, _, _, _ = model(x_c, x_d)
                
                preds.extend(pred.cpu().numpy())
                targets.extend(score.numpy())
                all_keys.extend(keys) # 收集文件名
        
        preds_arr = np.array(preds).flatten()
        targets_arr = np.array(targets).flatten()
        srcc = calculate_srcc(preds_arr, targets_arr)
        plcc = calculate_plcc(preds_arr, targets_arr)
        krcc = calculate_krcc(preds_arr, targets_arr)
        rmse = np.sqrt(np.mean((preds_arr*100 - targets_arr*100)**2))
        
        # --- 记录并保存最佳结果 (留痕) ---
        if srcc > best_srcc:
            best_srcc = srcc
            best_metrics = {"srcc": srcc, "plcc": plcc, "krcc": krcc, "rmse": rmse, "epoch": epoch+1}
            
            # 1. 保存模型权重
            torch.save(model.state_dict(), f"checkpoints_repeated/{args.experiment_name}/best_model_run_{run_idx}.pth")
            
            # 2. [新增] 保存这一轮的详细 JSON
            json_filename = f"run_{run_idx}_seed_{seed}_best.json"
            json_path = os.path.join(save_dir, json_filename)
            save_detailed_results(json_path, epoch+1, best_metrics, preds_arr, targets_arr, all_keys, seed, run_idx)
            
        if not args.no_wandb:
            wandb.log({"val/srcc": srcc, "val/plcc": plcc})

    print(f"Run {run_idx} Finished. Best SRCC: {best_srcc:.4f} at Epoch {best_metrics['epoch']}")
    if not args.no_wandb:
        wandb.finish()
        
    return best_metrics

# ==========================================
# 5. 主程序入口
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="Repeated Experiments for Dis-NeRF-VQA")
    parser.add_argument("--root_dir", type=str, default="/media/abc/One Touch/NVS-SQA/TWO/renders")
    parser.add_argument("--mos_file", type=str, default="mos_advanced.json")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    # 权重
    parser.add_argument("--lambda_rank", type=float, default=0.1)
    parser.add_argument("--lambda_mi", type=float, default=0.1)
    parser.add_argument("--lambda_sub", type=float, default=0.5)
    parser.add_argument("--lambda_cont", type=float, default=0.1)
    parser.add_argument("--lambda_ssl", type=float, default=0.2)
    # 实验设置
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--use_subscores", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="Dis-NeRF-Repeated")
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument("--no_fusion", action="store_true")
    parser.add_argument("--no_multiscale", action="store_true")
    # 重复实验参数
    parser.add_argument("--num_repeats", type=int, default=5)
    parser.add_argument("--experiment_name", type=str, default="exp_v1")
    
    # 1. 消融自监督训练集构建 (Exp 1: w/o SSL Set)
    parser.add_argument("--no_ssl", action="store_true", help="Disable Self-Supervised Learning module")
    
    # 2. 消融双流特征解耦 (Exp 2: w/o Decoupling)
    parser.add_argument("--no_decouple", action="store_true", help="Disable Feature Decoupling (MI Loss)")
    
    # 3. 消融多任务联合优化 (Exp 3: w/o Multi-task)
    parser.add_argument("--no_multitask", action="store_true", help="Disable Multi-task (Sub-scores)")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    seeds = [42, 3407, 1234, 2024, 999, 100, 777, 888, 555, 111]
    if args.num_repeats > len(seeds):
        extra_seeds = [random.randint(0, 10000) for _ in range(args.num_repeats - len(seeds))]
        seeds.extend(extra_seeds)
    seeds = seeds[:args.num_repeats]
    
    all_srccs = []
    all_plccs = []
    
    print("="*50)
    print(f"Starting Repeated Experiments: {args.num_repeats} Runs")
    print(f"Output Directory: eval_results_repeated/{args.experiment_name}")
    print("="*50)
    
    for i, seed in enumerate(seeds):
        metrics = train_single_run(args, seed, run_idx=i+1)
        
        all_srccs.append(metrics['srcc'])
        all_plccs.append(metrics['plcc'])
        
        print(f"-> Run {i+1} Result: SRCC={metrics['srcc']:.4f}")

    mean_srcc = np.mean(all_srccs)
    std_srcc = np.std(all_srccs)
    mean_plcc = np.mean(all_plccs)
    std_plcc = np.std(all_plccs)
    
    print("\n" + "="*50)
    print(f"FINAL RESULTS ({args.num_repeats} Runs)")
    print("="*50)
    print(f"SRCC: {mean_srcc:.4f} ± {std_srcc:.4f}")
    print(f"PLCC: {mean_plcc:.4f} ± {std_plcc:.4f}")
    print("="*50)
    
    final_stats = {
        "experiment_name": args.experiment_name,
        "num_repeats": args.num_repeats,
        "seeds": seeds,
        "mean_srcc": float(mean_srcc),
        "std_srcc": float(std_srcc),
        "mean_plcc": float(mean_plcc),
        "std_plcc": float(std_plcc),
        "all_run_srccs": all_srccs
    }
    
    with open(f"eval_results_repeated/{args.experiment_name}/final_summary.json", "w") as f:
        json.dump(final_stats, f, indent=4)
        print(f"Summary saved to eval_results_repeated/{args.experiment_name}/final_summary.json")

if __name__ == "__main__":

    main()
