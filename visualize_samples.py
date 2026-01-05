import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as T
from PIL import Image
import os
import math

from datasets.of_nerf import OFNeRFDataset
from models.dis_nerf_advanced import DisNeRFQA_Advanced
from train_advanced import AdvancedOFNeRFDataset

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Dis-NeRF-VQA Predictions")
    parser.add_argument("--root_dir", type=str, default="../renders")
    parser.add_argument("--mos_file", type=str, default="mos_advanced.json")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--save_path", type=str, default="visualization_results.png")
    parser.add_argument("--use_subscores", action="store_true")
    return parser.parse_args()

def unnormalize(tensor):
    """Restore tensor to 0-1 range for visualization."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

def visualize(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Transform
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Resolve root_dir
    root_path = Path(args.root_dir)
    if not root_path.exists():
        # Try checking if it's relative to project root (K:\NVS-SQA)
        # If we are in K:\NVS-SQA, and args.root_dir is "../renders", that fails.
        # If we are in K:\NVS-SQA, it should be "renders".
        # Let's try to find "renders" in common locations.
        candidates = [
            Path("renders"),
            Path("../renders"),
            Path("K:/NVS-SQA/renders")
        ]
        for cand in candidates:
            if cand.exists():
                print(f"Found renders at: {cand}")
                root_path = cand
                break
    
    # Dataset (Use Test set or Val set)
    dataset = AdvancedOFNeRFDataset(str(root_path), args.mos_file, mode='test', transform=transform, distortion_sampling=False, use_subscores=args.use_subscores)
    # If test set is empty/small, fallback to val
    if len(dataset) == 0:
        print("Test set empty, using validation set.")
        dataset = AdvancedOFNeRFDataset(args.root_dir, args.mos_file, mode='val', transform=transform, distortion_sampling=False, use_subscores=args.use_subscores)
        
    loader = DataLoader(dataset, batch_size=1, shuffle=True) # Shuffle to get random samples
    
    # Model
    model = DisNeRFQA_Advanced(num_subscores=4).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    # Plotting
    cols = 1
    rows = args.num_samples
    fig, axes = plt.subplots(rows, cols, figsize=(8, 4 * rows))
    if rows == 1: axes = [axes]
    
    print(f"Generating {args.num_samples} visualizations...")
    
    with torch.no_grad():
        for i, (x_c, x_d, score, sub_scores_gt) in enumerate(loader):
            if i >= args.num_samples:
                break
                
            x_c, x_d = x_c.to(device), x_d.to(device)
            
            # Forward
            pred_score, pred_subs, _, _, _, _ = model(x_c, x_d)
            
            pred_val = pred_score.item()
            gt_val = score.item()
            
            # Get first frame for visualization
            # x_c is [B, T, C, H, W] -> [1, 8, 3, 224, 224]
            img_tensor = x_c[0, 0].cpu() # First frame
            img = unnormalize(img_tensor).permute(1, 2, 0).numpy()
            img = np.clip(img, 0, 1)
            
            ax = axes[i]
            ax.imshow(img)
            
            title = f"GT MOS: {gt_val:.2f} | Pred: {pred_val:.2f}"
            if args.use_subscores:
                # Sub-scores: Discomfort, Blur, Lighting, Artifacts
                sub_names = ["Disc", "Blur", "Light", "Artif"]
                pred_s = pred_subs[0].cpu().numpy() * 5.0 # Scale back to 1-5
                gt_s = sub_scores_gt[0].numpy() * 5.0
                
                sub_text = " | ".join([f"{n}:{p:.1f}({g:.1f})" for n, p, g in zip(sub_names, pred_s, gt_s)])
                title += f"\n{sub_text}"
                
            ax.set_title(title, fontsize=10)
            ax.axis('off')
            
    plt.tight_layout()
    plt.savefig(args.save_path, dpi=150)
    print(f"Saved visualization to {args.save_path}")

if __name__ == "__main__":
    args = parse_args()
    visualize(args)
