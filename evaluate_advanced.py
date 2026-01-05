import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr, pearsonr
import json
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
import os

from datasets.of_nerf import OFNeRFDataset
from models.dis_nerf_advanced import DisNeRFQA_Advanced
from train_advanced import AdvancedOFNeRFDataset, MultiScaleCrop

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Dis-NeRF-VQA Advanced (THREE)")
    parser.add_argument("--root_dir", type=str, default="../renders")
    parser.add_argument("--mos_file", type=str, default="mos_advanced.json")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best_model.pth")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--use_subscores", action="store_true", help="Evaluate sub-score predictions")
    parser.add_argument("--save_results", type=str, default=None, help="Path to save results. If None, auto-generated in eval_results/")
    parser.add_argument("--no_fusion", action="store_true", help="Disable Adaptive Fusion (must match training)")
    parser.add_argument("--note", type=str, default="", help="Optional note to include in results")
    return parser.parse_args()

def evaluate(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Transform (Standard Resize for Evaluation, no random crop)
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Dataset (Test set)
    test_set = AdvancedOFNeRFDataset(args.root_dir, args.mos_file, mode='test', transform=transform, distortion_sampling=False, use_subscores=args.use_subscores)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    print(f"Test set size: {len(test_set)}")
    
    # Model
    model = DisNeRFQA_Advanced(num_subscores=4, use_fusion=not args.no_fusion).to(device)
    
    # Load Checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Handle state dict (if saved as full checkpoint vs state_dict)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    
    preds = []
    targets = []
    
    sub_preds = []
    sub_targets = []
    
    keys = []
    
    with torch.no_grad():
        for i, (x_c, x_d, score, sub_scores_gt) in enumerate(tqdm(test_loader, desc="Evaluating")):
            x_c, x_d = x_c.to(device), x_d.to(device)
            
            # Forward
            pred_score, pred_subs, _, _, _, _ = model(x_c, x_d)
            
            preds.extend(pred_score.cpu().numpy().flatten())
            targets.extend(score.numpy().flatten())
            
            if args.use_subscores:
                sub_preds.append(pred_subs.cpu().numpy())
                sub_targets.append(sub_scores_gt.numpy())
                
            # Get keys for results file
            start_idx = i * args.batch_size
            end_idx = start_idx + x_c.size(0)
            batch_indices = range(start_idx, end_idx)
            for idx in batch_indices:
                if idx < len(test_set):
                    folder_path = test_set.valid_samples[idx]
                    keys.append(test_set._get_key_from_path(folder_path))

    # Calculate Metrics
    # Scale back to 0-100 for reporting
    preds_100 = np.array(preds) * 100.0
    targets_100 = np.array(targets) * 100.0
    
    srcc = spearmanr(preds_100, targets_100)[0]
    plcc = pearsonr(preds_100, targets_100)[0]
    rmse = np.sqrt(np.mean((preds_100 - targets_100)**2))
    
    print("\n" + "="*30)
    print("EVALUATION RESULTS (Advanced)")
    print("="*30)
    print(f"SRCC: {srcc:.4f}")
    print(f"PLCC: {plcc:.4f}")
    print(f"RMSE: {rmse:.4f} (Scale 0-100)")
    
    # Prepare Results Directory
    if args.save_results is None:
        from datetime import datetime
        os.makedirs("K:/NVS-SQA/TWO/eval_results", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        chk_name = Path(args.checkpoint).stem
        args.save_results = f"K:/NVS-SQA/TWO/eval_results/eval_{chk_name}_{timestamp}.json"

    results = {
        "metadata": {
            "checkpoint": args.checkpoint,
            "use_subscores": args.use_subscores,
            "no_fusion": args.no_fusion,
            "note": args.note,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "metrics": {
            "srcc": float(srcc),
            "plcc": float(plcc),
            "rmse": float(rmse)
        },
        "predictions": {}
    }
    
    # Save detailed results
    # Helper to convert numpy/torch types to python types
    def to_python(obj):
        if isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
            return float(obj)
        if isinstance(obj, list):
            return [to_python(x) for x in obj]
        return obj

    for i, key in enumerate(keys):
        res_entry = {
            "pred": float(preds_100[i]),
            "target": float(targets_100[i])
        }
        if args.use_subscores:
            res_entry["sub_pred"] = [float(x) for x in sub_preds[i]]
            res_entry["sub_target"] = [float(x) for x in sub_targets[i]]
            
        results["predictions"][key] = res_entry
        
    if "sub_score_srcc" in results:
        results["metrics"]["sub_score_srcc"] = {k: float(v) for k, v in results.pop("sub_score_srcc").items()}
        
    with open(args.save_results, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {args.save_results}")

if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
