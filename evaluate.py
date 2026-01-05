import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import json
import os
from datetime import datetime

from datasets.of_nerf import OFNeRFDataset
from models.dis_nerf import DisNeRFQA
from utils import calculate_srcc, calculate_plcc, calculate_krcc

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Dis-NeRF-VQA")
    parser.add_argument("--root_dir", type=str, default="../renders", help="Path to renders directory")
    parser.add_argument("--mos_file", type=str, default="mos.json", help="Path to MOS JSON file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--output_dir", type=str, default="eval_results", help="Directory to save evaluation results")
    return parser.parse_args()

def evaluate(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Transforms
    import torchvision.transforms as T
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets - STRICTLY USE TEST MODE
    test_set = OFNeRFDataset(args.root_dir, args.mos_file, mode='test', transform=transform, distortion_sampling=False)
    
    if len(test_set) == 0:
        print("Error: Test set is empty. Please check your dataset split or root directory.")
        return

    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # Model
    model = DisNeRFQA().to(device)
    
    # Load Checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint not found at {args.checkpoint}")
        return
        
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Handle both full checkpoint dict and state_dict only
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        epoch = checkpoint.get('epoch', 'Unknown')
        best_srcc = checkpoint.get('best_srcc', 'Unknown')
        print(f"Loaded model from epoch {epoch} (Best SRCC: {best_srcc})")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded model state dict directly.")
        
    model.eval()
    
    preds = []
    targets = []
    video_info = []
    
    print("Starting evaluation on TEST set...")
    with torch.no_grad():
        batch_idx = 0
        for i, (x_c, x_d, score) in enumerate(tqdm(test_loader)):
            x_c, x_d, score = x_c.to(device), x_d.to(device), score.to(device)
            
            # Forward
            pred_score, _, _ = model(x_c, x_d)
            
            # Convert to numpy and denormalize (from [0,1] back to [0,100])
            pred_np = pred_score.cpu().numpy().flatten() * 100.0
            target_np = score.cpu().numpy().flatten() * 100.0
            
            # Collect per-video info
            for j in range(len(pred_np)):
                sample_idx = batch_idx * args.batch_size + j
                if sample_idx < len(test_set.valid_samples):
                    video_path = test_set.valid_samples[sample_idx]
                    video_name = video_path.name
                    
                    video_info.append({
                        "video_name": video_name,
                        "predicted_mos": float(pred_np[j]),
                        "ground_truth_mos": float(target_np[j])
                    })
            
            preds.extend(pred_np)
            targets.extend(target_np)
            batch_idx += 1
            
    # Calculate Metrics
    srcc = calculate_srcc(preds, targets)
    plcc = calculate_plcc(preds, targets)
    krcc = calculate_krcc(preds, targets)
    rmse = np.sqrt(np.mean((np.array(preds) - np.array(targets))**2))
    
    print("\n" + "="*30)
    print(f"TEST RESULTS")
    print("="*30)
    print(f"SRCC: {srcc:.4f}")
    print(f"PLCC: {plcc:.4f}")
    print(f"KRCC: {krcc:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print("="*30)
    
    # Get test video names
    test_video_list = [str(p.name) for p in test_set.valid_samples]
    
    # Save detailed results with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"eval_{timestamp}.json")
    
    results = {
        "evaluation_info": {
            "timestamp": timestamp,
            "checkpoint": args.checkpoint,
            "num_test_videos": len(test_video_list)
        },
        "metrics": {
            "SRCC": float(srcc),
            "PLCC": float(plcc),
            "KRCC": float(krcc),
            "RMSE": float(rmse)
        },
        "test_videos": test_video_list,
        "per_video_results": video_info
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\nDetailed results saved to {output_file}")
    print(f"Evaluated {len(test_video_list)} test videos")

if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
