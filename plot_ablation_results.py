import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse
import glob

def parse_args():
    parser = argparse.ArgumentParser(description="Plot Ablation Study Results")
    parser.add_argument("--results_dir", type=str, default="K:/NVS-SQA/TWO/eval_results", help="Directory containing result JSONs")
    parser.add_argument("--save_path", type=str, default="ablation_comparison.png")
    return parser.parse_args()

def load_results(results_dir):
    results = []
    files = glob.glob(f"{results_dir}/*.json")
    
    for f in files:
        try:
            with open(f, 'r') as json_file:
                data = json.load(json_file)
                
            # Extract info
            meta = data.get("metadata", {})
            metrics = data.get("metrics", {})
            
            # Determine name
            note = meta.get("note", "").lower()
            if "no fusion" in note:
                name = "w/o Fusion"
            elif "multi-task" in note or "multitask" in note:
                name = "w/o Multi-task"
            elif "multi-scale" in note or "multiscale" in note:
                name = "w/o Multi-scale"
            elif "full" in note:
                name = "Full Model"
            elif not meta.get("no_fusion", False): # Default to Full if not specified and has fusion
                name = "Full Model"
            else:
                name = Path(f).stem
                
            results.append({
                "name": name,
                "srcc": metrics.get("srcc", 0),
                "plcc": metrics.get("plcc", 0),
                "rmse": metrics.get("rmse", 0)
            })
        except Exception as e:
            print(f"Skipping {f}: {e}")
            
    return results

def plot_results(results, save_path):
    if not results:
        print("No results found to plot.")
        return

    names = [r['name'] for r in results]
    srcc = [r['srcc'] for r in results]
    plcc = [r['plcc'] for r in results]
    rmse = [r['rmse'] for r in results]
    
    x = np.arange(len(names))
    width = 0.25
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # Plot SRCC and PLCC on left axis
    rects1 = ax1.bar(x - width/2, srcc, width, label='SRCC', color='skyblue')
    rects2 = ax1.bar(x + width/2, plcc, width, label='PLCC', color='lightgreen')
    
    ax1.set_ylabel('Correlation (Higher is better)')
    ax1.set_ylim(0, 1.0)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=15)
    ax1.legend(loc='upper left')
    
    # Plot RMSE on right axis
    ax2 = ax1.twinx()
    rects3 = ax2.plot(x, rmse, 'r-o', label='RMSE', linewidth=2)
    ax2.set_ylabel('RMSE (Lower is better)')
    # ax2.set_ylim(0, 20) # Adjust based on scale
    ax2.legend(loc='upper right')
    
    plt.title('Ablation Study Results')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    args = parse_args()
    results = load_results(args.results_dir)
    # Sort results to keep Full Model first if possible, or just sort by name
    results.sort(key=lambda x: x['name'])
    
    print("Loaded Results:")
    for r in results:
        print(f"{r['name']}: SRCC={r['srcc']:.4f}, PLCC={r['plcc']:.4f}, RMSE={r['rmse']:.4f}")
        
    plot_results(results, args.save_path)
