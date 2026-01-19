import os
import itertools

# 基础命令 (Full Model)
base_cmd = "python train_final1.py --root_dir 'renders' --mos_file 'mos_advanced.json' --batch_size 4 --gpu 0 --epochs 50 --use_subscores"

# === 科学的超参数网格 ===
param_grid = {
    # 1. 排序损失：SRCC 的核心。试试比 0.1 大的，也试试小的。
    "lambda_rank": [0.05, 0.1, 0.2, 0.5],
    
    # 2. 解耦损失：之前 0.1 导致失败。重点搜小数值。
    "lambda_mi": [0.001, 0.01, 0.05],
    
    # 3. 多任务损失：之前 0.5 导致失败。重点搜小数值。
    "lambda_sub": [0.2, 0.1, 0.05]
}

# 注意：为了节省时间，我暂时把 SSL 固定为 0.2。
# 如果这三个参数定下来了，你可以单独再搜一下 SSL。
# 目前的组合数：4 * 3 * 3 = 36 组。可以接受。

keys = param_grid.keys()
combinations = list(itertools.product(*param_grid.values()))

print(f"🚀 开始最终版全参数搜索，共 {len(combinations)} 组实验...")

for i, values in enumerate(combinations):
    params = dict(zip(keys, values))
    
    # 实验命名：Rank_0.2_MI_0.01_Sub_0.1
    exp_name = f"Search_Rank_{params['lambda_rank']}_MI_{params['lambda_mi']}_Sub_{params['lambda_sub']}"
    
    # 构造命令
    flags = f"--experiment_name {exp_name} --lambda_rank {params['lambda_rank']} --lambda_mi {params['lambda_mi']} --lambda_sub {params['lambda_sub']} --lambda_ssl 0.2"
    
    # 先跑 1 次重复快速验证
    cmd = f"{base_cmd} {flags} --num_repeats 1"
    
    print(f"\n[{i+1}/{len(combinations)}] 正在运行: {exp_name}")
    # print(f"Command: {cmd}")
    
    # === 关键修正：这里去掉了注释，现在会真正执行了 ===
    exit_code = os.system(cmd)
    
    if exit_code != 0:
        print(f"❌ {exp_name} 运行出错！退出码: {exit_code}")

print("\n🎉 所有搜参实验结束！请去 eval_results_repeated 文件夹查看各组的 best.json。")
