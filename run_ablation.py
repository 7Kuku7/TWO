import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 定义基础配置 (请修改为你的实际路径和参数)
base_cmd = "python train_final1.py --root_dir 'renders' --mos_file 'mos_advanced.json' --batch_size 4 --gpu 0 --epochs 50 --num_repeats 3"

# 必须包含 --use_subscores 才能在完整版中激活多任务，消融时会被 --no_multitask 覆盖
base_cmd += " --use_subscores" 

# 定义 4 组实验
experiments = [
    {
        "name": "Exp0_Full_Model",
        "flags": ""  # 默认全开
    },
    {
        "name": "Exp1_Wo_SSL",
        "flags": "--no_ssl" # 去掉自监督
    },
    {
        "name": "Exp2_Wo_Decoupling",
        "flags": "--no_decouple" # 去掉解耦 Loss
    },
    {
        "name": "Exp3_Wo_Multitask",
        "flags": "--no_multitask" # 去掉多任务辅助 Loss
    }
]

print("🚀 开始运行消融实验...")

for exp in experiments:
    exp_name = exp["name"]
    flags = exp["flags"]
    
    print(f"\n[Running] {exp_name} ...")
    
    # 组合最终命令
    cmd = f"{base_cmd} --experiment_name {exp_name} {flags}"
    
    print(f"Command: {cmd}")
    
    # 执行命令
    exit_code = os.system(cmd)
    
    if exit_code != 0:
        print(f"❌ {exp_name} 运行出错！退出码: {exit_code}")
        break
    else:
        print(f"✅ {exp_name} 完成。")

print("\n🎉 所有消融实验结束！结果保存在 eval_results_repeated/ 目录下。")