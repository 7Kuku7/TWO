# ================= consts_simple_split.py (简单随机划分，不按场景分) =================
"""
简单划分策略：
- 不区分场景，将所有 scene__method__condition__path 组合视为独立样本
- 按 7:2:1 比例随机划分训练集、验证集、测试集（验证集占20%更合理）
- 适合快速实验和对比基线
"""
from pathlib import Path
import random

# 你的数据根
NERFSTUDIO_ROOT = "datasets/nerfstudio"

# -------------------------
# 所有场景（11个）
# -------------------------
all_scenes = [
    "LiZhaoJi_Building",
    "office1b",
    "riverview",
    "office_view1",
    "office_view2",
    "Sqaure2",
    "Center_Garden",
    "Sqaure1",
    "raf_furnishedroom",
    "Architecture",
    "apartment",
]

# -------------------------
# 简单划分：不按场景分，所有场景都参与训练
# 在训练时会从所有样本中随机划分
# -------------------------
# 训练时使用所有场景
nerfstudio_train_scenes = all_scenes.copy()

# 验证和测试场景设为空（表示使用样本级划分）
nerfstudio_val_scenes = []
nerfstudio_test_scenes = []

# 工作集（所有场景）
nerfstudio_scenes = all_scenes.copy()

# -------------------------
# 方法 / 条件 / 轨迹（与目录名一致）
# -------------------------
nerfstudio_methods = [
    "instant-ngp",
    "nerfacto",
    "mipnerf",
    "tensorf",
    "nerf",
]

# 让 train_utils.init_log 有可用的数据集名称
dataset_names = {
    "nerfstudio": "nerfstudio"
}

nerfstudio_conditions = [
    "baseline",
    "clip_0.4", "clip_0.7", "clip_1.5", "clip_2.5",
    "gamma_0.6", "gamma_1.6",
]

nerfstudio_trajectories = [
    "path1",
    "path2",
]

eva_metrics = {
    "SRCC": float("-inf"),
    "PLCC": float("-inf"),
    "KRCC": float("-inf"),
}

tr_metrics = {
    "mse": float("inf"), "mae": float("inf"), "loss": float("inf"),
    "SRCC": float("-inf"),
    "PLCC": float("-inf"),
    "KRCC": float("-inf"),
}

# -------------------------
# 样本级划分函数（7:2:1）
# -------------------------
def build_simple_split(renders_root="renders", train_ratio=0.7, val_ratio=0.2, seed=42):
    """
    构建简单的样本级划分
    
    Args:
        renders_root: 渲染数据根目录
        train_ratio: 训练集比例（默认0.7）
        val_ratio: 验证集比例（默认0.2）
        seed: 随机种子
    
    Returns:
        train_samples, val_samples, test_samples: 三个列表，每个元素是Path对象
    """
    random.seed(seed)
    renders_root = Path(renders_root)
    
    # 收集所有有效样本
    all_samples = []
    for d in sorted([p for p in renders_root.iterdir() if p.is_dir()]):
        parts = d.name.split("__")
        if len(parts) != 4:
            continue
        scene, method, cond, path_name = parts
        
        # 检查是否有足够的帧
        imgs = sorted(d.glob("frame_*.png"))
        if len(imgs) < 50:  # 至少50帧
            continue
        
        all_samples.append(d)
    
    # 随机打乱
    random.shuffle(all_samples)
    
    # 计算划分点
    n_total = len(all_samples)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # 划分
    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:n_train+n_val]
    test_samples = all_samples[n_train+n_val:]
    
    print(f"[简单划分] 总样本: {n_total}")
    print(f"  - 训练集: {len(train_samples)} ({len(train_samples)/n_total*100:.1f}%)")
    print(f"  - 验证集: {len(val_samples)} ({len(val_samples)/n_total*100:.1f}%)")
    print(f"  - 测试集: {len(test_samples)} ({len(test_samples)/n_total*100:.1f}%)")
    
    return train_samples, val_samples, test_samples


def build_simple_data_index(train_samples=None, val_samples=None, test_samples=None, mode='train'):
    """
    从样本列表构建数据索引（兼容训练脚本的格式）
    
    Args:
        train_samples, val_samples, test_samples: 样本列表
        mode: 'train', 'val', 'test' 或 'all'
    
    Returns:
        index = {scene: {method: [dir1, dir2, ...]}}
    """
    # 选择要使用的样本
    if mode == 'train':
        samples = train_samples
    elif mode == 'val':
        samples = val_samples
    elif mode == 'test':
        samples = test_samples
    elif mode == 'all':
        samples = (train_samples or []) + (val_samples or []) + (test_samples or [])
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    if samples is None:
        samples = []
    
    # 构建索引
    index = {}
    for d in samples:
        parts = d.name.split("__")
        if len(parts) != 4:
            continue
        scene, method, cond, path_name = parts
        
        index.setdefault(scene, {}).setdefault(method, []).append(d)
    
    return index


# -------------------------
# 可选：快速检查
# -------------------------
def sanity_check(root=NERFSTUDIO_ROOT):
    root = Path(root)
    ok, miss = 0, []
    for sc in all_scenes:
        for m in nerfstudio_methods:
            for c in nerfstudio_conditions:
                for p in nerfstudio_trajectories:
                    frames = root / sc / m / c / p / "frames"
                    if frames.is_dir():
                        ok += 1
                    else:
                        miss.append(str(frames))
    print(f"[consts_simple_split] Found {ok} existing <scene/method/cond/path>/frames folders.")
    if miss:
        print(f"[consts_simple_split] Missing {len(miss)} folders (OK: 不存在的组合会自动跳过)。示例缺失：")
        for s in miss[:8]:
            print("  -", s)
# ============================================================================
