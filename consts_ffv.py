"""
FFV数据集常量配置
用于按 llff/fieldwork/lab 三个子集分别评估
"""
from pathlib import Path
import json

# ==========================================
# 数据集基本信息
# ==========================================
FFV_ROOT = "/media/abc/One Touch/NVS-SQA/benchmark_bank"
FFV_LABELS_FILE = "/media/abc/One Touch/NVS-SQA/benchmark_bank/flat_labels_offset_by_ref_scenewise_split_11.json"

# 三个子集
SUBSETS = ["llff", "fieldwork", "lab"]

# 各子集的场景
SUBSET_SCENES = {
    "llff": ["fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex"],
    "fieldwork": ["Bears", "Dinosaur", "Elephant", "Giraffe", "Leopards", 
                  "Naiad_statue", "Puccini_statue", "Vespa", "Whale"],
    "lab": ["CD_occlusion_extr", "CD_occlusion_intr", "Glass", 
            "Glossy_animals_extr", "Glossy_animals_intr", "Metal", "Toys"],
}

# 所有方法
METHODS = [
    "directvoxgo", "gnt_crossscene", "gnt_singlescene", 
    "ibrnet_finetune", "ibrnet_pretrain", "light_field", 
    "mipnerf", "nerf", "nex", "plenoxel"
]

# 方法可读名称映射
METHOD_NAMES = {
    "directvoxgo": "DVGO",
    "gnt_crossscene": "GNT-C",
    "gnt_singlescene": "GNT-S",
    "ibrnet_finetune": "IBRNet-S",
    "ibrnet_pretrain": "IBRNet-C",
    "light_field": "LFNR",
    "mipnerf": "MipNeRF",
    "nerf": "NeRF",
    "nex": "NeX",
    "plenoxel": "Plenoxel",
}

# ==========================================
# 评估指标
# ==========================================
eva_metrics = {
    "SRCC": float("-inf"),
    "PLCC": float("-inf"),
    "KRCC": float("-inf"),
    "RMSE": float("inf"),
}

tr_metrics = {
    "mse": float("inf"),
    "mae": float("inf"),
    "loss": float("inf"),
    "SRCC": float("-inf"),
    "PLCC": float("-inf"),
    "KRCC": float("-inf"),
}

# ==========================================
# 辅助函数
# ==========================================
def load_ffv_labels(labels_file=FFV_LABELS_FILE):
    """加载FFV标签文件"""
    with open(labels_file, 'r') as f:
        data = json.load(f)
    return data

def get_train_test_keys(labels_file=FFV_LABELS_FILE):
    """获取训练/测试集的key列表"""
    data = load_ffv_labels(labels_file)
    return data['tr_keys'], data['tt_keys']

def filter_keys_by_subset(keys, subset):
    """按子集过滤keys"""
    if subset == 'all':
        return keys
    return [k for k in keys if k.startswith(subset + '+')]

def get_subset_from_key(key):
    """从key中提取子集名称"""
    return key.split('+')[0]

def get_scene_from_key(key):
    """从key中提取场景名称"""
    parts = key.split('+')
    return parts[1] if len(parts) >= 2 else None

def get_method_from_key(key):
    """从key中提取方法名称"""
    parts = key.split('+')
    return parts[2] if len(parts) >= 3 else None

def build_sample_path(root_dir, key):
    """根据key构建样本路径"""
    parts = key.split('+')
    if len(parts) != 3:
        return None
    subset, scene, method = parts
    return Path(root_dir) / subset / scene / method / 'frames'

def validate_sample(root_dir, key):
    """验证样本是否存在"""
    path = build_sample_path(root_dir, key)
    if path is None:
        return False
    return path.exists()

# ==========================================
# JOD分数处理
# ==========================================
def get_jod_range(labels_file=FFV_LABELS_FILE):
    """获取JOD分数范围"""
    data = load_ffv_labels(labels_file)
    scores = list(data['labels'].values())
    return min(scores), max(scores)

def normalize_jod(jod_score, jod_min=None, jod_max=None, labels_file=FFV_LABELS_FILE):
    """
    将JOD分数归一化到[0, 1]
    JOD越高质量越好
    """
    if jod_min is None or jod_max is None:
        jod_min, jod_max = get_jod_range(labels_file)
    
    normalized = (jod_score - jod_min) / (jod_max - jod_min + 1e-8)
    return max(0, min(1, normalized))

def denormalize_jod(normalized_score, jod_min=None, jod_max=None, labels_file=FFV_LABELS_FILE):
    """将归一化分数还原为JOD分数"""
    if jod_min is None or jod_max is None:
        jod_min, jod_max = get_jod_range(labels_file)
    
    return normalized_score * (jod_max - jod_min) + jod_min

# ==========================================
# 数据集统计
# ==========================================
def print_dataset_stats(labels_file=FFV_LABELS_FILE):
    """打印数据集统计信息"""
    data = load_ffv_labels(labels_file)
    tr_keys = data['tr_keys']
    tt_keys = data['tt_keys']
    labels = data['labels']
    
    print("=" * 60)
    print("FFV Dataset Statistics")
    print("=" * 60)
    print(f"Total samples: {len(labels)}")
    print(f"Training samples: {len(tr_keys)}")
    print(f"Test samples: {len(tt_keys)}")
    print()
    
    jod_min, jod_max = get_jod_range(labels_file)
    print(f"JOD score range: [{jod_min:.4f}, {jod_max:.4f}]")
    print()
    
    for subset in SUBSETS:
        tr_subset = filter_keys_by_subset(tr_keys, subset)
        tt_subset = filter_keys_by_subset(tt_keys, subset)
        print(f"{subset.upper()}:")
        print(f"  Scenes: {len(SUBSET_SCENES[subset])}")
        print(f"  Train: {len(tr_subset)}, Test: {len(tt_subset)}")
    
    print("=" * 60)


if __name__ == "__main__":
    print_dataset_stats()
