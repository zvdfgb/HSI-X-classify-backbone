import os
import numpy as np
from scipy.io import savemat
from pathlib import Path

def create_mock_houston2013(target_dir):
    os.makedirs(target_dir, exist_ok=True)
    H, W = 30, 30
    C_hsi = 144
    num_classes = 15
    
    # 1. HSI data (H, W, C)
    hsi = np.random.randn(H, W, C_hsi).astype(np.float32)
    savemat(os.path.join(target_dir, "houston_hsi.mat"), {"houston_hsi": hsi})
    
    # 2. LiDAR data (H, W)
    lidar = np.random.randn(H, W).astype(np.float32)
    savemat(os.path.join(target_dir, "houston_lidar.mat"), {"houston_lidar": lidar})
    
    # 3. Ground truth (H, W), values 1..15
    gt = np.random.randint(1, num_classes + 1, size=(H, W)).astype(np.int32)
    # Leave some unlabeled
    gt[0:5, 0:5] = 0
    savemat(os.path.join(target_dir, "houston_gt.mat"), {"houston_gt": gt})
    
    # 4. Index
    labeled_pos = np.argwhere(gt > 0)
    np.random.shuffle(labeled_pos)
    n_train = len(labeled_pos) // 2
    train_idx = labeled_pos[:n_train]
    test_idx = labeled_pos[n_train:]
    
    all_pos = np.indices((H, W)).reshape(2, -1).T
    
    savemat(os.path.join(target_dir, "houston_index.mat"), {
        "houston_train": train_idx,
        "houston_test": test_idx,
        "houston_all": all_pos
    })
    print(f"Mock Houston2013 created at {target_dir}")

def create_mock_augsburg(target_dir):
    os.makedirs(target_dir, exist_ok=True)
    H, W = 30, 30
    C_hsi = 180
    C_sar = 4
    num_classes = 7
    
    hsi = np.random.randn(H, W, C_hsi).astype(np.float32)
    savemat(os.path.join(target_dir, "augsburg_hsi.mat"), {"augsburg_hsi": hsi})
    
    sar = np.random.randn(H, W, C_sar).astype(np.float32)
    savemat(os.path.join(target_dir, "augsburg_sar.mat"), {"augsburg_sar": sar})
    
    gt = np.random.randint(1, num_classes + 1, size=(H, W)).astype(np.int32)
    gt[0:5, 0:5] = 0
    savemat(os.path.join(target_dir, "augsburg_gt.mat"), {"augsburg_gt": gt})
    
    labeled_pos = np.argwhere(gt > 0)
    np.random.shuffle(labeled_pos)
    n_train = len(labeled_pos) // 2
    train_idx = labeled_pos[:n_train]
    test_idx = labeled_pos[n_train:]
    all_pos = np.indices((H, W)).reshape(2, -1).T
    
    savemat(os.path.join(target_dir, "augsburg_index.mat"), {
        "augsburg_train": train_idx,
        "augsburg_test": test_idx,
        "augsburg_all": all_pos
    })
    print(f"Mock Augsburg created at {target_dir}")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[1]
    mock_dir_houston = project_root / "data" / "Houston2013"
    create_mock_houston2013(str(mock_dir_houston))
    mock_dir_augsburg = project_root / "data" / "Augsburg"
    create_mock_augsburg(str(mock_dir_augsburg))
