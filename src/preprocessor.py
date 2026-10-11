import os
import json
import numpy as np
import glob
from config import Config

def normalize_landmarks(landmarks):
    """Normalize landmarks relative to wrist (point 0), scale to unit range.
    Handles both single hand (21 points) and two hands (42 points)."""
    if landmarks is None or len(landmarks) == 0:
        return landmarks

    landmarks = np.array(landmarks, copy=True, dtype=np.float64)

    if len(landmarks) == Config.LANDMARKS_PER_HAND * 2:
        # Two hands: normalize each hand separately
        hand1 = landmarks[:Config.LANDMARKS_PER_HAND]
        hand2 = landmarks[Config.LANDMARKS_PER_HAND:]

        hand1 = _normalize_single_hand(hand1)
        hand2 = _normalize_single_hand(hand2)

        return np.vstack([hand1, hand2])
    else:
        return _normalize_single_hand(landmarks)


def _normalize_single_hand(landmarks):
    """Normalize a single hand's landmarks relative to wrist."""
    landmarks = np.array(landmarks, copy=True, dtype=np.float64)
    wrist = landmarks[0].copy()
    landmarks = landmarks - wrist
    max_dist = np.max(np.abs(landmarks))
    if max_dist > 0:
        landmarks = landmarks / max_dist
    return landmarks


def flatten_landmarks(landmarks):
    """Convert landmark matrix to 1D vector."""
    return np.array(landmarks).flatten()


def _extract_single_hand_features(landmarks):
    """Extract hybrid features for a single hand (88 features)."""
    landmarks = np.array(landmarks, copy=True, dtype=np.float64)
    
    # 1. Normalized coordinates (63)
    norm_coords = _normalize_single_hand(landmarks).flatten()
    
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    palm_size = np.linalg.norm(middle_mcp - wrist)
    if palm_size == 0:
        palm_size = 1.0
        
    # 2. Finger joint angles (10)
    angles = []
    finger_bases = [1, 5, 9, 13, 17]
    for base_idx in finger_bases:
        a, b, c, d = landmarks[base_idx], landmarks[base_idx+1], landmarks[base_idx+2], landmarks[base_idx+3]
        v1 = b - a
        v2 = c - b
        v3 = d - c
        
        def angle_between(u, v):
            u_norm = np.linalg.norm(u)
            v_norm = np.linalg.norm(v)
            if u_norm == 0 or v_norm == 0:
                return 0.0
            u_u = u / u_norm
            v_u = v / v_norm
            return np.arccos(np.clip(np.dot(u_u, v_u), -1.0, 1.0))
            
        angles.append(angle_between(v1, v2))
        angles.append(angle_between(v2, v3))
        
    angles = np.array(angles, dtype=np.float64)
    
    # 3. Fingertip-to-wrist distances (5)
    fingertips = [4, 8, 12, 16, 20]
    tip_wrist_dist = np.array([np.linalg.norm(landmarks[i] - wrist) / palm_size for i in fingertips])
    
    # 4. Inter-fingertip distances (10)
    inter_tip_dist = []
    for i in range(len(fingertips)):
        for j in range(i+1, len(fingertips)):
            dist = np.linalg.norm(landmarks[fingertips[i]] - landmarks[fingertips[j]]) / palm_size
            inter_tip_dist.append(dist)
    inter_tip_dist = np.array(inter_tip_dist, dtype=np.float64)
    
    return np.concatenate([norm_coords, angles, tip_wrist_dist, inter_tip_dist])


def extract_features(landmarks):
    """
    Extract hybrid features from landmarks (176 features total).
    Handles both single hand and two hands.
    """
    if landmarks is None or len(landmarks) == 0:
        return landmarks
        
    landmarks = np.array(landmarks, copy=True, dtype=np.float64)
    
    if len(landmarks) == Config.LANDMARKS_PER_HAND * 2:
        hand1 = landmarks[:Config.LANDMARKS_PER_HAND]
        hand2 = landmarks[Config.LANDMARKS_PER_HAND:]
        feat1 = _extract_single_hand_features(hand1)
        feat2 = _extract_single_hand_features(hand2)
        return np.concatenate([feat1, feat2])
    else:
        feat1 = _extract_single_hand_features(landmarks)
        feat2 = np.zeros(88, dtype=np.float64)
        return np.concatenate([feat1, feat2])


def augment_landmarks(landmarks, n=5):
    """Data augmentation: rotation, mirroring, noise, scale, translation."""
    augmented = []
    for _ in range(n):
        aug_lm = np.array(landmarks, copy=True, dtype=np.float64)
        
        # Mirroring (50% chance)
        if np.random.rand() > 0.5:
            aug_lm[:, 0] = -aug_lm[:, 0]
            
        # Rotation
        angles = np.random.uniform(-15, 15, 3) * np.pi / 180.0
        rx = np.array([[1, 0, 0], [0, np.cos(angles[0]), -np.sin(angles[0])], [0, np.sin(angles[0]), np.cos(angles[0])]])
        ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])], [0, 1, 0], [-np.sin(angles[1]), 0, np.cos(angles[1])]])
        rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0], [np.sin(angles[2]), np.cos(angles[2]), 0], [0, 0, 1]])
        R = rx @ ry @ rz
        
        center = np.mean(aug_lm, axis=0)
        aug_lm = (aug_lm - center) @ R.T + center
        
        # Noise
        noise = np.random.normal(0, 0.01, aug_lm.shape)
        aug_lm += noise
        
        # Scale
        scale = np.random.uniform(0.9, 1.1)
        aug_lm *= scale
        
        # Translation
        translation = np.random.uniform(-0.05, 0.05, 3)
        aug_lm += translation
        
        augmented.append(aug_lm)
    return augmented


def load_dataset(use_hybrid_features=True):
    """Load all .npy files from LANDMARKS_DIR.
    Backward compatible: pads 1-hand data to 2-hand depending on feature format."""
    X = []
    y = []
    label_names = []

    if os.path.exists(Config.LANDMARKS_DIR):
        label_names = sorted([d for d in os.listdir(Config.LANDMARKS_DIR)
                              if os.path.isdir(os.path.join(Config.LANDMARKS_DIR, d))])

    old_dim = Config.LANDMARKS_PER_HAND * Config.NUM_COORDS  # 63
    
    # Determine target dimension based on config and mode
    if use_hybrid_features and hasattr(Config, 'HYBRID_FEATURE_DIM'):
        target_dim = Config.HYBRID_FEATURE_DIM
    else:
        target_dim = Config.FEATURE_DIM

    for label_idx, label_name in enumerate(label_names):
        class_dir = os.path.join(Config.LANDMARKS_DIR, label_name)
        npy_files = glob.glob(os.path.join(class_dir, "*.npy"))

        for npy_file in npy_files:
            try:
                landmarks = np.load(npy_file)
                
                if use_hybrid_features:
                    features = extract_features(landmarks)
                else:
                    norm_lm = normalize_landmarks(landmarks)
                    features = flatten_landmarks(norm_lm)
                    
                    # Backward compat: pad old 1-hand data (63) to 2-hand (126)
                    if len(features) == old_dim and target_dim == old_dim * 2:
                        features = np.concatenate([features, np.zeros(old_dim)])

                if len(features) == target_dim:
                    X.append(features)
                    y.append(label_idx)
            except Exception as e:
                print(f"[WARN] Error loading {npy_file}: {e}")

    return np.array(X), np.array(y), label_names
