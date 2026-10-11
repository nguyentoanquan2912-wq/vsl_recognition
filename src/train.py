import os
import glob
import json
import joblib
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from collections import Counter
from config import Config
from src.preprocessor import extract_features, augment_landmarks, normalize_landmarks, flatten_landmarks
from src.models import get_model
from src.evaluate import evaluate_model, cross_validate_model

def train_model(model_type="svm"):
    """
    Train ML model. Handles small datasets with data augmentation.
    """
    print(f"[TRAIN] Starting training: {model_type}")

    # 1. Load RAW data
    print("[TRAIN] Loading raw data...")
    X_raw = []
    y_raw = []
    label_names = []

    if os.path.exists(Config.LANDMARKS_DIR):
        label_names = sorted([d for d in os.listdir(Config.LANDMARKS_DIR)
                              if os.path.isdir(os.path.join(Config.LANDMARKS_DIR, d))])

    for label_idx, label_name in enumerate(label_names):
        class_dir = os.path.join(Config.LANDMARKS_DIR, label_name)
        npy_files = glob.glob(os.path.join(class_dir, "*.npy"))
        for npy_file in npy_files:
            try:
                landmarks = np.load(npy_file)
                X_raw.append(landmarks)
                y_raw.append(label_idx)
            except Exception as e:
                print(f"[WARN] Error loading {npy_file}: {e}")

    if len(X_raw) == 0:
        raise ValueError("Du lieu trong. Vui long thu thap du lieu truoc.")

    print(f"[TRAIN] Loaded {len(X_raw)} raw samples for {len(label_names)} labels.")

    # 2. Data augmentation
    class_counts = Counter(y_raw)
    max_class_count = max(class_counts.values()) if class_counts else 0
    target_count = max(50, max_class_count)

    augmented_X_raw = []
    augmented_y = []

    for cls in range(len(label_names)):
        cls_count = class_counts.get(cls, 0)
        if cls_count == 0:
            continue
        
        cls_indices = [i for i, label in enumerate(y_raw) if label == cls]
        # Keep original
        for idx in cls_indices:
            augmented_X_raw.append(X_raw[idx])
            augmented_y.append(cls)
            
        needed = target_count - cls_count
        if needed > 0:
            # We want to add 'needed' samples by augmenting existing ones.
            # "Apply 3-5x augmentation per original sample" -> we can just randomly pick existing ones
            # and generate augmentations until we reach target.
            # To ensure we apply 3-5x per original if needed, we iterate over original samples.
            per_sample = max(1, needed // cls_count)
            # Ensure it's at least 3-5 if possible, but the requirement specifically says:
            # "Use a target of max(50, max_class_count) to balance classes."
            # "Apply 3-5x augmentation per original sample."
            # So let's just generate `per_sample` augmentations per original sample.
            # If `per_sample` < 3, we still reach the target. Let's make sure we generate enough.
            added = 0
            while added < needed:
                for idx in cls_indices:
                    if added >= needed:
                        break
                    # Generate a random number of augs per sample, e.g., 1 (since we loop)
                    aug_samples = augment_landmarks(X_raw[idx], n=1)
                    augmented_X_raw.extend(aug_samples)
                    augmented_y.extend([cls] * len(aug_samples))
                    added += len(aug_samples)

    print(f"[TRAIN] After augmentation: {len(augmented_X_raw)} total raw samples.")

    # 3. Extract features
    print(f"[TRAIN] Extracting features (Hybrid Mode: {getattr(Config, 'USE_HYBRID_FEATURES', False)})...")
    X = []
    old_dim = Config.LANDMARKS_PER_HAND * Config.NUM_COORDS

    for landmarks in augmented_X_raw:
        if getattr(Config, 'USE_HYBRID_FEATURES', False):
            feat = extract_features(landmarks)
        else:
            norm_lm = normalize_landmarks(landmarks)
            feat = flatten_landmarks(norm_lm)
            if len(feat) == old_dim and Config.FEATURE_DIM == old_dim * 2:
                feat = np.concatenate([feat, np.zeros(old_dim)])
        X.append(feat)

    X = np.array(X)
    y = np.array(augmented_y)

    # 4. Filter valid classes
    class_counts = Counter(y)
    valid_classes = [cls for cls, count in class_counts.items() if count >= 4]

    if len(valid_classes) < 2:
        raise ValueError(f"Can it nhat 2 lop du lieu. Hien tai chi co {len(valid_classes)} lop hop le.")

    valid_indices = np.isin(y, valid_classes)
    X = X[valid_indices]
    y = y[valid_indices]

    print(f"[TRAIN] Using {len(valid_classes)} classes, {len(X)} samples after filtering.")

    # 5. Train/test split
    print("[TRAIN] Splitting data...")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE, stratify=y
        )
    except ValueError as e:
        warnings.warn(f"Stratified split failed: {e}. Falling back to random split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE
        )

    # 6. Train model
    print(f"[TRAIN] Training {model_type.upper()}...")
    model = get_model(model_type)
    model.fit(X_train, y_train)

    # 7. Evaluate
    print("[TRAIN] Evaluating model...")
    eval_label_names = label_names
    metrics = evaluate_model(model, X_test, y_test, eval_label_names)
    print(f"[TRAIN] Accuracy: {metrics.get('accuracy', 0):.4f}")

    # 8. Cross validation
    min_class_count = min(Counter(y).values())
    cv_folds = min(Config.CV_FOLDS, min_class_count, len(X) // 2)
    cv_folds = max(2, cv_folds)

    print(f"[TRAIN] Cross-validation ({cv_folds}-fold)...")
    cv_scores = cross_validate_model(model, X, y, cv=cv_folds)
    metrics['cv_scores'] = cv_scores
    if cv_scores:
        print(f"[TRAIN] CV Mean: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores)*2:.4f})")

    # 9. Save model and metadata
    os.makedirs(Config.MODELS_DIR, exist_ok=True)
    model_path = os.path.join(Config.MODELS_DIR, f"model_{model_type}.pkl")
    labels_path = os.path.join(Config.MODELS_DIR, "label_names.pkl")
    meta_path = os.path.join(Config.MODELS_DIR, f"model_{model_type}_meta.json")

    print(f"[TRAIN] Saving model to {model_path}...")
    joblib.dump(model, model_path)
    joblib.dump(label_names, labels_path)

    use_hybrid = getattr(Config, 'USE_HYBRID_FEATURES', False)
    meta_data = {
        "model_type": model_type,
        "use_hybrid_features": use_hybrid,
        "feature_dim": Config.HYBRID_FEATURE_DIM if use_hybrid else Config.FEATURE_DIM
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta_data, f, indent=2)

    # 10. Save metrics
    metrics["model_type"] = model_type
    metrics["labels"] = label_names
    metrics["total_samples"] = int(len(X))
    metrics["augmented"] = True
    metrics["use_hybrid_features"] = use_hybrid
    metrics_path = os.path.join(Config.MODELS_DIR, "metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"[TRAIN] Done! Metrics saved to {metrics_path}")
    return metrics

if __name__ == "__main__":
    train_model("svm")
