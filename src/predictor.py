import os
import glob
import joblib
import json
import time
import numpy as np
from config import Config
from src.preprocessor import extract_features, normalize_landmarks, flatten_landmarks

class SignPredictor:
    """Load model and predict sign from landmarks. Supports 1 or 2 hands."""
    def __init__(self, model_type=None):
        self.model_type = model_type
        self.model = None
        self.label_names = []
        self.model_mtime = 0
        self.model_path = None
        self.labels_path = os.path.join(Config.MODELS_DIR, "label_names.pkl")
        self.use_hybrid_features = False
        self.feature_dim = Config.FEATURE_DIM
        self._last_mtime_check = 0

        if self.model_type is None:
            self.model_type = self._detect_model_type()

        if self.model_type:
            self.model_path = os.path.join(Config.MODELS_DIR, f"model_{self.model_type}.pkl")

        self.reload()

    def _detect_model_type(self):
        pattern = os.path.join(Config.MODELS_DIR, "model_*.pkl")
        model_files = [f for f in glob.glob(pattern) if not f.endswith("_meta.json")]
        if not model_files:
            return "svm"
        latest = max(model_files, key=os.path.getmtime)
        basename = os.path.basename(latest)
        model_type = basename.replace("model_", "").replace(".pkl", "")
        print(f"[Auto-detect] Using model: {model_type}")
        return model_type

    def reload(self):
        current_time = time.time()
        if current_time - self._last_mtime_check < 5.0 and self.model is not None:
            return True

        self._last_mtime_check = current_time

        if not self.model_path or not os.path.exists(self.model_path):
            self.model_type = self._detect_model_type()
            self.model_path = os.path.join(Config.MODELS_DIR, f"model_{self.model_type}.pkl")

        if not os.path.exists(self.model_path) or not os.path.exists(self.labels_path):
            print(f"[Warning] Model not found: {self.model_path}")
            return False

        current_mtime = os.path.getmtime(self.model_path)
        if current_mtime > self.model_mtime:
            try:
                self.model = joblib.load(self.model_path)
                self.label_names = joblib.load(self.labels_path)
                self.model_mtime = current_mtime
                
                # Load meta if exists
                meta_path = os.path.join(Config.MODELS_DIR, f"model_{self.model_type}_meta.json")
                if os.path.exists(meta_path):
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                        self.use_hybrid_features = meta.get("use_hybrid_features", False)
                        self.feature_dim = meta.get("feature_dim", Config.FEATURE_DIM)
                else:
                    self.use_hybrid_features = False
                    self.feature_dim = Config.FEATURE_DIM
                    
                print(f"[OK] Loaded model: {self.model_type} ({len(self.label_names)} classes). Hybrid: {self.use_hybrid_features}")
                return True
            except Exception as e:
                print(f"[ERR] Model load error: {e}")
                return False
        return True

    def predict(self, landmarks):
        """Predict sign from landmarks."""
        if self.model is None or not self.label_names:
            if not self.reload():
                return {"error": "Model not loaded. Train first."}

        # Check mtime periodically instead of every frame
        current_time = time.time()
        if current_time - self._last_mtime_check >= 5.0:
            self.reload()

        try:
            old_dim = Config.LANDMARKS_PER_HAND * Config.NUM_COORDS  # 63
            
            if self.use_hybrid_features:
                flat_lm = extract_features(landmarks)
            else:
                norm_lm = normalize_landmarks(landmarks)
                flat_lm = flatten_landmarks(norm_lm)
                # Backward compat: pad 1-hand (63) to 2-hand (126) if model expects 126
                if len(flat_lm) == old_dim and self.feature_dim == old_dim * 2:
                    flat_lm = np.concatenate([flat_lm, np.zeros(old_dim)])

            if len(flat_lm) != self.feature_dim:
                return {"error": f"Feature dim mismatch: got {len(flat_lm)}, expected {self.feature_dim}"}

            X_input = np.array([flat_lm])

            if hasattr(self.model, "predict_proba"):
                probs = self.model.predict_proba(X_input)[0]
                top_indices = np.argsort(probs)[::-1][:5]

                best_idx = top_indices[0]
                best_label = self.label_names[best_idx]
                best_confidence = float(probs[best_idx])
                
                final_label = best_label if best_confidence >= 0.3 else "unknown"

                top_5 = []
                for idx in top_indices:
                    if idx < len(self.label_names):
                        top_5.append({
                            "label": self.label_names[idx],
                            "confidence": float(probs[idx])
                        })

                return {
                    "label": final_label,
                    "display_name": final_label,
                    "confidence": best_confidence,
                    "top_5": top_5
                }
            else:
                pred_idx = self.model.predict(X_input)[0]
                best_label = self.label_names[pred_idx]
                return {
                    "label": best_label,
                    "display_name": best_label,
                    "confidence": 1.0,
                    "top_5": [{"label": best_label, "confidence": 1.0}]
                }

        except Exception as e:
            print(f"[ERR] Prediction error: {e}")
            return {"error": str(e)}
