import os
import json
import threading
import numpy as np
from flask import Flask, render_template, request, jsonify, send_file
from flask_socketio import SocketIO, emit

from config import Config

app = Flask(__name__)
app.config["SECRET_KEY"] = Config.SECRET_KEY
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

# Lazy-load predictor
_predictor = None
_predictor_lock = threading.Lock()

def get_predictor():
    global _predictor
    if _predictor is None:
        with _predictor_lock:
            if _predictor is None:
                from src.predictor import SignPredictor
                _predictor = SignPredictor()
    return _predictor

# Training state
_training_lock = threading.Lock()
_training_in_progress = False
_training_error = None


# ─── Page Routes ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/collect")
def collect():
    return render_template("collect.html")

@app.route("/recognize")
def recognize():
    return render_template("recognize.html")

@app.route("/train")
def train_page():
    return render_template("train.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")


# ─── Data Collection API ────────────────────────────────────────────────────

@app.route("/api/collect", methods=["POST"])
def api_collect():
    """Save a single landmark sample for a sign class."""
    data = request.get_json()
    label = data.get("label", "").strip()
    landmarks = data.get("landmarks")

    if not label or landmarks is None:
        return jsonify({"success": False, "error": "Thiếu label hoặc landmarks"}), 400

    save_dir = os.path.join(Config.LANDMARKS_DIR, label)
    os.makedirs(save_dir, exist_ok=True)

    existing = len([f for f in os.listdir(save_dir) if f.endswith(".npy")])
    filename = f"sample_{existing:05d}.npy"
    np.save(os.path.join(save_dir, filename), np.array(landmarks, dtype=np.float32))

    return jsonify({"success": True, "label": label, "count": existing + 1})


@app.route("/api/collect/batch", methods=["POST"])
def api_collect_batch():
    """Save multiple landmark samples at once."""
    data = request.get_json()
    label = data.get("label", "").strip()
    samples = data.get("samples", [])

    if not label or not samples:
        return jsonify({"success": False, "error": "Thiếu label hoặc samples"}), 400

    save_dir = os.path.join(Config.LANDMARKS_DIR, label)
    os.makedirs(save_dir, exist_ok=True)

    existing = len([f for f in os.listdir(save_dir) if f.endswith(".npy")])
    for i, lm in enumerate(samples):
        filename = f"sample_{existing + i:05d}.npy"
        np.save(os.path.join(save_dir, filename), np.array(lm, dtype=np.float32))

    return jsonify({"success": True, "label": label, "count": existing + len(samples), "added": len(samples)})


@app.route("/api/dataset/stats", methods=["GET"])
def api_dataset_stats():
    """Get dataset statistics per class."""
    stats = {}
    total = 0
    if os.path.exists(Config.LANDMARKS_DIR):
        for label in sorted(os.listdir(Config.LANDMARKS_DIR)):
            label_dir = os.path.join(Config.LANDMARKS_DIR, label)
            if os.path.isdir(label_dir):
                count = len([f for f in os.listdir(label_dir) if f.endswith(".npy")])
                stats[label] = count
                total += count
    return jsonify({
        "stats": stats,
        "total_samples": total,
        "total_classes": len(stats),
        "min_samples": min(stats.values()) if stats else 0,
        "max_samples": max(stats.values()) if stats else 0,
    })


@app.route("/api/dataset/delete/<label>", methods=["DELETE"])
def api_dataset_delete(label):
    """Delete all data for a label."""
    import shutil
    label_dir = os.path.join(Config.LANDMARKS_DIR, label)
    if os.path.exists(label_dir):
        shutil.rmtree(label_dir)
        return jsonify({"success": True, "message": f"Đã xóa dữ liệu '{label}'"})
    return jsonify({"success": False, "error": "Không tìm thấy label"}), 404


# ─── Labels API ──────────────────────────────────────────────────────────────

@app.route("/api/labels", methods=["GET"])
def api_labels():
    """Get defined sign labels."""
    if os.path.exists(Config.LABELS_PATH):
        with open(Config.LABELS_PATH, "r", encoding="utf-8") as f:
            labels = json.load(f)
        return jsonify({"success": True, **labels})
    return jsonify({"success": False, "error": "Không tìm thấy file labels"})


# ─── Training API ────────────────────────────────────────────────────────────

@app.route("/api/train", methods=["POST"])
def api_train():
    """Start model training in background thread."""
    global _training_in_progress, _training_error

    with _training_lock:
        if _training_in_progress:
            return jsonify({"success": False, "error": "Đang huấn luyện, vui lòng chờ"}), 409
        _training_in_progress = True
        _training_error = None

    data = request.get_json() or {}
    model_type = data.get("model_type", "svm")

    def run_training():
        global _training_in_progress, _training_error
        try:
            socketio.emit("training_progress", {"step": "data", "message": "Dang tai du lieu..."})
            from src.train import train_model
            
            socketio.emit("training_progress", {"step": "train", "message": "Dang huan luyen mo hinh..."})
            metrics = train_model(model_type=model_type)
            
            socketio.emit("training_progress", {"step": "eval", "message": "Dang danh gia mo hinh..."})
            
            # Reload predictor with new model
            global _predictor
            _predictor = None
            
            socketio.emit("training_progress", {"step": "done", "message": "Hoan tat!"})
            socketio.emit("training_complete", {"success": True, "metrics": metrics})
            print("[OK] Training complete!")
        except Exception as e:
            import traceback
            _training_error = str(e)
            tb = traceback.format_exc()
            print(f"[ERR] Training error: {tb}")
            socketio.emit("training_complete", {"success": False, "error": str(e)})
        finally:
            _training_in_progress = False

    socketio.start_background_task(run_training)

    return jsonify({"success": True, "message": "Da bat dau huan luyen mo hinh..."})


@app.route("/api/train/status", methods=["GET"])
def api_train_status():
    return jsonify({"in_progress": _training_in_progress, "error": _training_error})


# ─── Prediction API ─────────────────────────────────────────────────────────

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Predict sign from landmark data."""
    data = request.get_json()
    landmarks = data.get("landmarks")
    if landmarks is None:
        return jsonify({"success": False, "error": "Thiếu landmarks"}), 400

    predictor = get_predictor()
    result = predictor.predict(landmarks)
    return jsonify({"success": True, **result})


# ─── Model Metrics API ──────────────────────────────────────────────────────

@app.route("/api/models/metrics", methods=["GET"])
def api_model_metrics():
    """Get stored evaluation metrics."""
    metrics_path = os.path.join(Config.MODELS_DIR, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        return jsonify({"success": True, "metrics": metrics})
    return jsonify({"success": False, "error": "Chưa có metrics. Hãy huấn luyện mô hình trước."})


# ─── TTS API ─────────────────────────────────────────────────────────────────

@app.route("/api/tts", methods=["POST"])
def api_tts():
    """Text to speech — returns audio file."""
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"success": False, "error": "Thiếu text"}), 400

    from src.tts import speak_text
    audio_path = speak_text(text)
    if audio_path and os.path.exists(audio_path):
        return send_file(audio_path, mimetype="audio/mpeg")
    return jsonify({"success": False, "error": "TTS thất bại"}), 500


# ─── WebSocket Events ───────────────────────────────────────────────────────

@socketio.on("predict_frame")
def handle_predict_frame(data):
    """Real-time prediction from WebSocket."""
    landmarks = data.get("landmarks")
    if landmarks:
        predictor = get_predictor()
        result = predictor.predict(landmarks)
        emit("prediction_result", result)


@socketio.on("connect")
def handle_connect():
    print("[+] Client connected")


@socketio.on("disconnect")
def handle_disconnect():
    print("[-] Client disconnected")


# ─── Main ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs(Config.DATA_DIR, exist_ok=True)
    os.makedirs(Config.MODELS_DIR, exist_ok=True)
    os.makedirs(Config.LANDMARKS_DIR, exist_ok=True)
    os.makedirs(Config.TTS_DIR, exist_ok=True)

    print("=" * 60)
    print("  VN Check - Nhan Dien Ngon Ngu Ky Hieu Viet Nam")
    print("=" * 60)
    print(f"  Server: http://127.0.0.1:{Config.PORT}")
    print("=" * 60)

    socketio.run(app, host="0.0.0.0", port=Config.PORT, debug=Config.DEBUG, allow_unsafe_werkzeug=True)
