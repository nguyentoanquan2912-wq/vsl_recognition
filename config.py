import os

class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    MODELS_DIR = os.path.join(BASE_DIR, "models")
    LANDMARKS_DIR = os.path.join(DATA_DIR, "landmarks")
    LABELS_PATH = os.path.join(DATA_DIR, "labels.json")
    
    # Flask
    SECRET_KEY = "vsl-recognition-2026"
    PORT = 5000
    DEBUG = False
    
    # MediaPipe
    MAX_NUM_HANDS = 2
    MIN_DETECTION_CONFIDENCE = 0.7
    MIN_TRACKING_CONFIDENCE = 0.5
    
    # Model
    NUM_HANDS = 2
    LANDMARKS_PER_HAND = 21
    NUM_LANDMARKS = LANDMARKS_PER_HAND * NUM_HANDS  # 42
    NUM_COORDS = 3  # x, y, z
    FEATURE_DIM = NUM_LANDMARKS * NUM_COORDS  # 126
    HYBRID_FEATURE_DIM = 176
    USE_HYBRID_FEATURES = True
    
    # Training
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    CV_FOLDS = 5
    MIN_SAMPLES_PER_CLASS = 10
    
    # TTS
    TTS_LANG = "vi"
    TTS_DIR = os.path.join(BASE_DIR, "static", "audio")
