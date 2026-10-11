from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from config import Config

def create_svm_model():
    """SVM with RBF kernel and probability."""
    return SVC(
        kernel='rbf',
        probability=True,
        C=10,
        gamma='scale',
        random_state=Config.RANDOM_STATE
    )

def create_rf_model():
    """Random Forest classifier."""
    return RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        max_depth=None,
        min_samples_split=5,
        random_state=Config.RANDOM_STATE,
        n_jobs=-1
    )

def create_mlp_model():
    """Multi-layer Perceptron neural network."""
    return MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        alpha=0.001,
        batch_size='auto',
        learning_rate='adaptive',
        max_iter=1000,
        random_state=Config.RANDOM_STATE,
        early_stopping=True
    )

def create_knn_model():
    """K-Nearest Neighbors classifier."""
    return KNeighborsClassifier(
        n_neighbors=5,
        weights='distance',
        metric='minkowski',
        n_jobs=-1
    )

def get_model(model_type="svm"):
    """Factory function to create untrained model."""
    models = {
        "svm": create_svm_model,
        "rf": create_rf_model,
        "mlp": create_mlp_model,
        "knn": create_knn_model
    }

    model_type = model_type.lower()
    if model_type not in models:
        print(f"[WARN] Unknown model '{model_type}', using SVM default.")
        model_type = "svm"

    return models[model_type]()
