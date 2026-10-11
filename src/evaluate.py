import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score

def evaluate_model(model, X_test, y_test, label_names):
    """Evaluate model on test set, return performance metrics."""
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    cm = confusion_matrix(y_test, y_pred)

    unique_labels = np.unique(y_test)
    target_names = [label_names[i] for i in unique_labels if i < len(label_names)]

    try:
        report = classification_report(
            y_test, y_pred,
            target_names=target_names,
            output_dict=True,
            zero_division=0
        )
    except Exception:
        report = {}

    return {
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "labels": [label_names[i] for i in unique_labels if i < len(label_names)]
    }

def cross_validate_model(model, X, y, cv=5):
    """Cross-validation on input data."""
    try:
        # Adjust cv if too few samples
        n_samples = len(X)
        n_classes = len(np.unique(y))
        actual_cv = min(cv, n_samples // max(n_classes, 1))
        actual_cv = max(2, actual_cv)

        scores = cross_val_score(model, X, y, cv=actual_cv, scoring='accuracy')
        return scores.tolist()
    except Exception as e:
        print(f"[WARN] Cross-validation error: {e}")
        return []
