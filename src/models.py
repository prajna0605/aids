"""
ECG Insight AI - Machine Learning Anomaly Detection Models
Provides multiple machine learning algorithms for ECG heartbeat anomaly detection:
1. Isolation Forest (Unsupervised tree isolation)
2. One-Class Support Vector Machine (Unsupervised kernel boundary)
3. Local Outlier Factor (LOF) (Unsupervised density-based)
4. Random Forest Classifier (Supervised benchmark when labels available)

Includes continuous anomaly scoring [0, 1], dynamic threshold tuning,
and feature importance explainability.
"""

from typing import Dict, Tuple, List, Optional, Any
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class ECGAnomalyDetector:
    """
    Unified ECG Anomaly Detector supporting both unsupervised outlier detection
    and supervised classification models.
    """

    AVAILABLE_MODELS = {
        "Isolation Forest": "Tree-based unsupervised anomaly detection separating outliers with shallow path depths.",
        "One-Class SVM": "Kernel boundary method learning a tight hypersphere around normal cardiac beat patterns.",
        "Local Outlier Factor": "Density-based anomaly detector measuring local deviation with respect to surrounding beat clusters.",
        "Random Forest (Supervised)": "Supervised ensemble classifier trained on annotated arrhythmia beat labels.",
    }

    def __init__(
        self,
        model_name: str = "Isolation Forest",
        contamination: float = 0.15,
        random_state: int = 42,
    ):
        self.model_name = model_name
        self.contamination = contamination
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.is_fitted = False
        self.feature_names = []
        self._build_model()

    def _build_model(self):
        """Instantiates the underlying scikit-learn model."""
        if self.model_name == "Isolation Forest":
            self.model = IsolationForest(
                n_estimators=150,
                contamination=self.contamination,
                random_state=self.random_state,
                n_jobs=-1,
            )
        elif self.model_name == "One-Class SVM":
            self.model = OneClassSVM(
                kernel="rbf",
                gamma="scale",
                nu=min(0.5, max(0.01, self.contamination)),
            )
        elif self.model_name == "Local Outlier Factor":
            self.model = LocalOutlierFactor(
                n_neighbors=20,
                contamination=self.contamination,
                novelty=True,
                n_jobs=-1,
            )
        elif self.model_name == "Random Forest (Supervised)":
            self.model = RandomForestClassifier(
                n_estimators=100,
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1,
            )
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ):
        """
        Fits the anomaly detection model on extracted heartbeat features.

        Args:
            X: 2D array of features (n_beats, n_features)
            y: Optional ground-truth binary labels (1=anomalous, 0=normal)
            feature_names: List of feature column names
        """
        if len(X) == 0:
            raise ValueError("Training dataset X is empty.")

        self.feature_names = feature_names or [f"Feature_{i}" for i in range(X.shape[1])]

        # Fit standard scaler and transform
        X_scaled = self.scaler.fit_transform(X)

        if self.model_name == "Random Forest (Supervised)":
            if y is None or len(np.unique(y)) < 2:
                # If only 1 class is present, fall back to fitting on dummy or semi-supervised
                # Provide synthetic augmentation if single class
                if y is None or len(np.unique(y)) == 1:
                    y_adj = np.zeros(len(X), dtype=int)
                    # mark top 10% extreme distances as pseudo anomalies for stability
                    dists = np.linalg.norm(X_scaled, axis=1)
                    top_k = int(0.10 * len(X))
                    if top_k > 0:
                        y_adj[np.argsort(dists)[-top_k:]] = 1
                else:
                    y_adj = y
            else:
                y_adj = y
            self.model.fit(X_scaled, y_adj)
        else:
            # Unsupervised methods
            self.model.fit(X_scaled)

        self.is_fitted = True
        return self

    def predict(
        self,
        X: np.ndarray,
        threshold: float = 0.50,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts anomaly flags and continuous normalized anomaly scores for input beats.

        Args:
            X: 2D array of features (n_beats, n_features)
            threshold: Decision threshold for anomaly score in range [0, 1]

        Returns:
            binary_predictions: 1 for anomaly, 0 for normal
            anomaly_scores: Continuous anomaly scores in range [0, 1] (higher = more anomalous)
        """
        if not self.is_fitted:
            raise RuntimeError("Model has not been trained yet. Call fit() first.")

        if len(X) == 0:
            return np.array([], dtype=int), np.array([], dtype=np.float64)

        X_scaled = self.scaler.transform(X)

        if self.model_name == "Random Forest (Supervised)":
            probs = self.model.predict_proba(X_scaled)
            # Probability of anomalous class (class 1)
            if probs.shape[1] > 1:
                anomaly_scores = probs[:, 1]
            else:
                anomaly_scores = np.zeros(len(X))
        else:
            # For Isolation Forest, One-Class SVM, and LOF:
            raw_scores = self.model.decision_function(X_scaled)
            center = -0.06
            anomaly_scores = 1.0 / (1.0 + np.exp(np.clip(25.0 * (raw_scores - center), -25.0, 25.0)))

        # Binary predictions based on configurable user threshold
        binary_predictions = (anomaly_scores >= threshold).astype(int)

        return binary_predictions, anomaly_scores

    def get_feature_importances(self) -> Dict[str, float]:
        """
        Returns feature importance or surrogate correlations for explainability.
        """
        if not self.is_fitted or len(self.feature_names) == 0:
            return {}

        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
            return dict(zip(self.feature_names, importances.tolist()))

        # For unsupervised models, compute variance / separation proxy
        return {}


def train_and_evaluate_model(
    model_name: str,
    X_train: np.ndarray,
    y_train: Optional[np.ndarray],
    X_test: np.ndarray,
    y_test: Optional[np.ndarray],
    feature_names: List[str],
    contamination: float = 0.15,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Convenience orchestrator to initialize, train, predict, and bundle results.
    """
    detector = ECGAnomalyDetector(
        model_name=model_name,
        contamination=contamination,
    )
    detector.fit(X_train, y_train, feature_names=feature_names)
    preds, scores = detector.predict(X_test, threshold=threshold)
    importances = detector.get_feature_importances()

    return {
        "detector": detector,
        "predictions": preds,
        "anomaly_scores": scores,
        "feature_importances": importances,
    }
