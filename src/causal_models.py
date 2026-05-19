# src/causal_models.py
import numpy as np
from econml.dml import LinearDML
from lightgbm import LGBMRegressor, LGBMClassifier

class CausalUpliftEngine:
    def __init__(self):
        """Initializes Double Machine Learning with robust LightGBM estimators."""
        self.model = LinearDML(
            model_y=LGBMRegressor(n_estimators=100, max_depth=5, random_state=42),
            model_t=LGBMClassifier(n_estimators=100, max_depth=5, random_state=42),
            discrete_treatment=True,
            cv=3
        )

    def fit_pipeline(self, X_train, W_train, T_train, Y_train):
        """Fits the DML model on the training data partitions."""
        print("Training Double Machine Learning model...")
        self.model.fit(Y_train, T_train, X=X_train, W=W_train)
        return self

    def estimate_effects(self, X_test) -> np.ndarray:
        """Predicts the Conditional Average Treatment Effect (CATE / Uplift)."""
        # Returns the predicted incremental financial uplift for each specific individual
        return self.model.effect(X_test)
