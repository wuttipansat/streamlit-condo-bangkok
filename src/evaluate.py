from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import pandas as pd
from sklearn.inspection import permutation_importance

def evaluate_model(y_true, y_pred) -> dict:

    mae = mean_absolute_error(y_true, y_pred)
    rmse = root_mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }

def get_feature_importance(pipeline, X_test, y_test, random_state=42) -> pd.DataFrame:
    
    result = permutation_importance(
        estimator=pipeline,
        X=X_test,
        y=y_test,
        scoring='r2',
        n_repeats=10,
        random_state=random_state,
        n_jobs=-1
    )

    importance_df = pd.DataFrame({
        "feature": X_test.columns,
        "importance_mean": result.importances_mean,
        "importance_std": result.importances_std
    }).sort_values(by='importance_mean', ascending=False)

    importance_df = importance_df[importance_df["importance_mean"] > 0]

    return {
        "method": "permutation_importance",
        "scoring": "r2",
        "top_features": [{
            "feature": str(row['feature']),
            "importance_mean": round(float(row['importance_mean']), 4),
            "importance_std": round(float(row['importance_std']), 4)
        } for _, row in importance_df.head(20).iterrows()
        ]
    }