import joblib
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

from utils import get_project_root, save_json
from config import load_config
from data import load_data
from models import get_models
from preprocessor import *
from evaluate import evaluate_model

def train() -> None:

    config = load_config()
    project_root = Path(get_project_root())

    target = config['data']['target']

    raw_path = project_root / config['data']['raw_path']
    model_path = project_root / config['model']['artifact_path']
    metrics_path = project_root / config['model']['metrics_path']
    feature_path = project_root / config['features']['artifact_path']

    test_size = config['model']['test_size']
    random_state = config['model']['random_state']
    df = load_data(raw_path, config)

    if target not in df.columns:
        raise ValueError(f"Target column not found after processing: {target}")
    
    X = df.drop(columns=[target])
    y = df[target]

    numeric_features, categorical_features = get_feature_columns()

    validate_feature_columns(
        X,
        numeric_features=numeric_features,
        categorical_features=categorical_features
    )

    save_feature_columns(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        feature_path=feature_path
    )

    print(f"Dataset shape: {df.shape}")
    print(f"Feature shape: {X.shape}")
    print(f"Target: {target}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    models = get_models(random_state=random_state)

    results = {}
    best_model_name = None
    best_model = None
    best_r2 = -np.inf

    for model_name, regressor in models.items():
        print(f"\nTraining model: {model_name}")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", create_preprocessor(numeric_features=numeric_features, categorical_features=categorical_features)),
                ("regressor", regressor)
            ]
        )

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        metrics = evaluate_model(y_test, y_pred)
        results[model_name] = metrics

        print(
            f"MAE: {metrics['mae']:,.2f} | "
            f"RMSE: {metrics['rmse']:,.2f} | "
            f"R2: {metrics['r2']:.4f}"
            ) 
        
        if metrics['r2'] > best_r2:
            best_r2 = metrics['r2']
            best_model_name = model_name
            best_model = pipeline

    if best_model is None:
        raise RuntimeError("No model was trained successfully.")
    
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    feature_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(best_model, model_path)

    final_report = {
        "best_model": best_model_name,
        "best_r2": best_r2,
        "all_models": results,
        "data_shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        },
        "features": list(X.columns),
        "target": target,
    }

    save_json(final_report, metrics_path)

    print("\nTraining complete.")
    print(f"Best model: {best_model_name}")
    print(f"Best R2: {best_r2:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")

if __name__ == "__main__":
    train()