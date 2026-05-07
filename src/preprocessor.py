from pathlib import Path
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_feature_columns():
    numeric_features = [
        "no_bedroom",
        "floor",
        "room_size",
        "amenity_count",
        "kitchen_ready",
        "move_in_ready",
        "comfort_score",
        "is_popular_project",
    ]

    categorical_features = [
        "poster_status",
        "unit_type",
        "project_name_grouped",
        "room_size_group",
    ]

    return numeric_features, categorical_features


def validate_feature_columns(df, numeric_features, categorical_features):
    required_columns = numeric_features + categorical_features

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required feature columns: {missing_columns}"
        )


def save_feature_columns(numeric_features, categorical_features, feature_path):
    feature_path = Path(feature_path)
    feature_path.parent.mkdir(parents=True, exist_ok=True)

    feature_columns = {
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "all_features": numeric_features + categorical_features,
    }

    joblib.dump(feature_columns, feature_path)


def load_feature_columns(feature_path):
    feature_path = Path(feature_path)

    if not feature_path.exists():
        raise FileNotFoundError(
            f"Feature column artifact not found: {feature_path}"
        )

    return joblib.load(feature_path)


def create_preprocessor(numeric_features, categorical_features):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    return preprocessor