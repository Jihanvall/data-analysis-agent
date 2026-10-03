import warnings

import joblib
import sklearn

BUNDLE_FORMAT_VERSION = 1

REQUIRED_KEYS = {
    "model", 
    "model_name",
    "fitted",
    "feature_columns",
    "input_columns",
    "target",
    "task_type",
    "sklearn_version",
    "format_version",
}

def build_bundle(model, model_name, fitted, feature_columns, input_columns, target, task_type):
    return {
        "model": model,
        "model_name": model_name,
        "fitted": fitted,
        "feature_columns": list(feature_columns),
        "input_columns": list(input_columns),
        "target": target,
        "task_type": task_type,
        "sklearn_version": sklearn.__version__,
        "format_version": BUNDLE_FORMAT_VERSION,

    }

def save_bundle(bundle, path):
    joblib.dump(bundle, path)
    return path

def load_bundle(path):
    bundle = joblib.load(path)

    if not isinstance(bundle, dict):
        raise ValueError("Invalid model bundle: expected a dictionary")

    missing = REQUIRED_KEYS - set(bundle)
    if missing:
        raise ValueError(f"Invalide model bundle, missing keys: {sorted(missing)}")

    if bundle["sklearn_version"] != sklearn.__version__:
        warnings.warn(
            f"Model was saved with scikit-learn {bundle['sklearn_version']} "
            f"but {sklearn.__version__} is installed. Predictions may be unreliable."
        )

    return bundle
    