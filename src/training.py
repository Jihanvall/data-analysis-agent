from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, r2_score, accuracy_score, mean_absolute_error, make_scorer

CANDIDATES = {
    "classification": {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(random_state=42),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    },
    "regression": {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(random_state=42),
        "gradient_boosting": GradientBoostingRegressor(random_state=42),
    },
    "clustering": {
        "kmeans": KMeans(n_clusters=3, random_state=42, n_init=10),
    },
}

IMBALANCE_THRESHOLD = 0.1


def _get_scoring(y_train):
    counts = y_train.value_counts(normalize=True)
    minority_ratio = counts.min()

    if minority_ratio < IMBALANCE_THRESHOLD:
        minority_class = counts.idxmin()
        return make_scorer(f1_score, pos_label=minority_class, zero_division=0), "f1_minority_class"

    return "f1_weighted", "f1_weighted"


def select_and_train(X_train, y_train, task_type):
    if task_type not in CANDIDATES:
        raise ValueError(f"task_type must be one of {list(CANDIDATES)}")

    if task_type == "clustering":
        model = CANDIDATES["clustering"]["kmeans"]
        model.fit(X_train)
        return model, "kmeans", {}

    if task_type == "classification":
        scoring, scoring_name = _get_scoring(y_train)
    else:
        scoring, scoring_name = "r2", "r2"

    scores = {}
    for name, model in CANDIDATES[task_type].items():
        cv = cross_val_score(model, X_train, y_train, cv=5, scoring=scoring)
        scores[name] = float(cv.mean())

    best_name = max(scores, key=scores.get)
    best_model = CANDIDATES[task_type][best_name]
    best_model.fit(X_train, y_train)

    return best_model, best_name, {"scoring_used": scoring_name, "scores": scores}


def evaluate(model, X_test, y_test, task_type):
    if task_type == "clustering":
        labels = model.predict(X_test)
        return {"n_clusters": len(set(labels))}

    preds = model.predict(X_test)
    if task_type == "classification":
        counts = y_test.value_counts()
        minority_class = counts.idxmin()
        return {
            "f1_weighted": float(f1_score(y_test, preds, average="weighted")),
            "f1_minority_class": float(f1_score(y_test, preds, pos_label=minority_class, zero_division=0)),
            "accuracy": float(accuracy_score(y_test, preds)),
        }
    return {"r2": float(r2_score(y_test, preds)), "mae": float(mean_absolute_error(y_test, preds))}