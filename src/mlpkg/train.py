import hashlib
from pathlib import Path

import joblib
import mlflow
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlpkg.cleaning import clean_features, clean_train

MODELS = {
    "ridge": Ridge,
    "random_forest": RandomForestRegressor,
    "gradient_boosting": GradientBoostingRegressor,
}


def sha256_of(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build_pipeline(model_name, settings, numeric, categorical):
    pre = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    return Pipeline([("pre", pre), ("model", MODELS[model_name](**settings))])


def main():
    with open("params.yaml") as f:
        p = yaml.safe_load(f)

    numeric = p["features"]["numeric"]
    categorical = p["features"]["categorical"]
    feats = numeric + categorical
    target = p["target"]

    train = clean_train(pd.read_csv(p["paths"]["train"]))
    later = clean_features(pd.read_csv(p["paths"]["later"]))
    X, y = train[feats], train[target]
    X_later, y_later = later[feats], later[target]

    data_hash = sha256_of(p["paths"]["train"])  

    mlflow.set_tracking_uri(p["mlflow"]["tracking_uri"])
    mlflow.set_experiment(p["mlflow"]["experiment_name"])

    results = {}
    for r in p["runs"]:
        with mlflow.start_run(run_name=r["name"]) as run:
            pipe = build_pipeline(r["model"], r["settings"], numeric, categorical)

            mlflow.log_param("data_sha256", data_hash)
            mlflow.log_param("model_type", r["model"])
            mlflow.log_params(r["settings"])

            cv = cross_val_score(pipe, X, y, cv=5, scoring="neg_mean_absolute_error")
            mae_cv = float(-cv.mean())
            pipe.fit(X, y)
            mae_later = float(mean_absolute_error(y_later, pipe.predict(X_later)))

            mlflow.log_metric("mae_cv", mae_cv)
            mlflow.log_metric("mae_later", mae_later)
            mlflow.sklearn.log_model(pipe, name="model", serialization_format="cloudpickle")

            results[run.info.run_id] = (mae_cv, pipe)
            print(f"{r['name']}: mae_cv={mae_cv:.4f} mae_later={mae_later:.4f} run_id={run.info.run_id}")

    
    best_id = min(results, key=lambda k: results[k][0])
    best_pipe = results[best_id][1]

    name = p["mlflow"]["registered_model_name"]
    mv = mlflow.register_model(f"runs:/{best_id}/model", name)  
    MlflowClient().set_registered_model_alias(name, "champion", mv.version)

    Path(p["paths"]["model_out"]).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipe, p["paths"]["model_out"])

    print("\nBEST RUN ID:", best_id)
    print("REGISTERED:", name, "version", mv.version, "alias champion")


if __name__ == "__main__":
    main()