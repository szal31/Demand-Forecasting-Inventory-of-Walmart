"""Tune the LightGBM forecaster with rolling-origin cross-validation and Optuna, tracked in MLflow.

Only weeks before the 26-week test period are used: 4 folds, each validating on the next 13 weeks
after training on everything before it. The best parameters go to output/best_params.json,
which forecast.py then uses.

Usage:  python tune.py            (about 5 minutes)
        mlflow ui --backend-store-uri sqlite:///mlflow.db     to browse the runs
"""
import json

import mlflow
import numpy as np
import optuna

from forecast import BEST_PARAMS, DEFAULT_PARAMS, OUT, ROOT, TEST_WEEKS, add_features, build_weekly, lightgbm_forecast

FOLDS, FOLD_WEEKS, TRIALS = 4, 13, 50


def make_folds(weeks):
    """(train_end, val_mask_weeks) pairs walking back from the test start; the newest fold is last."""
    dev = weeks[:-TEST_WEEKS]
    return [(dev[-(k + 1) * FOLD_WEEKS], dev[-(k + 1) * FOLD_WEEKS: len(dev) - k * FOLD_WEEKS])
            for k in reversed(range(FOLDS))]


def wape(pred, actual):
    return np.abs(pred - actual).sum() / actual.sum()


def cv_scores(df, folds, params):
    scores = []
    for train_end, val_weeks in folds:
        mask = df.week_start.isin(val_weeks)
        pred, _ = lightgbm_forecast(df, train_end, mask, params)
        scores.append(wape(pred, df.loc[mask, "units"].to_numpy()))
    return scores


def baseline_scores(df, folds, column):
    return [wape(df.loc[df.week_start.isin(v), column], df.loc[df.week_start.isin(v), "units"]) for _, v in folds]


def main():
    OUT.mkdir(exist_ok=True)
    df = add_features(build_weekly())
    weeks = np.sort(df.week_start.unique())
    folds = make_folds(weeks)
    # Self-check: no fold may touch the test weeks, and folds must not overlap.
    assert max(v.max() for _, v in folds) < weeks[-TEST_WEEKS]
    assert len(np.unique(np.concatenate([v for _, v in folds]))) == FOLDS * FOLD_WEEKS

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("demand-forecasting")
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    with mlflow.start_run(run_name="optuna-tuning"):
        baselines = {"naive": baseline_scores(df, folds, "lag_1"), "moving_avg_4w": baseline_scores(df, folds, "mean_4"),
                     "lightgbm_default": cv_scores(df, folds, DEFAULT_PARAMS)}
        for name, s in baselines.items():
            mlflow.log_metric(f"cv_wape_{name}", float(np.mean(s)))
        mlflow.log_params({"folds": FOLDS, "fold_weeks": FOLD_WEEKS, "trials": TRIALS})

        def objective(trial):
            params = {
                "objective": trial.suggest_categorical("objective", ["regression_l1", "regression", "tweedie", "poisson"]),
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=50),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "num_leaves": trial.suggest_int("num_leaves", 8, 128, log=True),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0), "subsample_freq": 1,
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10, log=True),
            }
            scores = cv_scores(df, folds, params)
            with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("cv_wape", float(np.mean(scores)))
                for i, s in enumerate(scores):
                    mlflow.log_metric(f"fold_{i}_wape", float(s))
            return float(np.mean(scores))

        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=TRIALS)

        best = {**study.best_params, "subsample_freq": 1}
        BEST_PARAMS.write_text(json.dumps(best, indent=2))
        mlflow.log_metric("cv_wape_best", study.best_value)
        mlflow.log_artifact(str(BEST_PARAMS))

    print(f"Rolling-origin CV, {FOLDS} folds x {FOLD_WEEKS} weeks (mean WAPE):")
    for name, s in baselines.items():
        print(f"  {name:18s} {np.mean(s):.1%}   folds: {', '.join(f'{x:.1%}' for x in s)}")
    print(f"  {'lightgbm_tuned':18s} {study.best_value:.1%}   ({TRIALS} Optuna trials)")
    print("Best params:", json.dumps(best))


if __name__ == "__main__":
    main()
