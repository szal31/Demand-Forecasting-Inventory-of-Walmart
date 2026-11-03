"""Weekly demand forecasting and inventory planning for 50 grocery SKUs (Walmart M5, store CA_1).

1. Build a weekly sales panel with price and calendar features.
2. Backtest one-week-ahead forecasts: naive, 4-week moving average, seasonal naive, LightGBM.
3. Turn each forecast into an order-up-to policy (ABC service levels) and simulate it on the test weeks.

Usage:  python forecast.py
"""
import json
from pathlib import Path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).parent
RAW, OUT = ROOT / "data" / "raw", ROOT / "output"
STORE, DEPT, N_ITEMS = "CA_1", "FOODS_3", 50
TEST_WEEKS, VAL_WEEKS = 26, 13
LEAD_TIME, REVIEW = 1, 1                       # weeks: order weekly, delivery next week
SERVICE = {"A": 0.98, "B": 0.95, "C": 0.90}    # target cycle service level by ABC class
FILL_TARGET = 0.98                             # fill rate used to compare methods fairly
METHODS = ["Naive", "Moving avg (4w)", "Seasonal naive", "LightGBM"]
INK, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#fcfcfb", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"


# ---------- 1. data ----------
def build_weekly():
    cache = ROOT / "data" / "weekly.csv"
    if cache.exists():
        return pd.read_csv(cache, parse_dates=["week_start"])

    sales = pd.read_csv(RAW / "sales_train_evaluation.csv")
    sales = sales[(sales.store_id == STORE) & (sales.dept_id == DEPT)]
    cal = pd.read_csv(RAW / "calendar.csv", parse_dates=["date"])
    cal["d"] = "d_" + (cal.index + 1).astype(str)  # this mirror has no "d" column; d_1 = first row
    days = [c for c in sales.columns if c.startswith("d_")]
    long = sales.melt(id_vars="item_id", value_vars=days, var_name="d", value_name="units").merge(cal, on="d")
    long["event"] = long["event_name_1"].notna()

    weekly = long.groupby(["item_id", "wm_yr_wk"]).agg(
        week_start=("date", "min"), units=("units", "sum"), n_days=("d", "count"),
        snap_days=("snap_CA", "sum"), event_days=("event", "sum")).reset_index()
    weekly = weekly[weekly.n_days == 7].drop(columns="n_days")  # last week in the file is partial

    recent = weekly[weekly.week_start >= weekly.week_start.max() - pd.Timedelta(weeks=52)]
    top = recent.groupby("item_id")["units"].sum().nlargest(N_ITEMS).index
    weekly = weekly[weekly.item_id.isin(top)]

    prices = pd.read_csv(RAW / "sell_prices.csv")
    prices = prices[(prices.store_id == STORE) & prices.item_id.isin(top)]
    weekly = weekly.merge(prices[["item_id", "wm_yr_wk", "sell_price"]], on=["item_id", "wm_yr_wk"], how="inner")
    weekly = weekly[weekly.week_start >= "2013-01-01"].sort_values(["item_id", "week_start"])
    weekly.to_csv(cache, index=False)
    return weekly


def add_features(df):
    df = df.sort_values(["item_id", "week_start"]).copy()
    g = df.groupby("item_id")["units"]
    for lag in (1, 2, 3, 4, 52):
        df[f"lag_{lag}"] = g.shift(lag)
    # every rolling feature starts from shift(1): only past weeks are visible
    df["mean_4"] = g.transform(lambda s: s.shift(1).rolling(4).mean())
    df["mean_13"] = g.transform(lambda s: s.shift(1).rolling(13).mean())
    df["std_13"] = g.transform(lambda s: s.shift(1).rolling(13).std())
    # prices are set in advance, so this week's price is a fair input
    df["price_ratio"] = df.sell_price / df.groupby("item_id")["sell_price"].transform(lambda s: s.rolling(13, 1).mean())
    df["week_of_year"] = df.week_start.dt.isocalendar().week.astype(int)
    df["item"] = df.item_id.astype("category")
    return df


FEATURES = ["lag_1", "lag_2", "lag_3", "lag_4", "lag_52", "mean_4", "mean_13", "std_13",
            "sell_price", "price_ratio", "snap_days", "event_days", "week_of_year", "item"]


# ---------- 2. forecasts ----------
DEFAULT_PARAMS = dict(objective="regression_l1", n_estimators=400, learning_rate=0.03, num_leaves=31,
                      min_child_samples=20, subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
BEST_PARAMS = OUT / "best_params.json"  # written by tune.py; tuning never sees the test weeks


def lgb_params():
    return {**DEFAULT_PARAMS, **json.loads(BEST_PARAMS.read_text())} if BEST_PARAMS.exists() else DEFAULT_PARAMS


def lightgbm_forecast(df, train_end, pred_mask, params=None):
    train = df[(df.week_start < train_end)].dropna(subset=["lag_52"])
    model = lgb.LGBMRegressor(**(params or lgb_params()), random_state=42, verbose=-1)
    model.fit(train[FEATURES], train["units"])
    return model.predict(df.loc[pred_mask, FEATURES]).clip(0), model


def forecasts(df, weeks):
    val_start, test_start = weeks[-(TEST_WEEKS + VAL_WEEKS)], weeks[-TEST_WEEKS]
    df["Naive"] = df.lag_1
    df["Moving avg (4w)"] = df.mean_4
    df["Seasonal naive"] = df.lag_52.fillna(df.mean_4)  # items younger than a year fall back to the 4w average
    df["LightGBM"] = np.nan
    val = (df.week_start >= val_start) & (df.week_start < test_start)
    test = df.week_start >= test_start
    df.loc[val, "LightGBM"], _ = lightgbm_forecast(df, val_start, val)    # for safety-stock sizing
    df.loc[test, "LightGBM"], model = lightgbm_forecast(df, test_start, test)
    df["period"] = np.select([val, test], ["val", "test"], "train")
    return df, model


def accuracy(df, period):
    d = df[df.period == period]
    rows = []
    for m in METHODS:
        err = d[m] - d.units
        rows.append({"method": m, "WAPE": err.abs().sum() / d.units.sum(), "bias": err.sum() / d.units.sum()})
    return pd.DataFrame(rows).set_index("method")


# ---------- 3. inventory ----------
def abc(df):
    rev = df[df.period == "val"].assign(rev=lambda x: x.units * x.sell_price).groupby("item_id")["rev"].sum()
    share = rev.sort_values(ascending=False).cumsum() / rev.sum()
    return pd.cut(share, [0, 0.8, 0.95, 1.0], labels=list("ABC"), include_lowest=True).astype(str)


def simulate(demand, forecast, safety, price):
    """Periodic review order-up-to, lost sales. Each week: receive last week's order, reorder, then sell."""
    protect = LEAD_TIME + REVIEW
    on_hand, pipeline = forecast[0] * protect + safety, [0.0] * LEAD_TIME
    sold = inv = stockouts = 0.0
    for d, f in zip(demand, forecast):
        on_hand += pipeline.pop(0)
        order = max(0.0, f * protect + safety - (on_hand + sum(pipeline)))
        pipeline.append(order)
        s = min(on_hand, d)
        sold, on_hand = sold + s, on_hand - s
        stockouts += s < d
        inv += on_hand
    weeks = len(demand)
    return {"fill_rate": sold / demand.sum(), "avg_units": inv / weeks,
            "avg_value": inv / weeks * price, "stockout_weeks": stockouts}


def inventory(df, classes, scale=1.0):
    """Simulate every item x method. `scale` multiplies all safety stock (used to trace the trade-off curve)."""
    rows = []
    for item, d in df.groupby("item_id"):
        val, test = d[d.period == "val"], d[d.period == "test"]
        z = norm.ppf(SERVICE[classes[item]])
        price = test.sell_price.mean()
        for m in METHODS:
            sigma = (val[m] - val.units).std()       # forecast error spread, measured before the test
            safety = scale * z * sigma * np.sqrt(LEAD_TIME + REVIEW)
            r = simulate(test.units.to_numpy(float), test[m].to_numpy(float), safety, price)
            rows.append({"item_id": item, "abc": classes[item], "method": m, "safety_stock": safety, **r,
                         "demand": test.units.sum()})
    return pd.DataFrame(rows)


def tradeoff(df, classes):
    """Fill rate vs inventory value for each method as safety stock is scaled from 0x to 2.5x."""
    pts = [summarise(inventory(df, classes, k)).assign(scale=k) for k in np.arange(0, 2.51, 0.25)]
    return pd.concat(pts).rename_axis("method").reset_index()


def value_at_fill(curve, target):
    """Inventory value each method needs to reach `target` fill rate (linear interpolation on the curve)."""
    out = {}
    for m, c in curve.groupby("method"):
        c = c.sort_values("fill_rate")
        out[m] = np.interp(target, c.fill_rate, c.avg_inventory_value) if c.fill_rate.max() >= target else np.nan
    return pd.Series(out).loc[METHODS]


def summarise(inv):
    g = inv.groupby("method")
    out = pd.DataFrame({
        "fill_rate": g.apply(lambda x: (x.fill_rate * x.demand).sum() / x.demand.sum()),
        "avg_inventory_units": g.avg_units.sum(),
        "avg_inventory_value": g.avg_value.sum(),
        "safety_stock_units": g.safety_stock.sum(),
        "stockout_item_weeks": g.stockout_weeks.sum(),
    }).loc[METHODS]
    return out


# ---------- 4. charts ----------
def style(ax, title):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=12, color=INK, pad=12)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, facecolor=SURFACE)
    plt.close(fig)


def charts(df, acc, curve):
    # 1. accuracy
    fig, ax = plt.subplots(figsize=(8, 3.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style(ax, "Forecast error by method (WAPE, 26 test weeks, lower is better)")
    a = acc["WAPE"].iloc[::-1]
    bars = ax.barh(a.index, a.values * 100, height=0.6,
                   color=[BLUE if m == "LightGBM" else "#b9cde8" for m in a.index])
    for b, v in zip(bars, a.values):
        ax.text(b.get_width() + 0.5, b.get_y() + b.get_height() / 2, f"{v:.1%}", va="center", fontsize=8, color=MUTED)
    ax.xaxis.set_visible(False)
    save(fig, "1_accuracy.png")

    # 2. example item
    item = df[df.period == "test"].groupby("item_id")["units"].sum().idxmax()
    d = df[(df.item_id == item) & (df.week_start >= df[df.period == "val"].week_start.min() - pd.Timedelta(weeks=26))]
    t = d[d.period == "test"]
    fig, ax = plt.subplots(figsize=(9, 3.8), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style(ax, f"{item}: weekly units sold vs LightGBM forecast")
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.plot(d.week_start, d.units, color=MUTED, linewidth=1.5)
    ax.plot(t.week_start, t.LightGBM, color=BLUE, linewidth=2)
    ax.axvline(t.week_start.min(), color=GRID, linewidth=1)
    ax.text(d.week_start.iloc[2], d.units.max() * 0.98, "Actual", color=MUTED, fontsize=8)
    ax.text(t.week_start.iloc[-1], t.LightGBM.iloc[-1], "  Forecast", color=BLUE, fontsize=8, va="center")
    ax.text(t.week_start.min(), d.units.max() * 0.98, "  test period →", color=MUTED, fontsize=8, va="top")
    save(fig, "2_example_item.png")

    # 3. inventory trade-off curve
    fig, ax = plt.subplots(figsize=(9, 4.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style(ax, f"Inventory needed for each fill rate (dashed line = {FILL_TARGET:.0%} target)")
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    shades = {"Naive": "#8a8984", "Moving avg (4w)": "#b5b4ae", "Seasonal naive": "#d0cfca", "LightGBM": BLUE}
    for m in METHODS:
        c = curve[curve.method == m].sort_values("fill_rate")
        ax.plot(c.fill_rate * 100, c.avg_inventory_value / 1000, color=shades[m],
                linewidth=2.5 if m == "LightGBM" else 1.5, marker="o", markersize=3)
        ax.text(c.fill_rate.iloc[-1] * 100 + 0.1, c.avg_inventory_value.iloc[-1] / 1000, m,
                color=INK if m == "LightGBM" else MUTED, fontsize=8, va="center")
    ax.axvline(FILL_TARGET * 100, color=MUTED, linewidth=0.8, linestyle="--")
    ax.set_xlabel("Fill rate (% of demand served)", fontsize=8, color=MUTED)
    ax.set_ylabel("Avg inventory on hand ($k)", fontsize=8, color=MUTED)
    ax.set_xlim(curve.fill_rate.min() * 100 - 0.5, 100.6)
    save(fig, "3_inventory_tradeoff.png")


def shap_importance(model, test):
    """Mean |SHAP value| per feature on the test weeks, in units sold (LightGBM's built-in TreeSHAP)."""
    contrib = model.predict(test[FEATURES], pred_contrib=True)  # last column is the base value
    # Self-check: SHAP values plus base value must add back up to the raw prediction.
    assert np.allclose(contrib.sum(axis=1), model.predict(test[FEATURES]), atol=1e-3)
    return pd.Series(np.abs(contrib[:, :-1]).mean(axis=0), FEATURES).sort_values(ascending=False)


FEATURE_LABELS = {
    "lag_1": "Sales last week", "lag_2": "Sales 2 weeks ago", "lag_3": "Sales 3 weeks ago",
    "lag_4": "Sales 4 weeks ago", "lag_52": "Sales same week last year", "mean_4": "4-week average",
    "mean_13": "13-week average", "std_13": "13-week volatility", "item": "Which product",
    "snap_days": "SNAP benefit days", "week_of_year": "Week of year", "event_days": "Holiday/event days",
    "price_ratio": "Price vs recent price", "sell_price": "Price",
}


def shap_chart(impact):
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    style(ax, "What drives the forecast: average impact per feature (units/week, SHAP)")
    s = impact.iloc[::-1].rename(FEATURE_LABELS)
    bars = ax.barh(s.index, s.values, height=0.6, color=BLUE)
    for b, v in zip(bars, s.values):
        ax.text(b.get_width() + s.max() * 0.01, b.get_y() + b.get_height() / 2, f"{v:.1f}", va="center",
                fontsize=8, color=MUTED)
    ax.xaxis.set_visible(False)
    save(fig, "4_shap.png")


def log_run(acc, needed):
    import mlflow  # imported here so the forecast itself runs without MLflow installed
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("demand-forecasting")
    with mlflow.start_run(run_name="test-backtest"):
        mlflow.log_params({**lgb_params(), "tuned": BEST_PARAMS.exists(), "test_weeks": TEST_WEEKS})
        for m in METHODS:
            key = m.lower().replace(" ", "_").replace("(", "").replace(")", "")
            mlflow.log_metric(f"test_wape_{key}", float(acc.loc[m, "WAPE"]))
            mlflow.log_metric(f"inventory_at_98_fill_{key}", float(needed[m]))
        for chart in OUT.glob("*.png"):
            mlflow.log_artifact(str(chart))


def main():
    OUT.mkdir(exist_ok=True)
    df = add_features(build_weekly())
    weeks = np.sort(df.week_start.unique())
    df, model = forecasts(df, weeks)
    acc = accuracy(df, "test")
    classes = abc(df)
    inv = inventory(df, classes)
    summary = summarise(inv)
    curve = tradeoff(df, classes)
    needed = value_at_fill(curve, FILL_TARGET)

    # Self-checks
    test = df[df.period == "test"]
    assert test.item_id.nunique() == N_ITEMS and test.week_start.nunique() == TEST_WEEKS
    assert test[METHODS].notna().all().all(), "every method needs a forecast for every test week"
    assert (inv.fill_rate.between(0, 1)).all() and (inv.avg_units >= 0).all()
    assert set(classes.index) == set(test.item_id)

    acc.to_csv(OUT / "accuracy.csv")
    summary.to_csv(OUT / "inventory_summary.csv")
    plan = inv[inv.method == "LightGBM"].merge(
        test.sort_values("week_start").groupby("item_id").tail(1)[["item_id", "LightGBM"]], on="item_id")
    plan["reorder_up_to"] = plan.LightGBM * (LEAD_TIME + REVIEW) + plan.safety_stock
    plan[["item_id", "abc", "safety_stock", "reorder_up_to", "fill_rate"]].round(
        {"safety_stock": 0, "reorder_up_to": 0, "fill_rate": 3}) \
        .sort_values(["abc", "item_id"]).to_csv(OUT / "reorder_plan.csv", index=False)
    impact = shap_importance(model, test)
    impact.to_csv(OUT / "shap_importance.csv", header=["mean_abs_shap_units"])
    curve.to_csv(OUT / "tradeoff_curve.csv", index=False)
    charts(df, acc, curve)
    shap_chart(impact)
    log_run(acc, needed)

    print(f"{N_ITEMS} items, {pd.Timestamp(weeks.min()):%d %b %Y} to {pd.Timestamp(weeks.max()):%d %b %Y}; "
          f"test = last {TEST_WEEKS} weeks from {pd.Timestamp(weeks[-TEST_WEEKS]):%d %b %Y}")
    print("ABC classes:", classes.value_counts().to_dict(), "\n")
    print("Accuracy on test weeks:")
    print(acc.map("{:.1%}".format).to_string(), "\n")
    print("Inventory simulation on test weeks (ABC service targets):")
    print(summary.round(3).to_string(), "\n")
    print(f"Avg inventory value needed for a {FILL_TARGET:.0%} fill rate:")
    print(needed.map("${:,.0f}".format).to_string())


if __name__ == "__main__":
    main()
