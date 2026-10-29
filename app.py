"""Waste Intelligence - Streamlit dashboard.

Run:  python -m streamlit run app.py

Works with no trained CNN, no GPU, no Spark and no HDFS. Synthetic data is always
labelled SIMULATED.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

import analytics
import config
import data_io
import generate_data
import image_classifier as ic
import recommendations as rec
from integrations import hdfs_storage, spark_processor

st.set_page_config(page_title="Waste Intelligence", page_icon="♻️", layout="wide")

PAGES = ["1. Overview", "2. Classify Image", "3. Waste Categories", "4. Zone Clustering",
         "5. 7-Day Forecast", "6. Capacity & Recommendations", "7. Data Source & Status"]
CAT_COLORS = {"Paper": "#3B82F6", "Plastic": "#F59E0B", "Glass": "#14B8A6",
              "Metal": "#64748B", "Organic": "#22C55E", "Other": "#A855F7"}
TEMPLATE = "plotly_white"


# ---------------------------------------------------------------------------
# Cached helpers
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_classifier() -> ic.ClassifierService:
    return ic.ClassifierService()


def _file_stamp(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner="Loading data...")
def cached_load(source: str, upload_bytes: bytes | None, upload_name: str, stamp: float) -> data_io.DataBundle:
    import io
    return data_io.load_bundle(source, io.BytesIO(upload_bytes) if upload_bytes else None, upload_name)


@st.cache_data(show_spinner="Clustering zones (K-Means, K search)...")
def cached_clustering(_daily, _zones, key, level, k, k_min, k_max):
    return analytics.cluster_zones(_daily, level, k, k_min, k_max, zones=_zones)


@st.cache_data(show_spinner="Backtesting forecasters on the held-out test period...")
def cached_evaluation(_daily, _zones, key, trees, k_min, k_max):
    return analytics.evaluate_forecasters(_daily, _zones, n_estimators=trees, k_min=k_min, k_max=k_max)


@st.cache_data(show_spinner="Training Random Forests and forecasting the next 7 days (may take ~1 minute)...")
def cached_forecast(_daily, key, trees, k_min, k_max):
    return analytics.forecast_future(_daily, n_estimators=trees, k_min=k_min, k_max=k_max)


def fmt(n, d=0):
    return f"{n:,.{d}f}"


def show_df(df, **kw):
    st.dataframe(df, width="stretch", hide_index=True, **kw)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("♻️ Waste Intelligence")
page = st.sidebar.radio("Navigate", PAGES, key="page")
st.sidebar.divider()

st.sidebar.subheader("Data source")
source_label = st.sidebar.radio(
    "Source", ["Synthetic (SIMULATED)", "Local classified-records store", "Upload daily CSV/Parquet"],
    key="source", label_visibility="collapsed")
source = {"Synthetic (SIMULATED)": "synthetic", "Local classified-records store": "records",
          "Upload daily CSV/Parquet": "upload"}[source_label]
upload = None
if source == "upload":
    upload = st.sidebar.file_uploader("Daily file", type=["csv", "parquet"], key="daily_upload")

with st.sidebar.expander("Clustering settings"):
    cl_level_label = st.radio("Cluster", ["Zones (one row per zone)", "Zone-days (paper, Eq. 4.5)"], key="cl_level")
    cl_level = "zone" if cl_level_label.startswith("Zones") else "zone_day"
    k_min, k_max = st.slider("K search range", 2, 10, (config.K_MIN, config.K_MAX), key="k_range")
    k_mode = st.radio("Choose K", ["Auto (best silhouette)", "Manual"], key="k_mode")
    k_manual = st.number_input("K (manual)", k_min, k_max, min(4, k_max), key="k_manual") if k_mode == "Manual" else None

with st.sidebar.expander("Forecast settings"):
    trees = st.number_input("Random Forest trees (T)", 10, 500, config.RF_TREES, step=10, key="trees")
    st.caption("Horizon fixed at 7 days; lags 1, 7, 14 days; one third of predictors per split.")

with st.sidebar.expander("Decision-rule thresholds"):
    margin = st.slider("Capacity trigger (x capacity)", 0.80, 1.20, config.TRIGGER_MARGIN, 0.01, key="margin",
                       help="1.00 = nominal rule. Lower values warn earlier (more recall, less precision).")
    cap_mult = st.slider("Capacity multiplier", 0.50, 2.00, config.CAPACITY_MULTIPLIER, 0.05, key="capmult",
                         help="Scales the stored zone capacity (theta_z).")
    rho_max = st.slider("Contamination limit rho_max", 0.00, 0.50, config.CONTAMINATION_MAX, 0.01, key="rhomax")
    win_days = st.slider("Contamination window (days)", 7, 90, config.CONTAMINATION_WINDOW_DAYS, key="window")
    dom_min = st.slider("Dominant recyclable share", 0.05, 0.50, config.DOMINANT_RECYCLABLE_MIN_SHARE, 0.01,
                        key="dommin", help="Implementation choice: the paper does not define 'dominant'.")
    use_cal = st.checkbox("Calibrate rho with classifier sensitivity/specificity", key="usecal")
    sens = spec = None
    if use_cal:
        sens = st.slider("Sensitivity", 0.50, 1.00, 0.95, 0.01, key="sens")
        spec = st.slider("Specificity", 0.50, 1.00, 0.95, 0.01, key="spec")
        st.caption("Values are YOUR assumptions; measure them on a labelled validation set.")

with st.sidebar.expander("Image classifier"):
    threshold = st.slider("Confidence threshold", 0.50, 0.99, config.CONFIDENCE_THRESHOLD, 0.01, key="thr")

st.sidebar.divider()
st.sidebar.caption("Local Pandas/scikit-learn engine. No distributed (Spark/HDFS) processing is running.")


# ---------------------------------------------------------------------------
# Data loading (with graceful empty state)
# ---------------------------------------------------------------------------
def load_data_or_stop(allow_missing: bool = False):
    try:
        stamp = _file_stamp(config.SYNTHETIC_DAILY_PATH) if source == "synthetic" else _file_stamp(config.CLASSIFIED_RECORDS_PATH)
        return cached_load(source, upload.getvalue() if upload else None, upload.name if upload else "", stamp)
    except data_io.DataError as exc:
        if allow_missing:
            return None, str(exc)
        st.title(page)
        st.error(str(exc))
        if source == "synthetic":
            st.info("Generate the SYNTHETIC demo dataset (24 zones x 730 days). It is not real city data.")
            if st.button("Generate synthetic data now", type="primary", key="gen_missing"):
                with st.spinner("Generating..."):
                    daily, zones = generate_data.generate_dataset()
                    generate_data.save_dataset(daily, zones)
                st.rerun()
        st.stop()


def sim_banner(bundle: data_io.DataBundle):
    if bundle.is_simulated:
        st.warning("**SIMULATED DATA** - every number, chart and table on this page comes from a synthetic "
                   "generator. It is not real city data and does not show real model performance.", icon="⚠️")
    else:
        st.info(f"Data source: {bundle.source_name}. Results are computed from this data and have not been "
                "independently verified.")
    for n in bundle.notes:
        st.caption(f"Note: {n}")


bundle_or_err = load_data_or_stop(allow_missing=(page == PAGES[6]))
if isinstance(bundle_or_err, tuple):
    bundle, data_error = None, bundle_or_err[1]
else:
    bundle, data_error = bundle_or_err, None
if page != PAGES[6] and bundle is None:
    st.stop()


def get_clustering(b):
    k = int(k_manual) if k_manual else None
    try:
        return cached_clustering(b.daily, b.zones, b.cache_key, cl_level, k, k_min, k_max)
    except ValueError as exc:
        st.error(f"Clustering failed: {exc}")
        st.stop()


def get_forecast(b):
    try:
        return cached_forecast(b.daily, b.cache_key, int(trees), k_min, k_max)
    except ValueError as exc:
        st.error(f"Forecast failed: {exc}")
        st.stop()


def get_evaluation(b):
    try:
        return cached_evaluation(b.daily, b.zones, b.cache_key, int(trees), k_min, k_max)
    except ValueError as exc:
        st.warning(f"Backtest not available: {exc}")
        return None


# ---------------------------------------------------------------------------
# Page 1: Overview
# ---------------------------------------------------------------------------
def page_overview(b):
    st.title("Overview" + b.tag)
    sim_banner(b)
    d = analytics.ensure_total(b.daily)
    n_days = d["date"].nunique()
    total = int(d["total_items"].sum())
    comp = analytics.composition(d, None)
    shares = {c.title(): float(comp[f"share_{c}"].iloc[0]) for c in config.CATEGORY_COLS}
    top = max(shares, key=shares.get)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Zones", d["zone_id"].nunique())
    c2.metric("Days", n_days, help=f"{d['date'].min().date()} to {d['date'].max().date()}")
    c3.metric("Total items" + b.tag, fmt(total))
    c4.metric("Avg items/day/zone" + b.tag, fmt(total / (n_days * d["zone_id"].nunique())))
    c5.metric("Largest category" + b.tag, f"{top} ({shares[top]:.0%})")

    daily_total = d.groupby("date")["total_items"].sum().reset_index()
    daily_total["7-day mean"] = daily_total["total_items"].rolling(7).mean()
    fig = go.Figure()
    fig.add_scatter(x=daily_total["date"], y=daily_total["total_items"], name="Daily total", opacity=0.4)
    fig.add_scatter(x=daily_total["date"], y=daily_total["7-day mean"], name="7-day mean")
    fig.update_layout(title=f"Total daily waste volume, all zones (items/day){b.tag}", template=TEMPLATE,
                      yaxis_title="items/day", height=380, legend_orientation="h")
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    with left:
        pie = pd.DataFrame({"Category": list(shares), "Share": list(shares.values())})
        f2 = px.pie(pie, names="Category", values="Share", hole=0.5, color="Category",
                    color_discrete_map=CAT_COLORS, title=f"Overall composition (share of items){b.tag}")
        f2.update_layout(template=TEMPLATE, height=340)
        st.plotly_chart(f2, width="stretch")
    with right:
        st.subheader("System status")
        svc = get_classifier()
        files = ic.check_files()
        ready = files["model_exists"] and files["class_names_exists"] and files["tensorflow_installed"]
        st.write("**Image classifier:** " + ("model files found (see page 2)" if ready else
                 "not available - model/class names/TensorFlow not set up yet. Everything else works."))
        st.write("**Analytics engine:** local Pandas + scikit-learn")
        st.write("**Spark:** " + spark_processor.spark_status()["message"])
        st.write("**HDFS:** " + hdfs_storage.hdfs_status()["message"])
        st.caption("Forecasts and recommendations are on pages 5 and 6.")


# ---------------------------------------------------------------------------
# Page 2: Classify image
# ---------------------------------------------------------------------------
def page_classify(b):
    st.title("Classify a waste image")
    st.caption("MobileNetV2 interface - uses YOUR trained model. Nothing is predicted unless a valid model is loaded.")
    svc = get_classifier()
    svc.threshold = threshold
    files = ic.check_files()
    ready = svc.load()
    if not ready:
        st.warning(f"**Classifier not ready:** {svc.load_message}")
        with st.expander("How to set up your trained model", expanded=True):
            st.code(ic.SETUP_INSTRUCTIONS, language="text")
            st.write(f"Expected model path: `{files['model_path']}`")
            st.write(f"Expected class names path: `{files['class_names_path']}`")
        st.caption("You can still upload an image below to test validation and preprocessing; "
                   "no category will be shown.")
    else:
        st.success(f"Model loaded (TensorFlow {svc.tf_version}). Classes: {', '.join(svc.class_names)}. "
                   f"Confidence threshold: {threshold:.0%}.")

    zone_ids = sorted(b.zones["zone_id"].unique()) if b is not None else []
    zone = st.selectbox("Zone for the record", zone_ids, key="cls_zone") if zone_ids else st.text_input("Zone", "Z01")
    tab_up, tab_cam = st.tabs(["Upload image", "Use camera"])
    with tab_up:
        up = st.file_uploader("Waste image (JPG, PNG, WEBP, BMP)", type=["jpg", "jpeg", "png", "webp", "bmp"], key="cls_upload")
    with tab_cam:
        cam = st.camera_input("Take a photo", key="cls_cam")
    src = up or cam
    if src is None:
        return
    data = src.getvalue()
    try:
        img = ic.validate_image(data)
    except ic.ImageValidationError as exc:
        st.error(f"Invalid image: {exc}")
        return
    col_img, col_res = st.columns([1, 1])
    with col_img:
        st.image(img, caption=f"Input ({img.size[0]}x{img.size[1]} px; resized to 224x224 for the model)", width="stretch")
    with col_res:
        res = svc.classify_bytes(data)
        if res.status == ic.Status.OK:
            st.success(f"**Predicted category: {res.label}**")
            st.metric("Confidence", f"{res.confidence:.1%}", help=f"Accepted: >= {threshold:.0%}")
            pf = pd.DataFrame({"Category": list(res.probabilities), "Probability": list(res.probabilities.values())})
            fig = px.bar(pf, x="Probability", y="Category", orientation="h", color="Category",
                         color_discrete_map=CAT_COLORS, range_x=[0, 1], title="Model output probabilities")
            fig.update_layout(template=TEMPLATE, showlegend=False, height=300)
            st.plotly_chart(fig, width="stretch")
            if st.button("Save record to local store", key="save_rec"):
                rid = data_io.append_record(res.label, res.confidence, zone)
                st.success(f"Saved record {rid[:8]}... (category, confidence, zone, timestamp only; the image is not stored).")
        elif res.status == ic.Status.LOW_CONFIDENCE:
            st.warning(res.message)
            st.write(f"Model's highest-scoring class: **{res.top_label_unaccepted}** at {res.confidence:.1%} "
                     "(NOT accepted as a result).")
            if st.button("Add to manual review queue", key="queue"):
                data_io.append_review_item(res.top_label_unaccepted, res.confidence, zone, "below confidence threshold")
                st.info(f"Added metadata to {config.REVIEW_QUEUE_PATH.name}.")
        elif res.status in (ic.Status.MODEL_MISSING, ic.Status.CLASS_NAMES_MISSING, ic.Status.CLASS_NAMES_INVALID,
                            ic.Status.TENSORFLOW_MISSING, ic.Status.MODEL_LOAD_ERROR):
            st.info("The image is valid and was preprocessed, but no prediction can be made: " + res.message)
        else:
            st.error(res.message)


# ---------------------------------------------------------------------------
# Page 3: Waste categories
# ---------------------------------------------------------------------------
def page_categories(b):
    st.title("Waste-category analysis" + b.tag)
    sim_banner(b)
    d = analytics.ensure_total(b.daily)
    zones_all = sorted(d["zone_id"].unique())
    c1, c2, c3 = st.columns([2, 2, 1])
    sel = c1.multiselect("Zones", zones_all, default=zones_all, key="cat_zones")
    rng = c2.date_input("Date range", (d["date"].min().date(), d["date"].max().date()), key="cat_dates")
    freq_label = c3.selectbox("Aggregate", ["Daily", "Weekly", "Monthly"], index=1, key="cat_freq")
    if not sel:
        st.warning("Select at least one zone.")
        return
    if not (isinstance(rng, tuple) and len(rng) == 2):
        st.info("Select both a start and an end date.")
        return
    f = d[d["zone_id"].isin(sel) & (d["date"] >= pd.Timestamp(rng[0])) & (d["date"] <= pd.Timestamp(rng[1]))]
    if f.empty:
        st.warning("No data for the selected filters.")
        return
    freq = {"Daily": "D", "Weekly": "W", "Monthly": "MS"}[freq_label]
    per = analytics.aggregate_period(f, freq).groupby("date")[config.CATEGORY_COLS].sum().reset_index()
    long = per.melt("date", var_name="Category", value_name="Items")
    long["Category"] = long["Category"].str.title()
    unit = {"D": "items/day", "W": "items/week", "MS": "items/month"}[freq]
    fig = px.area(long, x="date", y="Items", color="Category", color_discrete_map=CAT_COLORS,
                  title=f"Waste volume by category ({unit}, selected zones){b.tag}")
    fig.update_layout(template=TEMPLATE, yaxis_title=unit, height=400, legend_orientation="h")
    st.plotly_chart(fig, width="stretch")

    left, right = st.columns(2)
    comp = analytics.composition(f, "zone_id")
    cl = comp.melt("zone_id", value_vars=[f"share_{c}" for c in config.CATEGORY_COLS],
                   var_name="Category", value_name="Share")
    cl["Category"] = cl["Category"].str.replace("share_", "").str.title()
    f1 = px.bar(cl, x="zone_id", y="Share", color="Category", color_discrete_map=CAT_COLORS,
                title=f"Composition by zone (share of items){b.tag}")
    f1.update_layout(template=TEMPLATE, yaxis_tickformat=".0%", xaxis_title="Zone", height=380, legend_orientation="h")
    left.plotly_chart(f1, width="stretch")
    prof = analytics.weekday_profile(f)
    f2 = px.imshow(prof.T, aspect="auto", color_continuous_scale="Teal",
                   labels=dict(x="Weekday", y="Category", color="items/day"),
                   title=f"Weekly pattern: mean items/day (selected zones, per zone-day){b.tag}")
    f2.update_layout(template=TEMPLATE, height=380)
    right.plotly_chart(f2, width="stretch")

    st.subheader("Composition table" + b.tag)
    show = comp.copy()
    for c in config.CATEGORY_COLS:
        show[f"share_{c}"] = (show[f"share_{c}"] * 100).round(1)
    show = show.rename(columns={**{c: c.title() + " (items)" for c in config.CATEGORY_COLS},
                                **{f"share_{c}": c.title() + " (%)" for c in config.CATEGORY_COLS},
                                "total_items": "Total (items)"})
    show_df(show)
    st.download_button("Download composition CSV", show.to_csv(index=False), "composition.csv", "text/csv")


# ---------------------------------------------------------------------------
# Page 4: Clustering
# ---------------------------------------------------------------------------
def page_clustering(b):
    st.title("Zone clustering (K-Means)" + b.tag)
    sim_banner(b)
    st.caption("Features: total volume + six category shares, standardised (Eq. 4.5). "
               "K searched over the range in the sidebar; silhouette is computed only where valid.")
    res = get_clustering(b)
    c1, c2, c3 = st.columns(3)
    c1.metric("Chosen K", res.chosen_k)
    c2.metric("Best silhouette K", res.best_k if res.best_k else "n/a")
    c3.metric("Elbow K (heuristic)", res.elbow_k if res.elbow_k else "n/a")
    if res.best_k and res.elbow_k and res.best_k != res.elbow_k:
        st.info("Silhouette and elbow criteria disagree here. Use the manual K option in the sidebar to compare.")

    kt = res.k_table
    fig = go.Figure()
    fig.add_scatter(x=kt["k"], y=kt["silhouette"], name="Silhouette", yaxis="y", mode="lines+markers")
    fig.add_scatter(x=kt["k"], y=kt["inertia"], name="Inertia", yaxis="y2", mode="lines+markers", line=dict(dash="dash"))
    fig.update_layout(template=TEMPLATE, title=f"K search: silhouette (left) and inertia (right){b.tag}",
                      xaxis_title="K (number of clusters)", yaxis=dict(title="Silhouette score"),
                      yaxis2=dict(title="Inertia", overlaying="y", side="right"), height=380, legend_orientation="h")
    st.plotly_chart(fig, width="stretch")
    with st.expander("K-search table"):
        show_df(kt.round(4))

    st.subheader("Cluster centroids (original units)" + b.tag)
    cent = res.centroids.copy()
    for c in config.CATEGORY_COLS:
        cent[f"share_{c}"] = (cent[f"share_{c}"] * 100).round(1)
    cent["items_per_day"] = cent["items_per_day"].round(0)
    cent = cent.rename(columns={"items_per_day": "Items/day", "n_samples": "Samples",
                                **{f"share_{c}": c.title() + " %" for c in config.CATEGORY_COLS}})
    show_df(cent)

    za = res.zone_assignment.copy()
    zone_means = analytics.zone_day_features(b.daily).groupby("zone_id")[analytics.CLUSTER_FEATURES].mean().reset_index()
    za = za.merge(zone_means, on="zone_id")
    ycat = st.selectbox("Y axis: category share", config.CATEGORIES, index=4, key="cl_y")
    za["cluster_label"] = "Cluster " + za["cluster"].astype(str) + ": " + za["description"].astype(str)
    za["y"] = za[f"share_{ycat.lower()}"] * 100
    fig2 = px.scatter(za, x="total_items", y="y", color="cluster_label", text="zone_id",
                      title=f"Zones by mean daily volume and {ycat.lower()} share{b.tag}",
                      labels={"total_items": "Mean items/day", "y": f"{ycat} share (%)"})
    fig2.update_traces(textposition="top center", marker_size=12)
    fig2.update_layout(template=TEMPLATE, height=460, legend_orientation="h", legend_title="")
    st.plotly_chart(fig2, width="stretch")
    st.subheader("Zone assignments")
    out = res.zone_assignment.copy()
    out["purity"] = (out["purity"] * 100).round(1)
    show_df(out.rename(columns={"purity": "purity % (zone-days in majority cluster)"}))
    if res.ari_vs_simulation is not None:
        st.info(f"SIMULATION CHECK: adjusted Rand index between clusters and the generator's hidden zone "
                f"archetypes = {res.ari_vs_simulation:.3f}. This only exists because the data is synthetic "
                "and says nothing about real zones.")


# ---------------------------------------------------------------------------
# Page 5: Forecast
# ---------------------------------------------------------------------------
def page_forecast(b):
    st.title("7-day waste-volume forecast" + b.tag)
    sim_banner(b)
    st.caption("Random Forest (T trees, lags 1/7/14 days, calendar features, cluster label, previous category shares). "
               "Forecast values are MODEL OUTPUT, not observations.")
    fc = get_forecast(b)
    ev = get_evaluation(b)
    d = analytics.ensure_total(b.daily)
    origin = fc["origin_date"].iloc[0]
    zone_ids = ["All zones (sum)"] + sorted(d["zone_id"].unique())
    c1, c2 = st.columns([2, 1])
    zsel = c1.selectbox("Zone", zone_ids, key="fc_zone")
    hist_days = c2.slider("History shown (days)", 14, 180, 60, key="fc_hist")

    if zsel == zone_ids[0]:
        hist = d.groupby("date")["total_items"].sum().reset_index()
        pred = fc.groupby(["target_date", "horizon"])["forecast"].sum().reset_index()
        cap = float(b.zones["capacity_items_per_day"].sum()) * cap_mult
    else:
        hist = d[d["zone_id"] == zsel][["date", "total_items"]]
        pred = fc[fc["zone_id"] == zsel][["target_date", "horizon", "forecast"]]
        cap = float(b.zones.set_index("zone_id").loc[zsel, "capacity_items_per_day"]) * cap_mult
    hist = hist[hist["date"] > origin - pd.Timedelta(days=hist_days)]
    fig = go.Figure()
    fig.add_scatter(x=hist["date"], y=hist["total_items"], name="Historical (observed)" + b.tag, mode="lines")
    fig.add_scatter(x=pred["target_date"], y=pred["forecast"], name="FORECAST (model output)" + b.tag,
                    mode="lines+markers", line=dict(dash="dash", color="#DC2626"))
    fig.add_vline(x=origin, line_dash="dot", line_color="gray")
    fig.add_vrect(x0=origin, x1=pred["target_date"].max(), fillcolor="#FDE68A", opacity=0.25, line_width=0,
                  annotation_text="Forecast period", annotation_position="top left")
    fig.add_hline(y=cap, line_color="#B91C1C", line_dash="longdash", annotation_text="Capacity (theta)")
    fig.update_layout(template=TEMPLATE, title=f"Observed history vs 7-day forecast (items/day){b.tag}",
                      yaxis_title="items/day", height=430, legend_orientation="h")
    st.plotly_chart(fig, width="stretch")
    st.caption(f"Last observed day (forecast origin): {origin.date()}. Dotted line separates history from forecast.")
    ft = pred.copy()
    ft["target_date"] = ft["target_date"].dt.date
    show_df(ft.rename(columns={"target_date": "Forecast date", "horizon": "Days ahead",
                               "forecast": "Forecast (items/day)"}).round(0))

    st.divider()
    st.subheader("Backtest on a held-out test period" + b.tag)
    if ev is None:
        return
    st.write(f"Chronological split by target date: models are trained on targets up to **{ev.cutoff.date()}** "
             f"({ev.n_train:,} samples) and tested on the later period ({ev.n_test:,} samples). "
             f"Cluster labels were fitted on training rows only (K = {ev.cluster_k}). Horizon = {ev.horizon} days.")
    if b.is_simulated:
        st.warning("These are metrics on SIMULATED data. They are not real-world forecasting performance and "
                   "should not be compared with the paper's numbers as if they were the same experiment.")
    m = ev.metrics.copy()
    m = m.rename(columns={"method": "Method", "MAE": "MAE (items/day)", "RMSE": "RMSE (items/day)",
                          "MAPE_%": "MAPE (%)", "Bias": "Bias (items/day)",
                          "MAE_vs_persistence_%": "MAE vs persistence (% lower)",
                          "MAE_normal_days": "MAE normal days", "MAE_event_days": "MAE unannounced-event days"})
    show_df(m.round(3))
    figm = px.bar(ev.metrics, x="method", y="MAE", title=f"Mean absolute error at 7 days ahead (items/day){b.tag}")
    figm.update_layout(template=TEMPLATE, xaxis_title="", yaxis_title="MAE (items/day)", height=340)
    st.plotly_chart(figm, width="stretch")

    p = ev.predictions
    zt = st.selectbox("Zone for test-period plot", ["All zones (sum)"] + sorted(p["zone_id"].unique()), key="fc_test_zone")
    q = p.groupby("target_date")[["actual", "persistence", "ridge", "rf"]].sum().reset_index() if zt.startswith("All") \
        else p[p["zone_id"] == zt]
    ftest = go.Figure()
    for col, name in (("actual", "Actual"), ("rf", "Random Forest"), ("ridge", "Ridge"), ("persistence", "Persistence")):
        ftest.add_scatter(x=q["target_date"], y=q[col], name=name, mode="lines")
    ftest.update_layout(template=TEMPLATE, height=380, yaxis_title="items/day", legend_orientation="h",
                        title=f"Test period: 7-day-ahead predictions vs actual{b.tag}")
    st.plotly_chart(ftest, width="stretch")
    if ev.has_event_flag:
        st.caption("The simulator includes an unannounced 5-day festival in the test period (not in the holiday flag). "
                   "All models under-forecast it because no predictor carries that information.")
    left, right = st.columns(2)
    with left:
        st.markdown("**Random Forest predictor importance**")
        fi = px.bar(ev.importances.head(10).iloc[::-1], x="importance", y="predictor", orientation="h")
        fi.update_layout(template=TEMPLATE, height=340)
        st.plotly_chart(fi, width="stretch")
    with right:
        st.markdown("**Overflow warnings 7 days ahead** (forecast > trigger x capacity vs actual > capacity)")
        show_df(ev.overflow.round(3))
    st.caption("Limitation: Random Forests cannot extrapolate beyond the range of their training volumes, so "
               "sudden events and peaks are under-predicted. Lowering the capacity trigger trades missed overflows "
               "for unnecessary extra pickups.")


# ---------------------------------------------------------------------------
# Page 6: Capacity warnings and recommendations
# ---------------------------------------------------------------------------
def page_recommendations(b):
    st.title("Capacity warnings & recommendations" + b.tag)
    sim_banner(b)
    st.warning(rec.DISCLAIMER, icon="ℹ️")
    fc = get_forecast(b)
    cres = get_clustering(b)
    cal = (sens, spec) if use_cal else None
    try:
        table = rec.build_recommendations(fc, b.zones, b.daily, cres, margin, cap_mult, rho_max, win_days, dom_min, cal)
    except ValueError as exc:
        st.error(str(exc))
        return
    trig = table[table["triggered"]]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Capacity warnings", int(((table.rule_id == "R1-CAPACITY") & table.triggered).sum()))
    c2.metric("Contamination flags", int(((table.rule_id == "R2-CONTAMINATION") & table.triggered).sum()))
    c3.metric("Schedule suggestions", int(((table.rule_id == "R3-SCHEDULE") & table.triggered).sum()))
    c4.metric("Zones evaluated", table["zone_id"].nunique())
    if cres.level != "zone":
        st.caption("Rule 3 uses each zone's majority cluster from the zone-day clustering.")

    with st.expander("How rho (contamination) is calculated", expanded=False):
        st.markdown(
            "**rho(z) = items in recyclable-designated bins labelled non-recyclable / all items in recyclable-designated "
            f"bins**, summed over the trailing {win_days} days for zone *z* (columns `recyc_bin_nonrecyc_items` and "
            "`recyc_bin_items`). In the synthetic dataset these counts are generated; with a real classifier the "
            "numerator comes from classifier labels, which is biased upward by misclassification - optionally correct "
            "with the sensitivity/specificity option in the sidebar (Rogan-Gladen estimator).")

    rates = rec.contamination_rates(b.daily, win_days)
    if not rates.empty:
        rr = rates.copy()
        rr["Above limit"] = rr["rho"] > rho_max
        figr = px.bar(rr, x="zone_id", y="rho", color="Above limit",
                      color_discrete_map={True: "#DC2626", False: "#0F766E"},
                      title=f"Contamination rate rho by zone, trailing {win_days} days{b.tag}")
        figr.add_hline(y=rho_max, line_dash="dash", annotation_text=f"rho_max = {rho_max:g}")
        figr.update_layout(template=TEMPLATE, xaxis_title="Zone", yaxis_title="rho (share)", height=360)
        st.plotly_chart(figr, width="stretch")
    else:
        st.info("This data has no contamination columns, so Rule 2 cannot be evaluated.")

    kinds = st.multiselect("Rules shown", sorted(table["rule_id"].unique()), default=sorted(table["rule_id"].unique()), key="rec_rules")
    show_all = st.checkbox("Also show rules that were evaluated but NOT triggered", key="rec_all")
    view = table[table["rule_id"].isin(kinds)]
    if not show_all:
        view = view[view["triggered"]]
    if view.empty:
        st.info("No recommendations triggered with the current thresholds.")
    order = {"high": 0, "medium": 1, "info": 2, "none": 3}
    view = view.sort_values(["rule_id", "zone_id"], key=lambda s: s.map(order) if s.name == "severity" else s)
    for r in view.itertuples():
        icon = {"high": "🔴", "medium": "🟠", "info": "🔵", "none": "⚪"}[r.severity]
        with st.expander(f"{icon} {r.zone_id} - {r.rule_id}: {r.recommendation}"):
            st.markdown(f"**Rule:** {r.rule}")
            st.markdown("**Input values:**")
            st.json(r.inputs)
            st.markdown(f"**Reason:** {r.reason}")
    flat = table.copy()
    flat["inputs"] = flat["inputs"].apply(lambda x: json.dumps(x, default=str))
    st.download_button("Download all rule evaluations (CSV)", flat.to_csv(index=False), "rule_evaluations.csv", "text/csv")


# ---------------------------------------------------------------------------
# Page 7: Data source & status
# ---------------------------------------------------------------------------
def page_status(b, err):
    st.title("Data source & system status")
    st.subheader("Data")
    if b is None:
        st.error(err)
    else:
        sim_banner(b)
        c1, c2, c3 = st.columns(3)
        c1.metric("Source", b.source_name)
        c2.metric("Rows (zone-days)", fmt(len(b.daily)))
        c3.metric("Date range", f"{b.daily['date'].min().date()} to {b.daily['date'].max().date()}")
        with st.expander("Preview daily data"):
            show_df(b.daily.head(50))
        with st.expander("Zone table (capacity theta_z)"):
            show_df(b.zones)
    st.markdown("**Generate / regenerate SYNTHETIC data** (overwrites `data/synthetic_*.parquet`)")
    g1, g2, g3, g4 = st.columns(4)
    nz = g1.number_input("Zones", 4, 60, config.N_ZONES, key="gen_zones")
    nd = g2.number_input("Days", 120, 1500, config.N_DAYS, key="gen_days")
    sd = g3.number_input("Seed", 0, 10_000, config.RANDOM_SEED, key="gen_seed")
    csv = g4.checkbox("Also write CSV", key="gen_csv")
    if st.button("Generate synthetic data", key="gen_btn"):
        with st.spinner("Generating..."):
            daily, zones = generate_data.generate_dataset(int(nz), int(nd), int(sd))
            paths = generate_data.save_dataset(daily, zones, write_csv=csv)
        st.success(f"[SIMULATED] Wrote {len(daily):,} rows to {paths['daily'].name}. Reloading...")
        st.cache_data.clear()
        st.rerun()

    st.subheader("Required daily schema (for uploads)")
    st.code("date, zone_id, paper, plastic, glass, metal, organic, other  "
            "[optional: recyc_bin_items, recyc_bin_nonrecyc_items]\n"
            "one row per zone per day, no gaps, non-negative integer counts", language="text")

    st.subheader("Image classifier")
    files = ic.check_files()
    show_df(pd.DataFrame([
        {"Item": "Model file", "Path": files["model_path"], "Status": "found" if files["model_exists"] else "MISSING"},
        {"Item": "Class names", "Path": files["class_names_path"],
         "Status": ("valid" if files["class_names_valid"] else f"INVALID: {files['class_names_error']}")
         if files["class_names_exists"] else "MISSING"},
        {"Item": "TensorFlow", "Path": "(optional package)", "Status": "installed" if files["tensorflow_installed"] else "not installed"},
    ]))
    st.caption("No model is bundled. Train your own and follow models/README.md.")

    st.subheader("Optional integrations")
    sp, hd = spark_processor.spark_status(), hdfs_storage.hdfs_status()
    show_df(pd.DataFrame([
        {"Integration": "Apache Spark", "Enabled": sp["available"], "Detail": sp["message"]},
        {"Integration": "HDFS (WebHDFS)", "Enabled": hd["available"], "Detail": hd["message"]},
    ]))
    st.info("Processing mode: **local Pandas / scikit-learn / Parquet**. No distributed processing is running, "
            "even if the optional packages above are installed - the dashboard does not call them automatically.")

    st.subheader("Documented features not yet implemented (PENDING)")
    show_df(pd.DataFrame([
        {"Feature (paper/report)": "Spark MLlib K-Means / Random Forest at cluster scale", "Status": "PENDING (scikit-learn used)"},
        {"Feature (paper/report)": "Spark aggregation wired into the dashboard", "Status": "PENDING (module provided, untested on a real Spark install)"},
        {"Feature (paper/report)": "HDFS append of classified records (replication 3)", "Status": "PENDING (module provided, untested on a real cluster)"},
        {"Feature (paper/report)": "Trained MobileNetV2 model", "Status": "PENDING - you train it"},
        {"Feature (paper/report)": "Validation on real image streams / real city data", "Status": "PENDING (future work)"},
        {"Feature (paper/report)": "IoT smart-dustbin fill-level integration", "Status": "PENDING (future work)"},
        {"Feature (paper/report)": "Streaming ingestion, YOLO multi-object detection, route optimisation", "Status": "PENDING (future work)"},
        {"Feature (paper/report)": "Quantile forecasts / planned-event calendar / manual override", "Status": "PENDING (suggested remedies)"},
    ]))


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------
try:
    if page == PAGES[0]:
        page_overview(bundle)
    elif page == PAGES[1]:
        page_classify(bundle)
    elif page == PAGES[2]:
        page_categories(bundle)
    elif page == PAGES[3]:
        page_clustering(bundle)
    elif page == PAGES[4]:
        page_forecast(bundle)
    elif page == PAGES[5]:
        page_recommendations(bundle)
    else:
        page_status(bundle, data_error)
except Exception as exc:  # last-resort guard so the user sees a message, not a traceback wall
    if type(exc).__name__ in ("StopException", "RerunException"):
        raise
    st.error(f"Unexpected error: {type(exc).__name__}: {exc}")
    st.exception(exc)
