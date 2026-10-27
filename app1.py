"""
Digital Wellbeing & Mental Health Dashboard
-------------------------------------------
Interactive Streamlit app: live predictions, what-if analysis, an interactive
dataset explorer, model insights and batch scoring.

Run locally:   streamlit run app.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Wellbeing Assessment Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent

# =========================================================
# CONSTANTS
# =========================================================
BAND_ORDER = ["At-risk", "Moderate", "Good"]
BAND_COLORS = {"At-risk": "#dc3545", "Moderate": "#ffc107", "Good": "#28a745"}
NIGHT_ORDER = ["Never", "Sometimes", "Often", "Every night"]
DETOX_ORDER = ["No", "Yes, failed", "Yes, succeeded"]
SUPPORT_ORDER = ["No", "Considering it", "Yes"]
AGE_LABELS = ["≤17", "18-24", "25-34", "35-44", "45-54", "55-64", "65+"]
CAT_ORDERS = {
    "wellbeing_band": BAND_ORDER,
    "night_time_use": NIGHT_ORDER,
    "attempted_digital_detox": DETOX_ORDER,
    "seeks_mental_health_support": SUPPORT_ORDER,
    "age_group": AGE_LABELS,
}

# Numeric inputs: key -> (label, min, max, step, widget kind)
NUM_SPECS = {
    "age": ("Age", 10, 80, 1, "slider"),
    "platforms_used_count": ("Platforms used", 1, 10, 1, "slider"),
    "daily_screen_hours": ("Daily screen hours", 0.0, 16.0, 0.1, "slider"),
    "daily_notifications": ("Daily notifications", 0, 1000, 10, "number"),
    "minutes_to_first_check_after_waking": ("Minutes to check phone after waking", 0, 120, 1, "number"),
    "avg_sleep_hours": ("Average sleep (hours)", 3.0, 12.0, 0.1, "slider"),
    "physical_activity_days_per_week": ("Physical activity (days/week)", 0, 7, 1, "slider"),
    "anxiety_score_0to27": ("Anxiety score (0-27)", 0, 27, 1, "slider"),
    "low_mood_score_0to27": ("Low mood score (0-27)", 0, 27, 1, "slider"),
    "life_satisfaction_1to10": ("Life satisfaction (1-10)", 1, 10, 1, "slider"),
    "loneliness_1to10": ("Loneliness (1-10)", 1, 10, 1, "slider"),
    "self_esteem_1to10": ("Self-esteem (1-10)", 1, 10, 1, "slider"),
    "fomo_1to10": ("FOMO (1-10)", 1, 10, 1, "slider"),
    "social_comparison_1to10": ("Social comparison (1-10)", 1, 10, 1, "slider"),
}

# Categorical inputs: key -> (label, options)
CAT_SPECS = {
    "gender": ("Gender", ["Female", "Male", "Non-binary", "Prefer not to say"]),
    "occupation": ("Occupation", ["Student", "Full-time employed", "Part-time employed",
                                  "Self-employed", "Unemployed", "Retired"]),
    "region": ("Region", ["Africa", "Asia", "Europe", "Latin America", "North America", "Oceania"]),
    "most_used_platform": ("Most used platform", ["TikTok", "Instagram", "YouTube", "Facebook",
                                                  "LinkedIn", "Snapchat", "Reddit", "X/Twitter"]),
    "primary_purpose": ("Primary purpose", ["Entertainment", "Connection with friends",
                                            "Passing time/boredom", "News/information",
                                            "Work/career", "Content creation"]),
    "night_time_use": ("Night-time screen use", NIGHT_ORDER),
    "uses_screen_time_limits": ("Uses screen-time limits?", ["No", "Yes"]),
    "attempted_digital_detox": ("Attempted digital detox?", DETOX_ORDER),
    "seeks_mental_health_support": ("Seeks mental-health support?", SUPPORT_ORDER),
}

# Sidebar groups (order = display order)
GROUPS = {
    "1️⃣ Demographics & platform": ["age", "gender", "occupation", "region",
                                   "most_used_platform", "primary_purpose"],
    "2️⃣ Digital habits & sleep": ["daily_screen_hours", "platforms_used_count", "daily_notifications",
                                  "minutes_to_first_check_after_waking", "night_time_use",
                                  "avg_sleep_hours", "physical_activity_days_per_week"],
    "3️⃣ Psychological indicators": ["anxiety_score_0to27", "low_mood_score_0to27",
                                    "life_satisfaction_1to10", "loneliness_1to10", "self_esteem_1to10",
                                    "fomo_1to10", "social_comparison_1to10", "uses_screen_time_limits",
                                    "attempted_digital_detox", "seeks_mental_health_support"],
}

# Column order the model was trained on
FEATURES = [
    "age", "gender", "occupation", "region", "most_used_platform", "platforms_used_count",
    "daily_screen_hours", "daily_notifications", "night_time_use",
    "minutes_to_first_check_after_waking", "primary_purpose", "avg_sleep_hours",
    "anxiety_score_0to27", "low_mood_score_0to27", "life_satisfaction_1to10", "loneliness_1to10",
    "self_esteem_1to10", "fomo_1to10", "social_comparison_1to10",
    "physical_activity_days_per_week", "uses_screen_time_limits", "attempted_digital_detox",
    "seeks_mental_health_support",
]

DEFAULTS = {
    "age": 25, "gender": "Female", "occupation": "Student", "region": "Asia",
    "most_used_platform": "Instagram", "primary_purpose": "Entertainment",
    "daily_screen_hours": 4.5, "platforms_used_count": 4, "daily_notifications": 120,
    "minutes_to_first_check_after_waking": 15, "night_time_use": "Sometimes",
    "avg_sleep_hours": 7.0, "physical_activity_days_per_week": 3,
    "anxiety_score_0to27": 8, "low_mood_score_0to27": 6, "life_satisfaction_1to10": 7,
    "loneliness_1to10": 4, "self_esteem_1to10": 6, "fomo_1to10": 5, "social_comparison_1to10": 5,
    "uses_screen_time_limits": "No", "attempted_digital_detox": "No",
    "seeks_mental_health_support": "No",
}

PRESETS = {
    "Healthy routine": {
        "daily_screen_hours": 2.0, "platforms_used_count": 2, "daily_notifications": 40,
        "minutes_to_first_check_after_waking": 45, "night_time_use": "Never", "avg_sleep_hours": 8.0,
        "physical_activity_days_per_week": 5, "anxiety_score_0to27": 3, "low_mood_score_0to27": 2,
        "life_satisfaction_1to10": 9, "loneliness_1to10": 2, "self_esteem_1to10": 8, "fomo_1to10": 2,
        "social_comparison_1to10": 3, "uses_screen_time_limits": "Yes",
        "attempted_digital_detox": "Yes, succeeded", "seeks_mental_health_support": "No",
    },
    "Average user": dict(DEFAULTS),
    "High strain": {
        "daily_screen_hours": 9.0, "platforms_used_count": 7, "daily_notifications": 350,
        "minutes_to_first_check_after_waking": 2, "night_time_use": "Every night",
        "avg_sleep_hours": 5.0, "physical_activity_days_per_week": 0, "anxiety_score_0to27": 20,
        "low_mood_score_0to27": 14, "life_satisfaction_1to10": 3, "loneliness_1to10": 8,
        "self_esteem_1to10": 3, "fomo_1to10": 8, "social_comparison_1to10": 8,
        "uses_screen_time_limits": "No", "attempted_digital_detox": "Yes, failed",
        "seeks_mental_health_support": "Considering it",
    },
}

# Metrics shown on the radar chart: label -> (column, scale maximum)
RADAR = {
    "Anxiety": ("anxiety_score_0to27", 27),
    "Low mood": ("low_mood_score_0to27", 27),
    "Loneliness": ("loneliness_1to10", 10),
    "FOMO": ("fomo_1to10", 10),
    "Social comparison": ("social_comparison_1to10", 10),
    "Self-esteem": ("self_esteem_1to10", 10),
    "Life satisfaction": ("life_satisfaction_1to10", 10),
    "Sleep (of 12h)": ("avg_sleep_hours", 12),
    "Screen (of 16h)": ("daily_screen_hours", 16),
}


def pretty(col: str) -> str:
    """Human friendly column name."""
    return (col.replace("_0to27", " (0-27)").replace("_1to10", " (1-10)")
            .replace("_", " ").strip().capitalize())


# =========================================================
# LOADERS
# =========================================================
@st.cache_resource(show_spinner="Loading model…")
def load_model_and_encoder():
    model_path = BASE_DIR / "wellbeing_model.joblib"
    enc_path = BASE_DIR / "label_encoder.joblib"
    if model_path.exists() and enc_path.exists():
        return joblib.load(model_path), joblib.load(enc_path)
    return None, None


@st.cache_data(show_spinner="Loading dataset…")
def load_dataset():
    csv_path = BASE_DIR / "social_media_screentime_mental_health_2026.csv"
    if not csv_path.exists():
        return None
    d = pd.read_csv(csv_path)
    d["gender"] = d["gender"].fillna("Unknown")
    d["age_group"] = pd.cut(
        d["age"], bins=[0, 17, 24, 34, 44, 54, 64, 120], labels=AGE_LABELS
    ).astype(str)
    return d


model, le = load_model_and_encoder()
df = load_dataset()


# =========================================================
# HELPERS
# =========================================================
def apply_preset(name: str):
    for k, v in PRESETS[name].items():
        st.session_state[k] = v


def load_random_profile():
    if df is None:
        return
    r = df.sample(1).iloc[0]
    for k in FEATURES:
        v = r[k]
        if k == "gender" and v == "Unknown":
            v = "Prefer not to say"
        if k == "avg_sleep_hours" and pd.isna(v):
            v = DEFAULTS[k]
        if k in NUM_SPECS:
            kind_float = isinstance(NUM_SPECS[k][1], float)
            v = round(float(v), 1) if kind_float else int(v)
        st.session_state[k] = v


def render_widget(key: str):
    if key in NUM_SPECS:
        label, lo, hi, step, kind = NUM_SPECS[key]
        if kind == "slider":
            st.slider(label, min_value=lo, max_value=hi, step=step, key=key)
        else:
            st.number_input(label, min_value=lo, max_value=hi, step=step, key=key)
    else:
        label, options = CAT_SPECS[key]
        st.selectbox(label, options, key=key)


def predict_proba_df(X: pd.DataFrame) -> pd.DataFrame:
    """Return class probabilities with readable band names, ordered At-risk/Moderate/Good."""
    probs = model.predict_proba(X)
    names = le.inverse_transform(model.classes_)
    out = pd.DataFrame(probs, columns=names, index=X.index)
    return out[[b for b in BAND_ORDER if b in out.columns]]


def show(fig, key: str, height: int = 380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch", key=key)


def get_importances():
    """Random-forest importances aggregated back to the original input columns."""
    try:
        pre = model.named_steps["preprocessor"]
        clf = model.named_steps["classifier"]
        names = pre.get_feature_names_out()
        imps = clf.feature_importances_
        by_len = sorted(FEATURES, key=len, reverse=True)
        rows = []
        for n, v in zip(names, imps):
            raw = n.split("__", 1)[-1]
            base = next((f for f in by_len if raw == f or raw.startswith(f + "_")), raw)
            rows.append((base, v))
        out = pd.DataFrame(rows, columns=["feature", "importance"])
        return out.groupby("feature", as_index=False)["importance"].sum().sort_values(
            "importance", ascending=False)
    except Exception:
        return None


def build_tips(v: dict) -> list:
    tips = []
    if v["daily_screen_hours"] >= 6:
        tips.append("📱 Screen time is high (6h+). Trying app timers or phone-free blocks could help.")
    if v["avg_sleep_hours"] < 7:
        tips.append("😴 Sleep is under 7 hours. Most adults do best with 7-9 hours.")
    if v["night_time_use"] in ("Often", "Every night"):
        tips.append("🌙 Frequent night-time phone use is linked to poorer sleep. Consider a wind-down routine.")
    if v["minutes_to_first_check_after_waking"] <= 5:
        tips.append("⏰ Checking the phone almost immediately after waking can set a reactive tone for the day.")
    if v["daily_notifications"] >= 200:
        tips.append("🔔 200+ notifications a day is a lot. Muting non-essential apps can reduce interruptions.")
    if v["physical_activity_days_per_week"] <= 1:
        tips.append("🏃 Very little physical activity. Even a few short walks a week can help mood.")
    if v["anxiety_score_0to27"] >= 15 or v["low_mood_score_0to27"] >= 15:
        tips.append("💬 Scores are in a high range. Talking to a trusted person or a qualified professional is a good step.")
    if not tips:
        tips.append("✅ No obvious red flags in these inputs. Keep up your current habits.")
    return tips


def radar_chart(user_vals: dict, band_means: pd.DataFrame | None):
    cats = list(RADAR.keys())
    fig = go.Figure()
    if band_means is not None:
        for b in BAND_ORDER:
            if b in band_means.index:
                r = [float(band_means.loc[b, col]) / mx for col, mx in RADAR.values()]
                fig.add_trace(go.Scatterpolar(
                    r=r + [r[0]], theta=cats + [cats[0]], name=f"{b} average",
                    line=dict(color=BAND_COLORS[b], dash="dot"), opacity=0.8))
    r = [float(user_vals[col]) / mx for col, mx in RADAR.values()]
    fig.add_trace(go.Scatterpolar(
        r=r + [r[0]], theta=cats + [cats[0]], name="You", fill="toself",
        line=dict(color="#4c78ff", width=3), opacity=0.55))
    fig.update_layout(
        title="Your profile vs. band averages (click legend items to toggle)",
        polar=dict(radialaxis=dict(range=[0, 1], tickformat=".0%")),
        legend=dict(orientation="h", y=-0.1),
    )
    return fig


# =========================================================
# SIDEBAR - ALL INPUTS (live, no submit button needed)
# =========================================================
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)

with st.sidebar:
    st.title("🎛️ Input Controls")
    st.caption("The prediction updates **live** as you change values.")

    st.markdown("**⚡ Quick profiles**")
    pc1, pc2, pc3 = st.columns(3)
    pc1.button("🌿 Healthy", on_click=apply_preset, args=("Healthy routine",), width="stretch")
    pc2.button("😐 Average", on_click=apply_preset, args=("Average user",), width="stretch")
    pc3.button("🔥 Strain", on_click=apply_preset, args=("High strain",), width="stretch")
    st.button("🎲 Load a random person from the dataset", on_click=load_random_profile,
              width="stretch", disabled=df is None)
    st.divider()

    for group_title, keys in GROUPS.items():
        with st.expander(group_title, expanded=(group_title.startswith("1"))):
            for k in keys:
                render_widget(k)

inputs = {k: st.session_state[k] for k in FEATURES}

# =========================================================
# MAIN
# =========================================================
st.title("🧠 Digital Wellbeing & Mental Health Dashboard")
st.caption("Explore how social-media habits, sleep and psychological indicators relate to wellbeing. "
           "Educational demo only. Not a medical tool.")

if model is None or le is None:
    st.error("Model files (`wellbeing_model.joblib`, `label_encoder.joblib`) were not found next to `app.py`.")
    st.stop()

tab_pred, tab_whatif, tab_eda, tab_model, tab_batch = st.tabs([
    "🔮 Live Prediction", "🧪 What-If Lab", "📊 Data Explorer", "🧬 Model Insights", "📁 Batch Processing",
])

# ---------------------------------------------------------
# TAB 1 - LIVE PREDICTION
# ---------------------------------------------------------
row = pd.DataFrame([inputs])[FEATURES]
probs = predict_proba_df(row).iloc[0]
pred_label = probs.idxmax()
confidence = float(probs.max())
card_color = BAND_COLORS.get(pred_label, "#6c757d")

with tab_pred:
    left, mid, right = st.columns([1.1, 1.2, 1.5])

    with left:
        st.markdown(
            f"""
            <div style="background-color:{card_color}20; border-left:6px solid {card_color};
                        padding:22px; border-radius:8px;">
                <span style="font-size:14px; color:gray;">PREDICTED STATUS</span>
                <h2 style="color:{card_color}; margin:5px 0 0 0;">{pred_label}</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("#### Quick comparison with the dataset")
        if df is not None:
            def pct(col):
                return float((df[col] < inputs[col]).mean() * 100)

            st.metric("Screen time", f"{inputs['daily_screen_hours']:.1f} h",
                      f"{inputs['daily_screen_hours'] - df['daily_screen_hours'].mean():+.1f} h vs avg",
                      delta_color="inverse")
            st.metric("Sleep", f"{inputs['avg_sleep_hours']:.1f} h",
                      f"{inputs['avg_sleep_hours'] - df['avg_sleep_hours'].mean():+.1f} h vs avg")
            st.metric("Anxiety", f"{inputs['anxiety_score_0to27']}/27",
                      f"{inputs['anxiety_score_0to27'] - df['anxiety_score_0to27'].mean():+.1f} vs avg",
                      delta_color="inverse")
            st.caption(f"Higher screen time than {pct('daily_screen_hours'):.0f}% of participants · "
                       f"higher anxiety than {pct('anxiety_score_0to27'):.0f}%.")

    with mid:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=confidence * 100,
            number={"suffix": "%"},
            title={"text": "Model confidence"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": card_color},
                "steps": [
                    {"range": [0, 50], "color": "rgba(150,150,150,0.15)"},
                    {"range": [50, 80], "color": "rgba(150,150,150,0.30)"},
                    {"range": [80, 100], "color": "rgba(150,150,150,0.45)"},
                ],
            },
        ))
        show(gauge, "gauge", height=260)

        prob_df = probs.reset_index()
        prob_df.columns = ["Wellbeing band", "Probability"]
        fig_p = px.bar(prob_df, x="Wellbeing band", y="Probability", color="Wellbeing band",
                       color_discrete_map=BAND_COLORS, category_orders={"Wellbeing band": BAND_ORDER},
                       text_auto=".1%", range_y=[0, 1], title="Probability breakdown")
        fig_p.update_layout(showlegend=False)
        fig_p.update_yaxes(tickformat=".0%")
        show(fig_p, "probs", height=260)

    with right:
        band_means = None
        if df is not None:
            band_means = df.groupby("wellbeing_band")[[c for c, _ in RADAR.values()]].mean()
        show(radar_chart(inputs, band_means), "radar", height=540)

    st.markdown("#### 💡 Observations from your inputs")
    for tip in build_tips(inputs):
        st.write(f"- {tip}")

    export = {**inputs, "predicted_band": pred_label,
              **{f"prob_{b}": round(float(p), 4) for b, p in probs.items()}}
    st.download_button("📥 Download this assessment (CSV)",
                       data=pd.DataFrame([export]).to_csv(index=False).encode("utf-8"),
                       file_name="wellbeing_assessment.csv", mime="text/csv")

# ---------------------------------------------------------
# TAB 2 - WHAT-IF LAB
# ---------------------------------------------------------
with tab_whatif:
    st.subheader("What happens if one thing changes?")
    st.caption("Everything else stays as set in the sidebar. Pick one input and see how the predicted "
               "probabilities respond across its whole range.")

    options = list(NUM_SPECS.keys()) + list(CAT_SPECS.keys())
    default_idx = options.index("daily_screen_hours")
    feat = st.selectbox("Input to vary", options, index=default_idx, format_func=pretty, key="whatif_feat")

    if feat in NUM_SPECS:
        _, lo, hi, step, _ = NUM_SPECS[feat]
        # Use the dataset's observed range when available so the sweep stays realistic
        if df is not None and feat in df.columns:
            lo, hi = type(lo)(df[feat].min()), type(hi)(df[feat].max())
        grid = np.unique(np.linspace(lo, hi, 40 if isinstance(step, float) else min(40, int((hi - lo) / step) + 1)))
        if not isinstance(step, float):
            grid = np.unique(np.round(grid).astype(int))
        sweep = pd.concat([row] * len(grid), ignore_index=True)
        sweep[feat] = grid
        sw_probs = predict_proba_df(sweep)
        sw_probs.insert(0, "x_value", grid)
        fig_w = px.line(sw_probs, x="x_value", y=[b for b in BAND_ORDER if b in sw_probs.columns],
                        color_discrete_map=BAND_COLORS, markers=True,
                        labels={"x_value": pretty(feat), "value": "Probability", "variable": "Band"},
                        title=f"Predicted probability vs. {pretty(feat).lower()}")
        fig_w.add_vline(x=inputs[feat], line_dash="dash", line_color="gray",
                        annotation_text="current value", annotation_position="top")
        fig_w.update_yaxes(range=[0, 1], tickformat=".0%")
        show(fig_w, "whatif_line", height=440)
    else:
        cats = CAT_SPECS[feat][1]
        sweep = pd.concat([row] * len(cats), ignore_index=True)
        sweep[feat] = cats
        sw_probs = predict_proba_df(sweep)
        sw_probs.insert(0, "option", cats)
        long = sw_probs.melt(id_vars="option", var_name="Band", value_name="Probability")
        fig_w = px.bar(long, x="option", y="Probability", color="Band", barmode="group",
                       color_discrete_map=BAND_COLORS, category_orders={"Band": BAND_ORDER},
                       labels={"option": pretty(feat)}, text_auto=".0%",
                       title=f"Predicted probability by {pretty(feat).lower()}")
        fig_w.update_yaxes(range=[0, 1], tickformat=".0%")
        show(fig_w, "whatif_bar", height=440)

    st.divider()
    st.subheader("Two-input heatmap")
    st.caption("See how two inputs interact: colour shows the probability of the chosen band.")
    num_opts = list(NUM_SPECS.keys())
    h1, h2, h3 = st.columns(3)
    fx = h1.selectbox("X axis", num_opts, index=num_opts.index("anxiety_score_0to27"),
                      format_func=pretty, key="hm_x")
    fy = h2.selectbox("Y axis", num_opts, index=num_opts.index("low_mood_score_0to27"),
                      format_func=pretty, key="hm_y")
    target_band = h3.selectbox("Band to map", BAND_ORDER, index=0, key="hm_band")

    if fx == fy:
        st.info("Choose two different inputs.")
    else:
        def axis_grid(k, n=14):
            _, lo, hi, step, _ = NUM_SPECS[k]
            if df is not None and k in df.columns:
                lo, hi = float(df[k].min()), float(df[k].max())
            g = np.linspace(lo, hi, n)
            return np.unique(np.round(g).astype(int)) if not isinstance(step, float) else np.round(g, 1)

        gx, gy = axis_grid(fx), axis_grid(fy)
        mesh = pd.concat([row] * (len(gx) * len(gy)), ignore_index=True)
        xs, ys = np.meshgrid(gx, gy)
        mesh[fx] = xs.ravel()
        mesh[fy] = ys.ravel()
        z = predict_proba_df(mesh)[target_band].to_numpy().reshape(len(gy), len(gx))
        fig_hm = go.Figure(go.Heatmap(z=z, x=gx, y=gy, colorscale="RdYlGn_r" if target_band == "At-risk" else "Viridis",
                                      zmin=0, zmax=1, colorbar=dict(title="P(" + target_band + ")")))
        fig_hm.add_trace(go.Scatter(x=[inputs[fx]], y=[inputs[fy]], mode="markers", name="You",
                                    marker=dict(size=14, color="white", line=dict(color="black", width=2))))
        fig_hm.update_layout(title=f"P({target_band}) across {pretty(fx)} × {pretty(fy)}",
                             xaxis_title=pretty(fx), yaxis_title=pretty(fy))
        show(fig_hm, "whatif_heat", height=480)

# ---------------------------------------------------------
# TAB 3 - DATA EXPLORER
# ---------------------------------------------------------
with tab_eda:
    if df is None:
        st.warning("`social_media_screentime_mental_health_2026.csv` was not found next to `app.py`.")
    else:
        with st.expander("🔎 Filters (they apply to every chart in this tab)", expanded=True):
            f1, f2, f3 = st.columns(3)
            sel_band = f1.multiselect("Wellbeing band", BAND_ORDER, default=BAND_ORDER)
            sel_region = f2.multiselect("Region", sorted(df["region"].unique()),
                                        default=sorted(df["region"].unique()))
            sel_occ = f3.multiselect("Occupation", sorted(df["occupation"].unique()),
                                     default=sorted(df["occupation"].unique()))
            f4, f5 = st.columns([2, 1])
            sel_plat = f4.multiselect("Most used platform", sorted(df["most_used_platform"].unique()),
                                      default=sorted(df["most_used_platform"].unique()))
            age_rng = f5.slider("Age range", int(df["age"].min()), int(df["age"].max()),
                                (int(df["age"].min()), int(df["age"].max())))

        fdf = df[
            df["wellbeing_band"].isin(sel_band)
            & df["region"].isin(sel_region)
            & df["occupation"].isin(sel_occ)
            & df["most_used_platform"].isin(sel_plat)
            & df["age"].between(*age_rng)
        ]

        if fdf.empty:
            st.warning("No participants match these filters. Widen the selection.")
        else:
            num_cols = [c for c in fdf.select_dtypes("number").columns]
            cat_cols = ["wellbeing_band", "gender", "occupation", "region", "most_used_platform",
                        "night_time_use", "primary_purpose", "uses_screen_time_limits",
                        "attempted_digital_detox", "seeks_mental_health_support", "age_group"]

            k1, k2, k3, k4, k5 = st.columns(5)
            k1.metric("Participants", f"{len(fdf):,}", f"{len(fdf) - len(df):,} vs all" if len(fdf) != len(df) else None)
            k2.metric("Avg screen time", f"{fdf['daily_screen_hours'].mean():.1f} h",
                      f"{fdf['daily_screen_hours'].mean() - df['daily_screen_hours'].mean():+.2f}", delta_color="inverse")
            k3.metric("Avg sleep", f"{fdf['avg_sleep_hours'].mean():.1f} h",
                      f"{fdf['avg_sleep_hours'].mean() - df['avg_sleep_hours'].mean():+.2f}")
            k4.metric("Avg anxiety", f"{fdf['anxiety_score_0to27'].mean():.1f}/27",
                      f"{fdf['anxiety_score_0to27'].mean() - df['anxiety_score_0to27'].mean():+.2f}", delta_color="inverse")
            k5.metric("At-risk share", f"{(fdf['wellbeing_band'] == 'At-risk').mean() * 100:.1f}%",
                      f"{((fdf['wellbeing_band'] == 'At-risk').mean() - (df['wellbeing_band'] == 'At-risk').mean()) * 100:+.1f} pp",
                      delta_color="inverse")

            t_over, t_rel, t_hab, t_corr, t_build, t_data = st.tabs(
                ["Overview", "Relationships", "Habits & sleep", "Correlations", "🛠️ Chart builder", "Data table"])

            # ---- Overview
            with t_over:
                c1, c2 = st.columns(2)
                with c1:
                    counts = fdf["wellbeing_band"].value_counts().reset_index()
                    counts.columns = ["wellbeing_band", "count"]
                    fig = px.pie(counts, names="wellbeing_band", values="count", hole=0.45,
                                 color="wellbeing_band", color_discrete_map=BAND_COLORS,
                                 title="Wellbeing band breakdown")
                    show(fig, "ov_pie")
                with c2:
                    fig = px.histogram(fdf, x="age", color="wellbeing_band", nbins=30, marginal="box",
                                       color_discrete_map=BAND_COLORS, category_orders=CAT_ORDERS,
                                       title="Age distribution by band", barmode="overlay", opacity=0.65)
                    show(fig, "ov_age")

                c3, c4 = st.columns(2)
                with c3:
                    by = st.selectbox("Break the bands down by", [c for c in cat_cols if c != "wellbeing_band"],
                                      index=2, format_func=pretty, key="ov_by")
                    fig = px.histogram(fdf, x=by, color="wellbeing_band", barnorm="percent",
                                       color_discrete_map=BAND_COLORS, category_orders=CAT_ORDERS,
                                       text_auto=".0f", title=f"Band mix (%) by {pretty(by).lower()}")
                    fig.update_yaxes(title="% of group")
                    show(fig, "ov_stack")
                with c4:
                    st.markdown("&nbsp;")
                    fig = px.sunburst(fdf, path=["wellbeing_band", "region", "occupation"],
                                      color="wellbeing_band", color_discrete_map=BAND_COLORS,
                                      title="Click to drill down: band → region → occupation")
                    show(fig, "ov_sun", height=420)

            # ---- Relationships
            with t_rel:
                st.caption("Choose any variables. Hover for details, drag to zoom, double-click to reset.")
                s1, s2, s3, s4 = st.columns(4)
                sx = s1.selectbox("X", num_cols, index=num_cols.index("anxiety_score_0to27"),
                                  format_func=pretty, key="sc_x")
                sy = s2.selectbox("Y", num_cols, index=num_cols.index("low_mood_score_0to27"),
                                  format_func=pretty, key="sc_y")
                scol = s3.selectbox("Colour", ["wellbeing_band"] + [c for c in cat_cols if c != "wellbeing_band"],
                                    format_func=pretty, key="sc_c")
                ssz = s4.selectbox("Bubble size", ["(none)"] + num_cols, format_func=lambda c: c if c == "(none)" else pretty(c),
                                   key="sc_s")
                samp = fdf.sample(min(len(fdf), 3000), random_state=1)
                fig = px.scatter(samp, x=sx, y=sy, color=scol, size=None if ssz == "(none)" else ssz,
                                 opacity=0.6, color_discrete_map=BAND_COLORS if scol == "wellbeing_band" else None,
                                 category_orders=CAT_ORDERS, marginal_x="histogram", marginal_y="histogram",
                                 hover_data=["age", "region", "most_used_platform"],
                                 title=f"{pretty(sy)} vs. {pretty(sx)} (sample of {len(samp):,})")
                show(fig, "rel_scatter", height=560)

                dims = st.multiselect("Scatter-matrix variables", num_cols,
                                      default=["daily_screen_hours", "avg_sleep_hours", "anxiety_score_0to27",
                                               "life_satisfaction_1to10"],
                                      format_func=pretty, key="mx_dims")
                if len(dims) >= 2:
                    msamp = fdf.sample(min(len(fdf), 1500), random_state=2)
                    fig = px.scatter_matrix(msamp, dimensions=dims, color="wellbeing_band",
                                            color_discrete_map=BAND_COLORS, category_orders=CAT_ORDERS,
                                            opacity=0.5, title="Scatter matrix")
                    fig.update_traces(diagonal_visible=False, marker=dict(size=3))
                    show(fig, "rel_matrix", height=640)
                else:
                    st.info("Pick at least two variables for the scatter matrix.")

                pdims = st.multiselect("Parallel-coordinates variables (drag along an axis to filter)", num_cols,
                                       default=["daily_screen_hours", "avg_sleep_hours", "anxiety_score_0to27",
                                                "low_mood_score_0to27", "life_satisfaction_1to10"],
                                       format_func=pretty, key="pc_dims")
                if len(pdims) >= 2:
                    pc = fdf.sample(min(len(fdf), 2000), random_state=3).copy()
                    pc["band_code"] = pc["wellbeing_band"].map({"At-risk": 0, "Moderate": 1, "Good": 2})
                    fig = px.parallel_coordinates(pc, dimensions=pdims, color="band_code",
                                                  color_continuous_scale=[BAND_COLORS["At-risk"],
                                                                          BAND_COLORS["Moderate"],
                                                                          BAND_COLORS["Good"]],
                                                  range_color=[0, 2],
                                                  title="Parallel coordinates (red = At-risk, yellow = Moderate, green = Good)")
                    fig.update_coloraxes(showscale=False)
                    show(fig, "rel_parallel", height=460)

            # ---- Habits & sleep
            with t_hab:
                metric = st.selectbox("Metric to analyse", num_cols,
                                      index=num_cols.index("daily_screen_hours"), format_func=pretty, key="hab_metric")
                h1, h2 = st.columns(2)
                with h1:
                    plat = fdf.groupby("most_used_platform")[metric].mean().reset_index().sort_values(metric)
                    fig = px.bar(plat, x=metric, y="most_used_platform", orientation="h", text_auto=".2f",
                                 color=metric, color_continuous_scale="Blues",
                                 labels={"most_used_platform": "Platform", metric: f"Avg {pretty(metric).lower()}"},
                                 title=f"Average {pretty(metric).lower()} by platform")
                    fig.update_coloraxes(showscale=False)
                    show(fig, "hab_plat")
                with h2:
                    fig = px.violin(fdf, x="night_time_use", y="avg_sleep_hours", color="night_time_use",
                                    box=True, category_orders=CAT_ORDERS,
                                    labels={"night_time_use": "Night-time use", "avg_sleep_hours": "Sleep (h)"},
                                    title="Night-time use vs. sleep hours")
                    fig.update_layout(showlegend=False)
                    show(fig, "hab_sleep")

                h3, h4 = st.columns(2)
                with h3:
                    pv = fdf.pivot_table(index="night_time_use", columns="most_used_platform",
                                         values=metric, aggfunc="mean").reindex(NIGHT_ORDER)
                    fig = px.imshow(pv, text_auto=".1f", aspect="auto", color_continuous_scale="RdYlGn_r",
                                    labels={"color": f"Avg {pretty(metric).lower()}"},
                                    title=f"Avg {pretty(metric).lower()}: platform × night-time use")
                    show(fig, "hab_heat")
                with h4:
                    dt = fdf.groupby(["attempted_digital_detox", "uses_screen_time_limits"])[metric].mean().reset_index()
                    fig = px.bar(dt, x="attempted_digital_detox", y=metric, color="uses_screen_time_limits",
                                 barmode="group", category_orders=CAT_ORDERS, text_auto=".2f",
                                 labels={"attempted_digital_detox": "Digital detox", metric: f"Mean {pretty(metric).lower()}",
                                         "uses_screen_time_limits": "Screen-time limits"},
                                 title=f"{pretty(metric)} by limits & detox attempts")
                    show(fig, "hab_detox")

                fig = px.density_heatmap(fdf, x="daily_screen_hours", y="avg_sleep_hours", nbinsx=28, nbinsy=24,
                                         color_continuous_scale="Viridis",
                                         labels={"daily_screen_hours": "Daily screen hours", "avg_sleep_hours": "Sleep (h)"},
                                         title="Where do participants sit? Screen time vs. sleep density")
                show(fig, "hab_density", height=420)

            # ---- Correlations
            with t_corr:
                default_corr = ["daily_screen_hours", "daily_notifications", "avg_sleep_hours",
                                "anxiety_score_0to27", "low_mood_score_0to27", "life_satisfaction_1to10",
                                "loneliness_1to10", "self_esteem_1to10", "fomo_1to10"]
                corr_sel = st.multiselect("Variables in the heatmap", num_cols, default=default_corr,
                                          format_func=pretty, key="corr_sel")
                cc1, cc2 = st.columns([1.3, 1])
                with cc1:
                    if len(corr_sel) >= 2:
                        fig = px.imshow(fdf[corr_sel].corr(), text_auto=".2f", color_continuous_scale="RdBu_r",
                                        zmin=-1, zmax=1, aspect="auto", title="Correlation heatmap")
                        show(fig, "corr_heat", height=520)
                    else:
                        st.info("Pick at least two variables.")
                with cc2:
                    tgt = st.selectbox("Rank everything by correlation with…", num_cols,
                                       index=num_cols.index("anxiety_score_0to27"), format_func=pretty, key="corr_tgt")
                    cs = fdf[num_cols].corr()[tgt].drop(tgt).sort_values().reset_index()
                    cs.columns = ["variable", "correlation"]
                    cs["variable"] = cs["variable"].map(pretty)
                    fig = px.bar(cs, x="correlation", y="variable", orientation="h", color="correlation",
                                 color_continuous_scale="RdBu_r", range_color=[-1, 1],
                                 title=f"Correlation with {pretty(tgt).lower()}")
                    fig.update_coloraxes(showscale=False)
                    show(fig, "corr_rank", height=520)

            # ---- Chart builder
            with t_build:
                st.caption("Build your own chart from any columns.")
                b1, b2, b3, b4 = st.columns(4)
                ctype = b1.selectbox("Chart type", ["Histogram", "Box", "Violin", "Bar (mean)", "Scatter",
                                                    "Density heatmap"], key="cb_type")
                all_cols = num_cols + cat_cols
                bx = b2.selectbox("X", all_cols, index=all_cols.index("daily_screen_hours"),
                                  format_func=pretty, key="cb_x")
                by_ = b3.selectbox("Y (numeric)", num_cols, index=num_cols.index("avg_sleep_hours"),
                                   format_func=pretty, key="cb_y")
                bcol = b4.selectbox("Colour", ["None"] + cat_cols, format_func=lambda c: c if c == "None" else pretty(c),
                                    key="cb_c")
                carg = None if bcol == "None" else bcol
                cmap = BAND_COLORS if carg == "wellbeing_band" else None

                try:
                    if ctype == "Histogram":
                        fig = px.histogram(fdf, x=bx, color=carg, color_discrete_map=cmap, category_orders=CAT_ORDERS,
                                           barmode="overlay", opacity=0.7, marginal="box",
                                           title=f"Distribution of {pretty(bx).lower()}")
                    elif ctype == "Box":
                        fig = px.box(fdf, x=bx, y=by_, color=carg, color_discrete_map=cmap, category_orders=CAT_ORDERS,
                                     title=f"{pretty(by_)} by {pretty(bx).lower()}")
                    elif ctype == "Violin":
                        fig = px.violin(fdf, x=bx, y=by_, color=carg, color_discrete_map=cmap, box=True,
                                        category_orders=CAT_ORDERS, title=f"{pretty(by_)} by {pretty(bx).lower()}")
                    elif ctype == "Bar (mean)":
                        grp = [bx] + ([carg] if carg and carg != bx else [])
                        agg = fdf.groupby(grp)[by_].mean().reset_index()
                        fig = px.bar(agg, x=bx, y=by_, color=carg if carg in agg.columns else None,
                                     color_discrete_map=cmap, category_orders=CAT_ORDERS, barmode="group",
                                     text_auto=".2f", title=f"Mean {pretty(by_).lower()} by {pretty(bx).lower()}")
                    elif ctype == "Scatter":
                        samp = fdf.sample(min(len(fdf), 3000), random_state=4)
                        fig = px.scatter(samp, x=bx, y=by_, color=carg, color_discrete_map=cmap,
                                         category_orders=CAT_ORDERS, opacity=0.6,
                                         title=f"{pretty(by_)} vs. {pretty(bx).lower()}")
                    else:
                        fig = px.density_heatmap(fdf, x=bx, y=by_, nbinsx=25, nbinsy=25,
                                                 color_continuous_scale="Viridis",
                                                 title=f"Density: {pretty(by_)} vs. {pretty(bx).lower()}")
                    show(fig, "cb_fig", height=500)
                except Exception as exc:  # unusual column combinations
                    st.warning(f"That combination can't be drawn: {exc}")

            # ---- Data table
            with t_data:
                st.dataframe(fdf.drop(columns=["age_group"]), width="stretch", height=420)
                st.download_button("📥 Download filtered data (CSV)",
                                   data=fdf.drop(columns=["age_group"]).to_csv(index=False).encode("utf-8"),
                                   file_name="filtered_wellbeing_data.csv", mime="text/csv")

# ---------------------------------------------------------
# TAB 4 - MODEL INSIGHTS
# ---------------------------------------------------------
with tab_model:
    st.subheader("What drives the model?")
    imp = get_importances()
    m1, m2 = st.columns(2)
    with m1:
        if imp is not None and not imp.empty:
            top_n = st.slider("Show top N features", 5, len(imp), min(12, len(imp)), key="imp_n")
            top = imp.head(top_n).sort_values("importance")
            top["feature"] = top["feature"].map(pretty)
            fig = px.bar(top, x="importance", y="feature", orientation="h", text_auto=".1%",
                         color="importance", color_continuous_scale="Teal",
                         title="Random-forest feature importance (summed over one-hot columns)")
            fig.update_coloraxes(showscale=False)
            show(fig, "imp_bar", height=460)
        else:
            st.info("Feature importances could not be read from this model file.")
    with m2:
        if df is not None:
            samp = df.sample(min(len(df), 3000), random_state=5)
            fig = px.scatter(samp, x="anxiety_score_0to27", y="low_mood_score_0to27", color="wellbeing_band",
                             color_discrete_map=BAND_COLORS, category_orders=CAT_ORDERS, opacity=0.6,
                             title="The two columns that separate the bands",
                             labels={"anxiety_score_0to27": "Anxiety (0-27)", "low_mood_score_0to27": "Low mood (0-27)"})
            fig.add_trace(go.Scatter(x=[inputs["anxiety_score_0to27"]], y=[inputs["low_mood_score_0to27"]],
                                     mode="markers", name="You",
                                     marker=dict(size=16, color="white", symbol="star", line=dict(color="black", width=2))))
            show(fig, "imp_scatter", height=460)

    if df is not None:
        st.markdown("#### Average profile per band")
        prof = df.groupby("wellbeing_band")[[c for c in df.select_dtypes("number").columns]].mean().round(2)
        prof = prof.reindex(BAND_ORDER)
        prof.columns = [pretty(c) for c in prof.columns]
        st.dataframe(prof.T, width="stretch")

    st.info(
        "**Honest note:** in this (synthetic) dataset the wellbeing band is almost fully determined by the "
        "anxiety and low-mood scores, so the model's near-perfect accuracy mostly reflects how the label was "
        "built, not real-world predictive power. Social-media variables contribute little here, and nothing on "
        "this page implies causation."
    )

# ---------------------------------------------------------
# TAB 5 - BATCH PROCESSING
# ---------------------------------------------------------
with tab_batch:
    st.subheader("Batch CSV prediction")
    st.caption(f"Your CSV needs these {len(FEATURES)} columns (extra columns such as `participant_id` are kept in the output).")

    bc1, bc2 = st.columns([2, 1])
    with bc2:
        if df is not None:
            template = df[["participant_id"] + FEATURES].head(5).copy()
            template.loc[template["gender"] == "Unknown", "gender"] = np.nan
        else:
            template = pd.DataFrame(columns=["participant_id"] + FEATURES)
        st.download_button("📄 Download CSV template", data=template.to_csv(index=False).encode("utf-8"),
                           file_name="batch_template.csv", mime="text/csv", width="stretch")
        use_sample = st.checkbox("Or try 200 random rows from the dataset", disabled=df is None)
    with bc1:
        uploaded = st.file_uploader("Upload CSV containing participant data", type=["csv"])

    batch = None
    if use_sample and df is not None:
        batch = (df.sample(200, random_state=7).drop(columns=["wellbeing_band", "age_group"])
                 .replace({"gender": {"Unknown": np.nan}}))
    elif uploaded is not None:
        try:
            batch = pd.read_csv(uploaded)
        except Exception as exc:
            st.error(f"Could not read that file as CSV: {exc}")

    if batch is not None:
        missing = [c for c in FEATURES if c not in batch.columns]
        if missing:
            st.error("Missing required columns: " + ", ".join(missing))
        else:
            try:
                bp = predict_proba_df(batch[FEATURES])
                result = batch.copy()
                result["Predicted_Wellbeing_Band"] = bp.idxmax(axis=1)
                result["Confidence"] = bp.max(axis=1).round(4)
                for b in bp.columns:
                    result[f"P({b})"] = bp[b].round(4)

                r1, r2 = st.columns(2)
                with r1:
                    cnt = result["Predicted_Wellbeing_Band"].value_counts().reset_index()
                    cnt.columns = ["band", "count"]
                    fig = px.pie(cnt, names="band", values="count", hole=0.45, color="band",
                                 color_discrete_map=BAND_COLORS, title="Predicted bands")
                    show(fig, "batch_pie", height=320)
                with r2:
                    fig = px.histogram(result, x="Confidence", color="Predicted_Wellbeing_Band", nbins=20,
                                       color_discrete_map=BAND_COLORS, title="Prediction confidence")
                    show(fig, "batch_hist", height=320)

                keep = st.multiselect("Show only these predicted bands", BAND_ORDER, default=BAND_ORDER, key="batch_filter")
                view = result[result["Predicted_Wellbeing_Band"].isin(keep)]
                st.dataframe(view.head(200), width="stretch")
                st.caption(f"Showing up to 200 of {len(view):,} filtered rows (download contains all rows).")
                st.download_button("📥 Download predictions (CSV)",
                                   data=result.to_csv(index=False).encode("utf-8"),
                                   file_name="wellbeing_predictions.csv", mime="text/csv")
            except Exception as exc:
                st.error(f"Prediction failed. Check the values and column types in your file. Details: {exc}")

st.divider()
st.caption("⚠️ Educational project built on a synthetic dataset. Predictions are not medical advice. "
           "If you or someone you know is struggling, please reach out to a qualified professional or a local helpline.")
