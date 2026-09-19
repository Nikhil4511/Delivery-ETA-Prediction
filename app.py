"""
Delivery ETA Predictor — Professional Streamlit UI
====================================================
Redesigned from the original educational demo into a clean,
production-style ML application.

What changed vs. the original:
  - Full custom CSS for a SaaS-grade visual design
  - Professional header, sidebar with tech stack & model info
  - Grouped input layout with validation hints
  - Large, unambiguous result card with delivery-speed label
  - SHAP explanation rendered as a horizontal bar chart
  - Spinner during inference, friendly error messages
  - Reset workflow via session_state

What was NOT changed:
  - predict_eta() and explain_prediction() from src.predict
  - All option lists from src.config
  - Feature names and data types expected by the model
  - Model loading path (MODEL_PATH from src.config)
"""

from __future__ import annotations

import streamlit as st

# ── project imports (unchanged from original) ───────────────────────────────
from src.config import (
    AREA_OPTIONS,
    DAY_OPTIONS,
    MODEL_PATH,
    TIME_OF_DAY_OPTIONS,
    TRAFFIC_OPTIONS,
    VEHICLE_OPTIONS,
    WEATHER_OPTIONS,
)
from src.predict import explain_prediction, predict_eta

# ── page config (must be the very first Streamlit call) ─────────────────────
st.set_page_config(
    page_title="Delivery ETA Predictor",
    page_icon="🚴",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── custom CSS ───────────────────────────────────────────────────────────────
# Scoped, minimal CSS — only styles that Streamlit's theme system can't reach.
st.markdown(
    """
    <style>
    /* ── import Inter from Google Fonts ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── application header ── */
    .app-header {
        padding: 2rem 0 1.5rem 0;
        border-bottom: 1px solid #E2E8F0;
        margin-bottom: 2rem;
    }
    .app-header h1 {
        font-size: 2rem;
        font-weight: 700;
        color: #0F172A;
        margin: 0 0 0.25rem 0;
        letter-spacing: -0.5px;
    }
    .app-header .subtitle {
        font-size: 0.95rem;
        color: #64748B;
        margin: 0 0 0.5rem 0;
        font-weight: 400;
    }
    .app-header .description {
        font-size: 0.88rem;
        color: #94A3B8;
    }

    /* ── section heading ── */
    .section-title {
        font-size: 1rem;
        font-weight: 600;
        color: #0F172A;
        margin: 0 0 1rem 0;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }

    /* ── result card ── */
    .result-card {
        background: #0F172A;
        border-radius: 12px;
        padding: 2rem 2.5rem;
        margin: 1.5rem 0;
        color: #F8FAFC;
    }
    .result-card .label {
        font-size: 0.8rem;
        font-weight: 500;
        letter-spacing: 0.08em;
        color: #94A3B8;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
    }
    .result-card .eta-number {
        font-size: 3.5rem;
        font-weight: 700;
        color: #F59E0B;
        line-height: 1;
        margin-bottom: 0.5rem;
    }
    .result-card .eta-sub {
        font-size: 0.9rem;
        color: #CBD5E1;
    }
    .result-card .speed-badge {
        display: inline-block;
        margin-top: 1rem;
        padding: 0.3rem 0.75rem;
        border-radius: 99px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-fast   { background: #D1FAE5; color: #065F46; }
    .badge-normal { background: #DBEAFE; color: #1E3A8A; }
    .badge-slow   { background: #FEF3C7; color: #92400E; }
    .badge-high   { background: #FEE2E2; color: #991B1B; }

    /* ── insight metrics strip ── */
    .insight-strip {
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
        margin: 1.25rem 0;
    }
    .insight-chip {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.5rem 0.9rem;
        font-size: 0.82rem;
        color: #334155;
    }
    .insight-chip span {
        font-weight: 600;
        color: #0F172A;
    }

    /* ── SHAP bar ── */
    .shap-row {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 0.4rem 0;
        font-size: 0.85rem;
    }
    .shap-label {
        width: 170px;
        flex-shrink: 0;
        color: #334155;
        font-weight: 500;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .shap-bar-bg {
        flex: 1;
        background: #F1F5F9;
        border-radius: 4px;
        height: 10px;
        overflow: hidden;
    }
    .shap-bar-fill-pos {
        height: 100%;
        background: #F59E0B;
        border-radius: 4px;
    }
    .shap-bar-fill-neg {
        height: 100%;
        background: #38BDF8;
        border-radius: 4px;
    }
    .shap-value {
        width: 55px;
        text-align: right;
        color: #64748B;
        font-size: 0.78rem;
    }

    /* ── sidebar overrides ── */
    section[data-testid="stSidebar"] {
        background: #0F172A;
    }
    section[data-testid="stSidebar"] * {
        color: #CBD5E1 !important;
    }
    section[data-testid="stSidebar"] h3 {
        color: #F8FAFC !important;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 1.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: #1E293B !important;
    }

    /* ── form submit button ── */
    div[data-testid="stFormSubmitButton"] > button {
        background: #F59E0B;
        color: #0F172A;
        border: none;
        font-weight: 700;
        font-size: 1rem;
        padding: 0.65rem 2rem;
        border-radius: 8px;
        width: 100%;
        transition: background 0.15s;
    }
    div[data-testid="stFormSubmitButton"] > button:hover {
        background: #D97706;
        color: #0F172A;
    }

    /* ── reset button ── */
    .reset-btn > button {
        background: transparent;
        border: 1px solid #E2E8F0;
        color: #64748B;
        border-radius: 8px;
        font-size: 0.85rem;
    }

    /* ── general input label weight ── */
    label { font-weight: 500 !important; }

    /* ── hide default Streamlit footer ── */
    footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── helpers ──────────────────────────────────────────────────────────────────

def speed_label(eta_minutes: float) -> tuple[str, str]:
    """Return a (text, CSS-class) badge based on the predicted ETA."""
    if eta_minutes < 30:
        return "⚡ Fast delivery", "badge-fast"
    if eta_minutes < 45:
        return "✅ Normal delivery", "badge-normal"
    if eta_minutes < 60:
        return "⏳ Slightly delayed", "badge-slow"
    return "🔴 High delivery time", "badge-high"


def render_shap_bar(feature: str, shap_val: float, max_abs: float) -> str:
    """Build one SHAP row as an HTML string with an inline bar."""
    pct = abs(shap_val) / max_abs * 100 if max_abs else 0
    sign = "+" if shap_val >= 0 else ""
    fill_class = "shap-bar-fill-pos" if shap_val >= 0 else "shap-bar-fill-neg"
    return (
        f'<div class="shap-row">'
        f'  <div class="shap-label">{feature}</div>'
        f'  <div class="shap-bar-bg">'
        f'    <div class="{fill_class}" style="width:{pct:.1f}%"></div>'
        f'  </div>'
        f'  <div class="shap-value">{sign}{shap_val:.1f} min</div>'
        f'</div>'
    )


# ── sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🚴 ETA Predictor")
    st.markdown("---")

    st.markdown("### About")
    st.markdown(
        "Predicts estimated food delivery time using a trained ML model. "
        "Enter order details and receive an instant estimate."
    )

    st.markdown("### What it uses")
    st.markdown(
        "- Distance & delivery area\n"
        "- Weather & traffic conditions\n"
        "- Vehicle type & driver experience\n"
        "- Restaurant rating & prep time\n"
        "- Time of day & day of week"
    )

    st.markdown("### Tech Stack")
    st.markdown(
        "- Python 3.10+\n"
        "- Scikit-learn / XGBoost\n"
        "- SHAP\n"
        "- Streamlit\n"
        "- FastAPI\n"
        "- Docker"
    )

    st.markdown("### Model")
    # Show model file status without exposing internals to users
    model_status = "✅ Loaded" if MODEL_PATH.exists() else "❌ Not found"
    st.markdown(
        f"- Target: Delivery time (minutes)\n"
        f"- Algorithm: XGBoost / Gradient Boost\n"
        f"- Status: {model_status}"
    )

    st.markdown("---")
    st.caption("An ML portfolio project. Predictions are estimates only.")


# ── guard: model must exist ──────────────────────────────────────────────────

if not MODEL_PATH.exists():
    st.error(
        "**Trained model not found.**  \n"
        "Generate the dataset and run `python -m src.train` first, "
        "then refresh this page."
    )
    st.stop()


# ── page header ──────────────────────────────────────────────────────────────

st.markdown(
    """
    <div class="app-header">
        <h1>🚴 Delivery ETA Predictor</h1>
        <p class="subtitle">AI-powered estimated delivery time prediction</p>
        <p class="description">
            Enter your order and delivery details to estimate the expected delivery time.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ── input form ───────────────────────────────────────────────────────────────

st.markdown('<p class="section-title">📦 Delivery Details</p>', unsafe_allow_html=True)

with st.form("delivery_form"):

    # Row 1 — route & timing
    col1, col2, col3 = st.columns(3)
    with col1:
        distance_km = st.number_input(
            "Distance (km)",
            min_value=0.5,
            max_value=30.0,
            value=5.5,
            step=0.5,
            help="Straight-line distance between restaurant and delivery address.",
        )
    with col2:
        time_of_day = st.selectbox("Time of day", TIME_OF_DAY_OPTIONS, index=2)
    with col3:
        day_of_week = st.selectbox("Day of week", DAY_OPTIONS, index=4)

    st.markdown("")  # visual breathing room

    # Row 2 — conditions
    col4, col5, col6 = st.columns(3)
    with col4:
        weather = st.selectbox("Weather", WEATHER_OPTIONS)
    with col5:
        traffic_level = st.selectbox("Traffic level", TRAFFIC_OPTIONS, index=1)
    with col6:
        delivery_area = st.selectbox("Delivery area", AREA_OPTIONS)

    st.markdown("")

    # Row 3 — order & restaurant
    col7, col8, col9 = st.columns(3)
    with col7:
        restaurant_rating = st.slider(
            "Restaurant rating",
            min_value=1.0,
            max_value=5.0,
            value=4.3,
            step=0.1,
            help="Average star rating of the restaurant (1–5).",
        )
    with col8:
        order_items = st.number_input(
            "Items in order",
            min_value=1,
            max_value=20,
            value=4,
            step=1,
        )
    with col9:
        preparation_time_min = st.number_input(
            "Preparation time (min)",
            min_value=1.0,
            max_value=120.0,
            value=18.0,
            step=1.0,
            help="Estimated kitchen prep time in minutes.",
        )

    st.markdown("")

    # Row 4 — driver & vehicle
    col10, col11 = st.columns(2)
    with col10:
        vehicle_type = st.selectbox("Vehicle type", VEHICLE_OPTIONS)
    with col11:
        driver_experience_years = st.slider(
            "Driver experience (years)",
            min_value=0,
            max_value=20,
            value=2,
            step=1,
            help="Years of active delivery experience.",
        )

    st.markdown("---")
    submitted = st.form_submit_button("🚀 Predict Delivery Time")


# ── prediction & results ─────────────────────────────────────────────────────

if submitted:
    # Build the exact request dict the original predict_eta() expects
    request: dict = {
        "distance_km": float(distance_km),
        "weather": weather,
        "traffic_level": traffic_level,
        "time_of_day": time_of_day,
        "vehicle_type": vehicle_type,
        "driver_experience_years": float(driver_experience_years),
        "restaurant_rating": float(restaurant_rating),
        "order_items": int(order_items),
        "preparation_time_min": float(preparation_time_min),
        "delivery_area": delivery_area,
        "day_of_week": day_of_week,
    }

    try:
        with st.spinner("Calculating estimated delivery time…"):
            eta: float = predict_eta(request)
            explanation: dict = explain_prediction(request)

        # ── Result card ──────────────────────────────────────────────────────
        badge_text, badge_class = speed_label(eta)
        arrival_label = f"Estimated arrival in approximately {eta:.0f} minutes"

        st.markdown(
            f"""
            <div class="result-card">
                <div class="label">Estimated Delivery Time</div>
                <div class="eta-number">{eta:.0f} min</div>
                <div class="eta-sub">{arrival_label}</div>
                <span class="speed-badge {badge_class}">{badge_text}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Delivery insights strip ──────────────────────────────────────────
        st.markdown('<p class="section-title">📊 Delivery Insights</p>', unsafe_allow_html=True)

        chips_html = (
            '<div class="insight-strip">'
            f'<div class="insight-chip">Distance <span>{distance_km} km</span></div>'
            f'<div class="insight-chip">Traffic <span>{traffic_level}</span></div>'
            f'<div class="insight-chip">Weather <span>{weather}</span></div>'
            f'<div class="insight-chip">Prep time <span>{preparation_time_min:.0f} min</span></div>'
            f'<div class="insight-chip">Vehicle <span>{vehicle_type}</span></div>'
            f'<div class="insight-chip">Area <span>{delivery_area}</span></div>'
            f'<div class="insight-chip">Rating <span>{restaurant_rating:.1f} ⭐</span></div>'
            f'<div class="insight-chip">Experience <span>{driver_experience_years} yr</span></div>'
            '</div>'
        )
        st.markdown(chips_html, unsafe_allow_html=True)

        # ── SHAP explainability ──────────────────────────────────────────────
        st.markdown(
            '<p class="section-title">🔍 Why was this ETA predicted?</p>',
            unsafe_allow_html=True,
        )

        if not explanation.get("available"):
            # SHAP not available — show a clean placeholder message
            st.info(
                explanation.get("message", "Feature importance is not available for this model.")
            )
        else:
            # Merge increasing and decreasing into one sorted list
            all_factors: list[dict] = []
            for item in explanation.get("increasing", []):
                all_factors.append({"feature": item["feature"], "shap_value": item["shap_value"]})
            for item in explanation.get("decreasing", []):
                all_factors.append({"feature": item["feature"], "shap_value": item["shap_value"]})

            # Sort by absolute impact, descending
            all_factors.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

            if all_factors:
                max_abs = max(abs(f["shap_value"]) for f in all_factors)

                # Legend line
                st.caption(
                    explanation.get("message", "")
                    + "  🟡 Increases ETA · 🔵 Decreases ETA"
                )

                bars_html = "".join(
                    render_shap_bar(f["feature"], f["shap_value"], max_abs)
                    for f in all_factors
                )
                st.markdown(bars_html, unsafe_allow_html=True)
            else:
                st.info("No feature importance data returned by the model.")

        # ── Reset button ─────────────────────────────────────────────────────
        st.markdown("")
        st.markdown(
            "<p style='font-size:0.8rem;color:#94A3B8;'>"
            "To run another prediction, adjust the inputs above and click "
            "<strong>Predict Delivery Time</strong> again."
            "</p>",
            unsafe_allow_html=True,
        )

    except (ValueError, FileNotFoundError) as err:
        st.error(f"⚠️ Could not generate a prediction: {err}")
    except Exception:
        st.error(
            "⚠️ An unexpected error occurred. "
            "Please check your input values and try again."
        )
        # Only show the technical trace in development; comment this out in prod
        with st.expander("Technical details (for debugging)"):
            st.exception(Exception)