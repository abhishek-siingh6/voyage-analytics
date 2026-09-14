"""
Voyage Analytics Streamlit Dashboard
"""

import os
import json
from textwrap import dedent
from typing import Optional
try:
    from streamlit_javascript import st_javascript
except ImportError:
    def st_javascript(code): return None

from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt


LOGO_PATH = Path(__file__).resolve().parent.parent / "assets" / "va_logo.png"

# =============================================================================
# CONFIGURATION
# =============================================================================

API_BASE = os.getenv(
    "API_BASE_URL",
    "http://localhost:5000"
).rstrip("/")

HOTELS_PATH = os.getenv(
    "HOTELS_DATA_PATH",
    "artifacts/data/hotels.csv"
)

USERS_PATH = os.getenv(
    "USERS_DATA_PATH",
    "artifacts/data/users.csv"
)

CATALOG_PATH = os.getenv(
    "HOTELS_CATALOG_PATH",
    "artifacts/hotel_catalog.json"
)


st.set_page_config(
    page_title="Voyage Analytics",
    page_icon="Γ£ê∩╕Å",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# IMPORTANT HTML RENDERER
# =============================================================================
# This is the main fix.
#
# Streamlit was interpreting indented HTML as a Markdown code block.
# We remove indentation/newline formatting before sending HTML to Streamlit.
# =============================================================================

def render_html(content: str) -> None:
    html = dedent(content).strip()

    html = " ".join(
        line.strip()
        for line in html.splitlines()
        if line.strip()
    )

    st.markdown(
        html,
        unsafe_allow_html=True,
    )


# =============================================================================
# CUSTOM CSS
# =============================================================================

st.markdown(
    """
<style>

:root {
    --bg: #050a14;
    --panel: #0b172b;
    --panel-2: #0e1a2f;
    --border: #263b5c;
    --text: #f4f7ff;
    --muted: #91a8c7;
    --blue: #2f80ff;
    --purple: #5b3ff5;
}


/* -------------------------------------------------------------------------- */
/* GLOBAL                                                                     */
/* -------------------------------------------------------------------------- */

.stApp {
    background:
        radial-gradient(
            circle at 80% 10%,
            rgba(47, 128, 255, 0.07),
            transparent 30%
        ),
        #0A1628 !important;
    color: var(--text);
}

[data-testid="stAppViewContainer"] {
    background: #0A1628 !important;
}

[data-testid="stHeader"] {
    background: rgba(5, 10, 20, 0.95) !important;
}

[data-testid="stDecoration"] {
    display: none !important;
}

[data-testid="stToolbar"] {
    visibility: hidden;
}

/* Main dashboard ΓÇö use only the main content container. */
section.main > div.block-container,
section[data-testid="stMain"] > div.block-container {
    max-width: none !important;
    width: 100% !important;
    box-sizing: border-box !important;
    margin-left: 0 !important;
    margin-right: 0 !important;
    padding-top: 1.4rem !important;
    padding-bottom: 3rem !important;
    padding-left: 0.75rem !important;
    padding-right: 0.75rem !important;
}

h1, h2, h3, h4, h5, h6 {
    color: #f4f7ff !important;
}

p, label, span {
    color: #dbe7ff;
}


/* -------------------------------------------------------------------------- */
/* SIDEBAR                                                                    */
/* -------------------------------------------------------------------------- */

section[data-testid="stSidebar"] {
    background: #0A1628 !important;

    border-right: 1px solid #172941 !important;
}

section[data-testid="stSidebar"][aria-expanded="true"] {
    width: 25vw !important;
    min-width: 25vw !important;
    max-width: 25vw !important;
    flex: 0 0 25vw !important;
}

section[data-testid="stSidebar"][aria-expanded="false"] {
    width: 0 !important;
    min-width: 0 !important;
    max-width: 0 !important;
}

section[data-testid="stSidebar"] > div {
    background: transparent !important;
}

section[data-testid="stSidebar"] [data-testid="stSidebarContent"],
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    background: transparent !important;
    width: 100% !important;
    max-width: none !important;
    box-sizing: border-box !important;
}

section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    padding-left: 18px !important;
    padding-right: 18px !important;
    padding-bottom: 20px !important;
    box-sizing: border-box !important;
}

@media (max-width: 768px) {
    section[data-testid="stSidebar"] {
        min-width: 0 !important;
    }

    section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
        padding-left: 12px !important;
        padding-right: 12px !important;
    }
}


/* Sidebar logo */

.va-logo {
    width: 58px;
    height: 58px;
    border-radius: 16px;
    overflow: hidden;

    display: flex;
    align-items: center;
    justify-content: center;

    box-shadow:
        0 0 18px rgba(47, 128, 255, 0.45),
        0 0 35px rgba(91, 63, 245, 0.20);
}

.va-logo img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    display: block;
}

    box-shadow:
        0 0 18px rgba(47, 128, 255, 0.45),
        0 0 35px rgba(91, 63, 245, 0.20);
}

.sidebar-brand {
    display: flex;
    align-items: center;
    gap: 13px;
    margin-bottom: 28px;
}

.sidebar-brand-title {
    color: #f5f7ff;
    font-size: 18px;
    font-weight: 800;
}

.sidebar-brand-subtitle {
    color: #7f95b4;
    font-size: 11px;
    margin-top: 2px;
}


/* Navigation title */

.nav-label {
    color: #6f86a6;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 1.8px;
    margin: 15px 0 10px;
}


/* Sidebar buttons */

section[data-testid="stSidebar"] div.stButton {
    width: 100%;
}

section[data-testid="stSidebar"] div.stButton > button {
    width: 100% !important;
    min-height: 48px !important;

    background: #0b1729 !important;
    color: #dce8fb !important;

    border: 1px solid #1c304d !important;
    border-radius: 12px !important;

    text-align: left !important;
    padding-left: 16px !important;

    font-size: 14px !important;
    font-weight: 600 !important;

    transition:
        border-color 0.2s ease,
        background 0.2s ease,
        transform 0.2s ease;
}

section[data-testid="stSidebar"] div.stButton > button:hover {
    background: #10213a !important;
    border-color: #31588b !important;
    color: #ffffff !important;
}

section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
    background:
        linear-gradient(
            90deg,
            #2f80ff,
            #5b3ff5
        ) !important;

    border: none !important;
    color: white !important;

    box-shadow:
        0 0 18px rgba(47, 128, 255, 0.22);
}


/* Sidebar status */

.sidebar-status {
    margin-top: 35px;
    padding-top: 18px;
    border-top: 1px solid #172941;
}

.status-row {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #9eb2ce;
    font-size: 12px;
    margin-bottom: 12px;
}

.status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #27d98b;
    box-shadow: 0 0 10px rgba(39, 217, 139, 0.8);
}

.sidebar-info-label {
    color: #617895;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.sidebar-info-value {
    color: #dbe7ff;
    font-size: 12px;
    margin-top: 2px;
}


/* -------------------------------------------------------------------------- */
/* HERO                                                                       */
/* -------------------------------------------------------------------------- */

.hero-card {
    width: 100%;

    background:
        radial-gradient(
            circle at 85% 20%,
            rgba(47, 128, 255, 0.12),
            transparent 30%
        ),
        linear-gradient(
            135deg,
            #0d1c34,
            #091528
        );

    border: 1px solid #294a73;
    border-radius: 22px;

    padding: 34px 38px;

    margin-bottom: 35px;

    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.025),
        0 15px 45px rgba(0,0,0,0.18);
}

.hero-title {
    color: #f6f8ff;
    font-size: 42px;
    line-height: 1.1;
    font-weight: 900;
    letter-spacing: -1.2px;
}

.hero-subtitle {
    color: #8fb5e8;
    font-size: 16px;
    margin-top: 10px;
}


/* -------------------------------------------------------------------------- */
/* SECTION HEADERS                                                            */
/* -------------------------------------------------------------------------- */

.page-title {
    color: #f4f7ff;
    font-size: 38px;
    font-weight: 850;
    letter-spacing: -0.8px;
    margin-bottom: 4px;
}

.page-subtitle {
    color: #88a9d3;
    font-size: 15px;
    margin-bottom: 26px;
}

.section-line {
    height: 1px;
    background: #1c2d46;
    margin: 28px 0 32px;
}


/* -------------------------------------------------------------------------- */
/* METRIC CARDS                                                               */
/* -------------------------------------------------------------------------- */

.metric-card {
    background:
        linear-gradient(
            145deg,
            #0e1b30,
            #0a1628
        );

    border: 1px solid #294466;
    border-radius: 17px;

    min-height: 145px;

    padding: 24px 25px;

    box-shadow:
        inset 0 1px 0 rgba(255,255,255,0.025),
        0 8px 30px rgba(0,0,0,0.15);
}

.metric-label {
    color: #8da5c5;
    font-size: 13px;
    font-weight: 600;
    margin-bottom: 13px;
}

.metric-value {
    color: #f6f8ff;
    font-size: 30px;
    font-weight: 850;
    line-height: 1;
}


/* -------------------------------------------------------------------------- */
/* RESULT CARD                                                                */
/* -------------------------------------------------------------------------- */

.result-card {
    background:
        linear-gradient(
            145deg,
            #0e1c31,
            #0a1628
        );

    border: 1px solid #2d5078;
    border-radius: 18px;

    padding: 28px 32px;
    margin-top: 25px;

    box-shadow:
        0 0 25px rgba(47,128,255,0.08);
}

.result-title {
    color: #8ea8c9;
    font-size: 14px;
    font-weight: 600;
}

.result-value {
    color: #f5f8ff;
    font-size: 36px;
    font-weight: 900;
    margin-top: 8px;
}


/* -------------------------------------------------------------------------- */
/* INPUTS                                                                     */
/* -------------------------------------------------------------------------- */

div[data-baseweb="input"] {
    background-color: #0e1a2f !important;
    border: 1px solid #31415f !important;
    border-radius: 12px !important;
}

div[data-baseweb="input"] > div {
    background-color: #0e1a2f !important;
}

div[data-baseweb="input"] input {
    background-color: #0e1a2f !important;
    color: #f5f7ff !important;
    -webkit-text-fill-color: #f5f7ff !important;
}

div[data-baseweb="input"] input::placeholder {
    color: #6e84a3 !important;
}


/* Date input */

div[data-baseweb="input"] button {
    color: #9db5d5 !important;
}


/* Number input buttons */

div[data-baseweb="input"] button:hover {
    background: #172a46 !important;
}


/* Selectbox */

div[data-baseweb="select"] > div {
    background-color: #0e1a2f !important;
    color: #f5f7ff !important;

    border: 1px solid #31415f !important;
    border-radius: 12px !important;
}

div[data-baseweb="select"] input,
div[data-baseweb="select"] span {
    color: #f5f7ff !important;
    -webkit-text-fill-color: #f5f7ff !important;
}


/* Selectbox dropdown */

div[role="listbox"] {
    background: #0e1a2f !important;
    border: 1px solid #31415f !important;
}

div[role="option"] {
    background: #0e1a2f !important;
    color: #f5f7ff !important;
}

div[role="option"]:hover {
    background: #172a46 !important;
}


/* Labels */

[data-testid="stWidgetLabel"] p {
    color: #9eb3d0 !important;
    font-weight: 600 !important;
}


/* -------------------------------------------------------------------------- */
/* BUTTONS                                                                    */
/* -------------------------------------------------------------------------- */

div.stButton > button,
div.stFormSubmitButton > button {
    background:
        linear-gradient(
            90deg,
            #2f80ff,
            #5b3ff5
        ) !important;

    color: white !important;

    border: none !important;
    border-radius: 11px !important;

    min-height: 45px !important;

    font-weight: 700 !important;

    box-shadow:
        0 5px 18px rgba(47,128,255,0.18);
}

div.stButton > button:hover,
div.stFormSubmitButton > button:hover {
    color: white !important;

    box-shadow:
        0 7px 24px rgba(47,128,255,0.30);
}


/* -------------------------------------------------------------------------- */
/* FORM / CONTAINERS                                                          */
/* -------------------------------------------------------------------------- */

[data-testid="stForm"] {
    background: #091528 !important;
    border: 1px solid #223957 !important;
    border-radius: 18px !important;
    padding: 25px !important;
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #0b172a !important;
    border-color: #243c5d !important;
}


/* -------------------------------------------------------------------------- */
/* ALERTS                                                                     */
/* -------------------------------------------------------------------------- */

div[data-testid="stAlert"] {
    border-radius: 12px !important;
}


/* -------------------------------------------------------------------------- */
/* MODEL INFORMATION                                                          */
/* -------------------------------------------------------------------------- */

div[data-testid="stCode"] {
    background: #0b172a !important;
    border: 1px solid #263f60 !important;
    border-radius: 14px !important;
}

div[data-testid="stCode"] pre {
    background: #0b172a !important;
    color: #dbe7ff !important;
}


/* -------------------------------------------------------------------------- */
/* DATAFRAME                                                                  */
/* -------------------------------------------------------------------------- */

[data-testid="stDataFrame"] {
    border: 1px solid #263f60 !important;
    border-radius: 12px !important;
}


/* -------------------------------------------------------------------------- */
/* SLIDER                                                                     */
/* -------------------------------------------------------------------------- */

div[data-baseweb="slider"] {
    color: #2f80ff !important;
}


/* -------------------------------------------------------------------------- */
/* CHART PANELS                                                               */
/* -------------------------------------------------------------------------- */

.chart-panel {
    background:
        linear-gradient(
            145deg,
            #0d1b30,
            #091528
        );

    border: 1px solid #263f61;
    border-radius: 18px;

    padding: 22px;

    margin-bottom: 24px;

    box-shadow:
        0 8px 30px rgba(0,0,0,0.12);
}

.chart-title {
    color: #f0f5ff;
    font-size: 18px;
    font-weight: 800;
}

.chart-subtitle {
    color: #7f98b9;
    font-size: 12px;
    margin-top: 4px;
}


/* -------------------------------------------------------------------------- */
/* EXPANDERS                                                                  */
/* -------------------------------------------------------------------------- */

div[data-testid="stExpander"] {
    background: #0b172a !important;
    border: 1px solid #263f60 !important;
    border-radius: 14px !important;
}

div[data-testid="stExpander"] summary {
    color: #e5edfb !important;
}


/* -------------------------------------------------------------------------- */
/* METRIC / CODE FIX                                                          */
/* -------------------------------------------------------------------------- */

pre {
    white-space: pre-wrap !important;
}


/* -------------------------------------------------------------------------- */
/* SCROLLBAR                                                                  */
/* -------------------------------------------------------------------------- */

::-webkit-scrollbar {
    width: 7px;
}

::-webkit-scrollbar-track {
    background: #050a14;
}

::-webkit-scrollbar-thumb {
    background: #1d3454;
    border-radius: 10px;
}


/* ============================================================
   CONTENT PROPORTIONS — FINAL POLISH
   ============================================================ */

/* Keep the dashboard content visually compact */
section.main > div.block-container,
section[data-testid="stMain"] > div.block-container {
    max-width: 1180px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    padding-left: 28px !important;
    padding-right: 28px !important;
}

/* Hero */
.hero-card {
    width: 100% !important;
    max-width: 1180px !important;
    padding: 28px 32px !important;
    margin-bottom: 28px !important;
}

.hero-title {
    font-size: 38px !important;
}

.hero-subtitle {
    font-size: 15px !important;
}

/* Page headings */
.page-title {
    font-size: 32px !important;
}

.page-subtitle {
    font-size: 14px !important;
    margin-bottom: 22px !important;
}

/* Metric cards — less bulky */
.metric-card {
    min-height: 125px !important;
    padding: 20px 22px !important;
}

.metric-label {
    font-size: 12px !important;
    margin-bottom: 10px !important;
}

.metric-value {
    font-size: 27px !important;
}

/* Result cards */
.result-card {
    padding: 24px 28px !important;
}

/* Inputs should not feel oversized */
div[data-baseweb="input"] input,
div[data-baseweb="select"] {
    font-size: 14px !important;
}

/* Sidebar should feel like one continuous surface */
section[data-testid="stSidebar"] > div,
section[data-testid="stSidebar"] [data-testid="stSidebarContent"],
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"],
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] > div,
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] > div {
    background: transparent !important;
}

/* Remove the "nested box" feeling */
section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    padding: 0 12px !important;
}

/* Sidebar branding */
.sidebar-brand {
    margin-top: 4px !important;
    margin-bottom: 22px !important;
}

/* Navigation */
.nav-label {
    margin-top: 10px !important;
    margin-bottom: 8px !important;
}

section[data-testid="stSidebar"] div.stButton > button {
    min-height: 46px !important;
    font-size: 13px !important;
}

/* Sidebar status */
.sidebar-status {
    margin-top: 28px !important;
    padding-top: 15px !important;
}


/* FINAL CONTENT PROPORTIONS */
.hero-card {
    max-width: 1180px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    padding: 28px 32px !important;
}

.metric-card {
    min-height: 125px !important;
    padding: 20px 22px !important;
}

section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    padding: 0 12px !important;
}


/* Match the outer Streamlit page background */
html,
body,
#root {
    background: #0A1628 !important;
}

[data-testid="stAppViewContainer"] {
    background: #0A1628 !important;
}

[data-testid="stHeader"] {
    background: #0A1628 !important;
}

</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# API HELPERS
# =============================================================================

def api_ok() -> bool:
    try:
        response = requests.get(
            f"{API_BASE}/api/health",
            timeout=3,
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


@st.cache_data(show_spinner=False)
def api_predict_flight(payload: dict) -> dict:
    response = requests.post(
        f"{API_BASE}/api/predict",
        json=payload,
        timeout=10,
    )
    return response.json()


@st.cache_data(show_spinner=False)
def api_predict_gender(payload: dict) -> dict:
    response = requests.post(
        f"{API_BASE}/api/gender/predict",
        json=payload,
        timeout=10,
    )
    return response.json()


@st.cache_data(show_spinner=False)
def api_recommend(payload: dict) -> list:
    response = requests.post(
        f"{API_BASE}/api/recommend/recommendations",
        json=payload,
        timeout=10,
    )

    return response.json().get(
        "recommendations",
        []
    )


@st.cache_data(show_spinner=False)
def api_places() -> list:
    response = requests.get(
        f"{API_BASE}/api/recommend/places",
        timeout=10,
    )

    return response.json().get(
        "places",
        []
    )


@st.cache_data(show_spinner=False)
def api_model_info() -> dict:
    try:
        response = requests.get(
            f"{API_BASE}/api/model-info",
            timeout=10,
        )

        if response.status_code == 200:
            return response.json()

        return {
            "status": "unavailable"
        }

    except requests.RequestException:
        return {
            "status": "unavailable"
        }


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data(show_spinner=False)
def load_hotels() -> Optional[pd.DataFrame]:
    if os.path.exists(HOTELS_PATH):
        return pd.read_csv(HOTELS_PATH)

    return None


@st.cache_data(show_spinner=False)
def load_users() -> Optional[pd.DataFrame]:
    if os.path.exists(USERS_PATH):
        return pd.read_csv(USERS_PATH)

    return None


# =============================================================================
# MATPLOTLIB HELPERS
# =============================================================================

def apply_dark_chart_style(
    fig,
    ax,
):
    fig.patch.set_facecolor("#0a1020")
    ax.set_facecolor("#0a1020")

    ax.tick_params(
        colors="#8ea6c5",
        labelsize=9,
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.grid(
        True,
        axis="y",
        color="#20314c",
        alpha=0.45,
        linewidth=0.7,
    )

    ax.set_axisbelow(True)


def glowing_line_chart(
    labels,
    values,
    color="#3b8cff",
    ylabel="Average Price (USD)",
):
    fig, ax = plt.subplots(
        figsize=(10, 4.4),
        constrained_layout=True,
    )

    apply_dark_chart_style(fig, ax)

    x = list(range(len(labels)))

    # Glow layers
    for linewidth, alpha in [
        (14, 0.025),
        (10, 0.045),
        (7, 0.08),
        (4, 0.13),
    ]:
        ax.plot(
            x,
            values,
            linewidth=linewidth,
            color=color,
            alpha=alpha,
        )

    # Main line
    ax.plot(
        x,
        values,
        linewidth=2.4,
        color=color,
        marker="o",
        markersize=7,
        markerfacecolor="#071226",
        markeredgecolor="#69a9ff",
        markeredgewidth=1.8,
    )

    # Highlight line
    ax.plot(
        x,
        values,
        linewidth=1,
        color="#a9d2ff",
        alpha=0.8,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        rotation=28,
        ha="right",
        color="#8ea6c5",
        fontsize=8,
    )

    ax.set_ylabel(
        ylabel,
        color="#7892b3",
        fontsize=9,
    )

    ax.margins(x=0.02)

    return fig


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:

    if "page" not in st.session_state:
        st.session_state.page = "Flight Price Predictor"


    st.markdown('<div class="sidebar-brand">', unsafe_allow_html=True)

    st.image(
        str(LOGO_PATH),
        width=58,
    )

    st.markdown(
        """
        <div class="brand-name">Voyage Analytics</div>
        <div class="brand-subtitle">Travel Intelligence Platform</div>
        """,
    unsafe_allow_html=True,
    )

    st.markdown('</div>', unsafe_allow_html=True)


    render_html(
        """
        <div class="nav-label">
            NAVIGATION
        </div>
        """
    )


    if st.button(
        "Flight Price Predictor",
        use_container_width=True,
    ):
        st.session_state.page = "Flight Price Predictor"


    if st.button(
        "Gender Classifier",
        use_container_width=True,
    ):
        st.session_state.page = "Gender Classifier"


    if st.button(
        "Hotel Recommender",
        use_container_width=True,
    ):
        st.session_state.page = "Hotel Recommender"


    if st.button(
        "Insights & Analytics",
        use_container_width=True,
    ):
        st.session_state.page = "Insights"


    # Sidebar system status

    render_html(
        f"""
        <div class="sidebar-status">

            <div class="nav-label">
                SYSTEM STATUS
            </div>

            <div class="status-row">
                <div class="status-dot"></div>
                <span>
                    {"API Online" if api_ok() else "API Offline"}
                </span>
            </div>

            <div style="margin-top:18px;">
                <div class="sidebar-info-label">
                    Model
                </div>

                <div class="sidebar-info-value">
                    voyage_flight_price
                </div>
            </div>

            <div style="margin-top:14px;">
                <div class="sidebar-info-label">
                    Version
                </div>

                <div class="sidebar-info-value">
                    1.0
                </div>
            </div>

        </div>
        """
    )



page = st.session_state.page


# =============================================================================
# HERO
# =============================================================================

render_html(
    """
    <div class="hero-card">

        <div class="hero-title">
            Voyage Analytics
        </div>

        <div class="hero-subtitle">
            Intelligent travel predictions powered by machine learning
        </div>

    </div>
    """
)


# =============================================================================
# PAGE 1 FLIGHT PRICE PREDICTOR
# =============================================================================

if page == "Flight Price Predictor":

    render_html(
        """
        <div class="page-title">
            Flight Price Predictor
        </div>

        <div class="page-subtitle">
            Estimate flight prices using the trained XGBoost model.
        </div>

        <div class="section-line"></div>
        """
    )

    with st.form("flight_form"):

        col1, col2 = st.columns(2)

        with col1:

            f_from = st.text_input(
                "From",
                "Recife (PE)",
            )

            # IMPORTANT:
            # These values exactly match the API schema.
            f_type = st.selectbox(
                "Flight Type",
                [
                    "economic",
                    "firstClass",
                    "premium",
                ],
                index=0,
            )

            f_time = st.number_input(
                "Duration (hours)",
                min_value=0.0,
                max_value=24.0,
                value=1.76,
                step=0.01,
            )

            f_agency = st.text_input(
                "Agency",
                "FlyingDrops",
            )

        with col2:

            f_to = st.text_input(
                "To",
                "Florianopolis (SC)",
            )

            f_date = st.date_input(
                "Flight Date"
            )

            f_distance = st.number_input(
                "Distance (km)",
                min_value=0.0,
                max_value=20000.0,
                value=676.53,
                step=0.01,
            )

        submitted = st.form_submit_button(
            "Predict Flight Price",
            use_container_width=True,
        )


    if submitted:

        payload = {
            "from": f_from,
            "to": f_to,
            "flightType": f_type,
            "agency": f_agency,
            "time": float(f_time),
            "distance": float(f_distance),
            "flight_year": f_date.year,
            "flight_month": f_date.month,
            "flight_day": f_date.day,
            "flight_dayofweek": f_date.weekday(),
        }

        try:

            result = api_predict_flight(
                payload
            )

            price = result.get(
                "predicted_price"
            )

            if price is not None:

                render_html(
                    f"""
                    <div id="flight-result" class="result-card">

                        <div class="result-title">
                            Estimated Flight Price
                        </div>

                        <div class="result-value">
                            ${float(price):,.2f}
                        </div>

                    </div>
                    """
                )
                
            else:

                st.error(
                    result
                )

        except Exception as e:

            st.error(
                f"Prediction failed: {e}"
            )


# =============================================================================
# PAGE 2 GENDER CLASSIFIER
# =============================================================================

elif page == "Gender Classifier":

    render_html(
        """
        <div class="page-title">
            Gender Classifier
        </div>

        <div class="page-subtitle">
            Predict gender from user profile information.
        </div>

        <div class="section-line"></div>
        """
    )

    with st.form("gender_form"):

        g_name = st.text_input(
            "Full Name",
            "Robert Braun",
        )

        g_age = st.number_input(
            "Age",
            min_value=0,
            max_value=120,
            value=33,
        )

        g_company = st.text_input(
            "Company",
            "4You",
        )

        submitted = st.form_submit_button(
            "Classify Gender",
            use_container_width=True,
        )


    if submitted:

        try:

            result = api_predict_gender(
                {
                    "user name": g_name,
                    "age": int(g_age),
                    "company": g_company,
                }
            )

            gender = result.get(
                "gender"
            )

            probability = result.get(
                "probability",
                0,
            )

            if gender:

                render_html(
                    f"""
                    <div class="result-card">

                        <div class="result-title">
                            Predicted Gender
                        </div>

                        <div class="result-value">
                            {gender}
                        </div>

                        <div style="
                            color:#7fa7dc;
                            margin-top:8px;
                            font-size:13px;
                        ">
                            Confidence: {float(probability):.1%}
                        </div>

                    </div>
                    """
                )

            else:

                st.error(
                    result
                )

        except Exception as e:

            st.error(
                f"Classification failed: {e}"
            )


# =============================================================================
# PAGE 3 HOTEL RECOMMENDER
# =============================================================================

elif page == "Hotel Recommender":

    render_html(
        """
        <div class="page-title">
            Hotel Recommender
        </div>

        <div class="page-subtitle">
            Find personalized hotel recommendations based on your preferences.
        </div>

        <div class="section-line"></div>
        """
    )

    places = (
        api_places()
        if api_ok()
        else []
    )

    with st.form("recommend_form"):

        col1, col2 = st.columns(2)

        with col1:

            r_place = st.selectbox(
                "Place",
                ["", *places],
            )

            r_budget = st.number_input(
                "Max Price / Night",
                min_value=0.0,
                max_value=1000.0,
                value=250.0,
                step=10.0,
            )

        with col2:

            r_days = st.number_input(
                "Nights",
                min_value=0,
                max_value=30,
                value=3,
            )

            r_company = st.text_input(
                "Company",
                "4You",
            )

        r_top = st.slider(
            "Number of Results",
            min_value=1,
            max_value=10,
            value=5,
        )

        submitted = st.form_submit_button(
            "Get Recommendations",
            use_container_width=True,
        )


    if submitted:

        payload = {}

        if r_place:
            payload["place"] = r_place

        if r_budget > 0:
            payload["max_price_per_day"] = float(
                r_budget
            )

        if r_days > 0:
            payload["days"] = int(
                r_days
            )

        if r_company:
            payload["company"] = r_company

        payload["top_n"] = int(
            r_top
        )

        try:

            recs = api_recommend(
                payload
            )

            if recs:

                render_html(
                    """
                    <div class="page-title"
                         style="font-size:24px;margin-top:28px;">
                        Recommended Hotels
                    </div>
                    """
                )

                for recommendation in recs:

                    total = ""

                    if recommendation.get(
                        "total_cost"
                    ):

                        total = (
                            f" ┬╖ Total for stay: "
                            f"${recommendation['total_cost']:,.2f}"
                        )

                    with st.container(
                        border=True
                    ):

                        render_html(
                            f"""
                            <div style="
                                color:#f1f6ff;
                                font-size:16px;
                                font-weight:750;
                            ">
                                {recommendation['hotel_name']}
                            </div>

                            <div style="
                                color:#8da6c5;
                                margin-top:5px;
                                font-size:13px;
                            ">
                                {recommendation['place']}
                                ┬╖
                                ${recommendation['price_per_day']:,.2f}/night
                                {total}
                            </div>
                            """
                        )

                        st.progress(
                            recommendation["score"] / 100,
                            text=(
                                f"Match score: "
                                f"{recommendation['score']}/100"
                            ),
                        )

                        st.caption(
                            recommendation["reason"]
                        )

            else:

                st.info(
                    "No recommendations returned."
                )

        except Exception as e:

            st.error(
                f"Recommendation failed: {e}"
            )


# =============================================================================
# PAGE 4 INSIGHTS & ANALYTICS
# =============================================================================

else:

    hotel_df = load_hotels()
    user_df = load_users()

    render_html(
        """
        <div class="page-title">
            Insights & Analytics
        </div>

        <div class="page-subtitle">
            Explore travel, hotel and user data through interactive analytics.
        </div>

        <div class="section-line"></div>
        """
    )


    # -------------------------------------------------------------------------
    # KPI METRICS
    # -------------------------------------------------------------------------

    if (
        hotel_df is not None
        and user_df is not None
    ):

        hotel_records = len(
            hotel_df
        )

        user_records = len(
            user_df
        )

        destination_count = hotel_df[
            "place"
        ].nunique()

        average_price = hotel_df[
            "price"
        ].mean()


        metric_cols = st.columns(
            4
        )


        with metric_cols[0]:

            render_html(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        Hotel Records
                    </div>

                    <div class="metric-value">
                        {hotel_records:,}
                    </div>

                </div>
                """
            )


        with metric_cols[1]:

            render_html(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        Users
                    </div>

                    <div class="metric-value">
                        {user_records:,}
                    </div>

                </div>
                """
            )


        with metric_cols[2]:

            render_html(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        Destinations
                    </div>

                    <div class="metric-value">
                        {destination_count:,}
                    </div>

                </div>
                """
            )


        with metric_cols[3]:

            render_html(
                f"""
                <div class="metric-card">

                    <div class="metric-label">
                        Avg. Hotel Price
                    </div>

                    <div class="metric-value">
                        ${average_price:,.0f}
                    </div>

                </div>
                """
            )


        st.markdown(
            "<div style='height:30px'></div>",
            unsafe_allow_html=True,
        )


        # ---------------------------------------------------------------------
        # ROW 1 ΓÇö PRICE BY DESTINATION
        # ---------------------------------------------------------------------

        destination_prices = (
            hotel_df
            .groupby("place")["price"]
            .mean()
            .sort_values(
                ascending=False
            )
        )

        labels = destination_prices.index.tolist()
        values = destination_prices.values.tolist()

        fig = glowing_line_chart(
            labels,
            values,
            color="#3b8cff",
            ylabel="Average Price (USD)",
        )

        render_html(
            """
            <div class="chart-panel">

                <div class="chart-title">
                    Average Hotel Price by Destination
                </div>

                <div class="chart-subtitle">
                    Average hotel price across available destinations
                </div>

            </div>
            """
        )

        st.pyplot(
            fig,
            use_container_width=True,
        )

        plt.close(fig)


        # ---------------------------------------------------------------------
        # ROW 2 ΓÇö HOTEL PRICE + GENDER
        # ---------------------------------------------------------------------

        left_col, right_col = st.columns(
            2
        )


        with left_col:

            render_html(
                """
                <div class="chart-panel">

                    <div class="chart-title">
                        Hotel Price Trend Over Time
                    </div>

                    <div class="chart-subtitle">
                        Average hotel prices over time
                    </div>

                </div>
                """
            )

            temp_hotels = hotel_df.copy()

            if "date" in temp_hotels.columns:

                temp_hotels["date"] = pd.to_datetime(
                    temp_hotels["date"],
                    errors="coerce",
                )

                monthly_prices = (
                    temp_hotels
                    .dropna(subset=["date"])
                    .groupby(
                        temp_hotels["date"].dt.to_period("M")
                    )["price"]
                    .mean()
                )

                if len(monthly_prices) > 0:

                    monthly_labels = [
                        str(x)
                        for x in monthly_prices.index
                    ]

                    monthly_values = (
                        monthly_prices.values
                    )

                    fig2 = glowing_line_chart(
                        monthly_labels,
                        monthly_values,
                        color="#805cff",
                        ylabel="Average Price (USD)",
                    )

                    st.pyplot(
                        fig2,
                        use_container_width=True,
                    )

                    plt.close(fig2)

            else:

                st.info(
                    "Date information is not available."
                )


        with right_col:

            render_html(
                """
                <div class="chart-panel">

                    <div class="chart-title">
                        User Gender Distribution
                    </div>

                    <div class="chart-subtitle">
                        Distribution of gender labels in the user dataset
                    </div>

                </div>
                """
            )


            # -----------------------------------------------------------------
            # KEEP THIS AS A PIE / DONUT CHART
            # -----------------------------------------------------------------

            gender_counts = (
                user_df["gender"]
                .fillna("None")
                .astype(str)
                .replace("", "None")
                .value_counts()
            )

            if len(gender_counts) > 0:

                fig3, ax3 = plt.subplots(
                    figsize=(6, 4.4),
                    constrained_layout=True,
                )

                fig3.patch.set_facecolor(
                    "#0a1020"
                )

                ax3.set_facecolor(
                    "#0a1020"
                )

                values_gender = (
                    gender_counts.values
                )

                labels_gender = (
                    gender_counts.index
                )

                # Fixed colors only for the donut.
                # This intentionally stays a PIE/DONUT chart.
                donut_colors = [
                    "#3b8cff",
                    "#9b6cff",
                    "#26e0c1",
                    "#ff5ca8",
                    "#f3b84b",
                ]

                selected_colors = [
                    donut_colors[i % len(donut_colors)]
                    for i in range(
                        len(values_gender)
                    )
                ]

                wedges, texts, autotexts = ax3.pie(
                    values_gender,
                    labels=None,
                    colors=selected_colors,
                    startangle=90,
                    counterclock=False,
                    autopct=lambda p: (
                        f"{p:.0f}%"
                        if p >= 3
                        else ""
                    ),
                    pctdistance=0.76,
                    wedgeprops={
                        "width": 0.40,
                        "edgecolor": "#0a1020",
                        "linewidth": 3,
                    },
                    textprops={
                        "color": "#dbe7ff",
                        "fontsize": 9,
                    },
                )

                for autotext in autotexts:

                    autotext.set_color(
                        "#ffffff"
                    )

                    autotext.set_fontweight(
                        "bold"
                    )

                # Center text
                ax3.text(
                    0,
                    0.06,
                    f"{len(user_df):,}",
                    ha="center",
                    va="center",
                    color="#f4f7ff",
                    fontsize=20,
                    fontweight="bold",
                )

                ax3.text(
                    0,
                    -0.12,
                    "Users",
                    ha="center",
                    va="center",
                    color="#8099bb",
                    fontsize=9,
                )

                ax3.legend(
                    wedges,
                    [
                        f"{label} "
                        f"{count / len(user_df) * 100:.1f}%"
                        for label, count
                        in gender_counts.items()
                    ],
                    loc="center left",
                    bbox_to_anchor=(
                        1.0,
                        0.5,
                    ),
                    frameon=False,
                    labelcolor="#dbe7ff",
                    fontsize=9,
                )

                ax3.set_aspect(
                    "equal"
                )

                st.pyplot(
                    fig3,
                    use_container_width=True,
                )

                plt.close(fig3)


        # ---------------------------------------------------------------------
        # ---------------------------------------------------------------------
        # ROW 3 ΓÇö BOOKINGS BY DESTINATION
        # ---------------------------------------------------------------------


        render_html(
            """
            <div class="chart-panel">

                <div class="chart-title">
                    Bookings by Destination
                </div>

                <div class="chart-subtitle">
                    Total hotel records per destination
                </div>

            </div>
            """
        )


        booking_counts = (
            hotel_df["place"]
            .value_counts()
            .head(10)
            .sort_values()
        )

        fig5, ax5 = plt.subplots(
            figsize=(7, 4.8),
            constrained_layout=True,
        )

        fig5.patch.set_facecolor(
            "#0a1020"
        )

        ax5.set_facecolor(
            "#0a1020"
        )

        y = range(
            len(booking_counts)
        )

        ax5.barh(
            y,
            booking_counts.values,
            color="#3b8cff",
            alpha=0.82,
        )

        ax5.set_yticks(
            list(y)
        )

        ax5.set_yticklabels(
            booking_counts.index,
            color="#9bb0cd",
            fontsize=8,
        )

        ax5.tick_params(
            axis="x",
            colors="#8099bb",
            labelsize=8,
        )

        ax5.grid(
            True,
            axis="x",
            color="#20314c",
            alpha=0.4,
        )

        for spine in ax5.spines.values():
            spine.set_visible(False)

        ax5.set_xlabel(
            "Number of Hotels",
            color="#7892b3",
            fontsize=9,
        )

        for index, value in enumerate(
            booking_counts.values
        ):

            ax5.text(
                value + max(
                    booking_counts.values
                ) * 0.015,
                index,
                f"{value:,}",
                va="center",
                color="#dbe7ff",
                fontsize=8,
            )

        st.pyplot(
            fig5,
            use_container_width=True,
        )

        plt.close(fig5)


    # ---------------------------------------------------------------------   
    #      # DATASET PREVIEW
    # ---------------------------------------------------------------------

        with st.expander(
            "Dataset Preview"
        ):

            render_html(
                """
                <div style="
                    color:#829abc;
                    font-size:12px;
                    margin-bottom:12px;
                ">
                    View raw dataset samples
                </div>
                """
            )

            preview_col1, preview_col2 = st.columns(
                2
            )

            with preview_col1:

                st.markdown(
                    "### Hotels"
                )

                st.dataframe(
                    hotel_df.head(10),
                    use_container_width=True,
                )

            with preview_col2:

                st.markdown(
                    "### Users"
                )

                st.dataframe(
                    user_df.head(10),
                    use_container_width=True,
                )


    # -------------------------------------------------------------------------
    # DATA UNAVAILABLE FALLBACK
    # -------------------------------------------------------------------------

    else:

        st.info(
            "Rich local analytics requires the datasets. "
            "Set HOTELS_DATA_PATH and USERS_DATA_PATH "
            "or place hotels.csv/users.csv under artifacts/data/."
        )

        if os.path.exists(
            CATALOG_PATH
        ):

            try:

                with open(
                    CATALOG_PATH,
                    encoding="utf-8",
                ) as file:

                    catalog = json.load(
                        file
                    )

                render_html(
                    """
                    <div class="page-title"
                         style="font-size:25px;">
                        Hotel Catalog
                    </div>
                    """
                )

                rows = [
                    {
                        "hotel": hotel_name,
                        "place": hotel_data[
                            "place"
                        ],
                        "price": hotel_data[
                            "price_per_day"
                        ],
                    }

                    for hotel_name, hotel_data
                    in catalog.get(
                        "hotels",
                        {}
                    ).items()
                ]

                st.dataframe(
                    pd.DataFrame(rows),
                    use_container_width=True,
                )

            except Exception as e:

                st.error(
                    f"Unable to load hotel catalog: {e}"
                )

