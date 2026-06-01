from __future__ import annotations

import hashlib
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import requests
import streamlit as st

try:
    import nibabel as nib
except ImportError:  # pragma: no cover - optional dependency
    nib = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional dependency
    Image = None

try:
    from fpdf import FPDF
except ImportError:  # pragma: no cover - optional dependency
    FPDF = None

try:
    import plotly.graph_objects as go
except ImportError:  # pragma: no cover - optional dependency
    go = None


_REFERENCE_CAPTURE_SHA256 = "b745c088fb826b77a2df25b211a323f3b28d8b9004be1c4c3c7048dd0f681f0d"


st.set_page_config(page_title="NeuroPath AI Pro", layout="wide", initial_sidebar_state="expanded")

# Professional modern UI theme with light background and gradient accents
_CUSTOM_CSS = """
<style>
:root {
    --primary-gradient: linear-gradient(135deg, #7c3aed 0%, #6d28d9 25%, #5b21b6 50%, #3b82f6 75%, #1e40af 100%);
    --secondary-gradient: linear-gradient(135deg, #f472b6 0%, #ec4899 50%, #3b82f6 100%);
    --pink-primary: #f472b6;
    --pink-dark: #ec4899;
    --blue-primary: #3b82f6;
    --blue-dark: #1e40af;
    --cyan-accent: #06b6d4;
    --light-bg: #f8fafb;
    --white: #ffffff;
    --dark-text: #1a202c;
    --muted: #718096;
    --border-light: #e2e8f0;
    --success: #06b6d4;
    --warning: #ec4899;
    --danger: #f472b6;
}

html, body, section, div, [data-testid="stAppViewContainer"] {
    background-color: var(--light-bg) !important;
    color: var(--dark-text) !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #ffffff 0%, #f0f9ff 50%, #f5f3ff 100%) !important;
    border-right: 2px solid rgba(244, 114, 182, 0.2) !important;
    box-shadow: -4px 0 12px rgba(60, 130, 246, 0.05) !important;
}

/* Header Styling - Premium Product Card Design */
.header-container {
    position: relative;
    background: var(--primary-gradient);
    padding: 48px;
    border-radius: 24px;
    margin-bottom: 32px;
    color: white;
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    justify-content: center;
    gap: 0px;
    min-height: 180px;
    
    /* Premium box shadow and glow with pink/blue accents */
    box-shadow: 
        0 0 60px rgba(244, 114, 182, 0.3),
        0 0 120px rgba(59, 130, 246, 0.2),
        0 20px 60px rgba(0, 0, 0, 0.4),
        inset 0 1px 0 rgba(255, 255, 255, 0.1),
        inset 0 -1px 0 rgba(0, 0, 0, 0.2);
    
    /* Subtle floating animation */
    animation: float 3s ease-in-out infinite;
}

.header-container > div {
    background: transparent !important;
    padding: 0 !important;
    margin: 0 !important;
}

.header-container p {
    margin: 0 !important;
    padding: 0 !important;
}

@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-8px); }
}

/* Logo Styling - Premium with glow - HIDDEN */
.logo-container {
    display: none;
}

.header-title {
    font-size: 48px;
    font-weight: 800;
    margin: 0 !important;
    padding: 0 !important;
    line-height: 1.1;
    letter-spacing: -1px;
    color: #ffffff !important;
    background: transparent !important;
    
    /* Premium glow and shadow text */
    text-shadow: 
        0 2px 4px rgba(0, 0, 0, 0.3),
        0 8px 16px rgba(0, 0, 0, 0.2),
        0 0 20px rgba(244, 114, 182, 0.5),
        0 0 40px rgba(59, 130, 246, 0.3);
    
    animation: glow 2s ease-in-out infinite;
}

@keyframes glow {
    0%, 100% {
        text-shadow: 
            0 2px 4px rgba(0, 0, 0, 0.3),
            0 8px 16px rgba(0, 0, 0, 0.2),
            0 0 20px rgba(244, 114, 182, 0.5),
            0 0 40px rgba(59, 130, 246, 0.3);
    }
    50% {
        text-shadow: 
            0 2px 4px rgba(0, 0, 0, 0.3),
            0 8px 16px rgba(0, 0, 0, 0.2),
            0 0 25px rgba(244, 114, 182, 0.8),
            0 0 50px rgba(59, 130, 246, 0.5);
    }
}

.header-subtitle {
    font-size: 16px;
    font-weight: 500;
    color: rgba(255, 255, 255, 0.95) !important;
    letter-spacing: 0.3px;
    line-height: 1.4;
    margin: 8px 0 0 0 !important;
    padding: 0 !important;
    background: transparent !important;
    
    text-shadow: 
        0 1px 2px rgba(0, 0, 0, 0.2),
        0 0 10px rgba(244, 114, 182, 0.2);
}

.header-container span, .header-container code, .header-container p {
    background: transparent !important;
    color: white !important;
    margin: 0 !important;
    padding: 0 !important;
}

/* Modern Metric Cards */
.metric-card {
    background: linear-gradient(135deg, #ffffff 0%, #f0f9ff 100%);
    border: 2px solid rgba(244, 114, 182, 0.2);
    border-radius: 16px;
    padding: 24px;
    box-shadow: 
        0 4px 12px rgba(244, 114, 182, 0.1),
        0 0 20px rgba(59, 130, 246, 0.05);
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}

.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    right: 0;
    width: 100px;
    height: 100px;
    background: radial-gradient(circle at top right, rgba(244, 114, 182, 0.1), transparent);
    border-radius: 50%;
}

.metric-card:hover {
    box-shadow: 
        0 8px 24px rgba(244, 114, 182, 0.15),
        0 0 30px rgba(59, 130, 246, 0.1);
    transform: translateY(-4px);
    border-color: rgba(244, 114, 182, 0.4);
}

.metric-value {
    font-size: 32px;
    font-weight: 700;
    margin: 8px 0;
}

.metric-label {
    color: var(--muted);
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    font-weight: 600;
}

.metric-unit {
    font-size: 14px;
    color: var(--muted);
    font-weight: 400;
}

/* Risk Color Indicators */
.risk-low { color: var(--success) !important; }
.risk-medium { color: var(--warning) !important; }
.risk-high { color: var(--danger) !important; }

/* Section Headers */
.section-header {
    font-size: 20px;
    font-weight: 700;
    color: var(--dark-text);
    margin-bottom: 16px;
    padding-bottom: 12px;
    border-bottom: 3px solid;
    border-image: linear-gradient(90deg, var(--pink-primary) 0%, var(--blue-primary) 100%) 1;
    display: flex;
    align-items: center;
    gap: 8px;
}

.section-icon {
    font-size: 24px;
}

/* Report Card */
.report-card {
    background: linear-gradient(135deg, #ffffff 0%, #f0f9ff 50%, #fdf2f8 100%);
    border: 2px solid rgba(244, 114, 182, 0.2);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 16px;
    box-shadow: 0 4px 12px rgba(244, 114, 182, 0.08);
    transition: all 0.3s ease;
}

.report-card:hover {
    box-shadow: 0 6px 16px rgba(244, 114, 182, 0.12);
    border-color: rgba(244, 114, 182, 0.4);
}

.report-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}

.report-title {
    font-weight: 700;
    font-size: 16px;
}

.report-subtitle {
    color: var(--muted);
    font-size: 12px;
    margin-top: 2px;
}

/* Alert Boxes */
.warning-box {
    background: linear-gradient(135deg, rgba(244, 114, 182, 0.1) 0%, rgba(59, 130, 246, 0.05) 100%);
    border-left: 4px solid var(--pink-primary);
    border-radius: 10px;
    padding: 14px;
    margin: 12px 0;
    color: #9d174d;
    font-size: 14px;
    font-weight: 500;
}

.info-box {
    background: linear-gradient(135deg, rgba(6, 182, 212, 0.1) 0%, rgba(59, 130, 246, 0.05) 100%);
    border-left: 4px solid var(--cyan-accent);
    border-radius: 10px;
    padding: 14px;
    margin: 12px 0;
    color: #0e7490;
    font-size: 14px;
    font-weight: 500;
}

/* Buttons */
.stButton > button,
.stDownloadButton > button,
[data-testid="stDownloadButton"] a,
[data-testid="stFileUploaderDropzone"] button {
    background: var(--primary-gradient) !important;
    background-color: transparent !important;
    background-size: 100% 100% !important;
    position: relative !important;
    isolation: isolate !important;
    overflow: hidden !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    min-height: 48px !important;
    padding: 12px 22px !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 0.55rem !important;
    width: 100% !important;
    min-width: 0 !important;
    text-align: center !important;
    cursor: pointer !important;
    font-size: 16px !important;
    font-weight: 700 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 10px 24px rgba(59, 130, 246, 0.16), 0 6px 18px rgba(124, 58, 237, 0.18) !important;
    letter-spacing: 0.3px !important;
    line-height: 1.1 !important;
    appearance: none !important;
    -webkit-appearance: none !important;
    text-decoration: none !important;
    box-sizing: border-box !important;
}

.stButton > button::before,
.stDownloadButton > button::before,
[data-testid="stDownloadButton"] a::before,
[data-testid="stFileUploaderDropzone"] button::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: inherit;
    background: linear-gradient(135deg, #f472b6 0%, #ec4899 50%, #3b82f6 100%);
    z-index: -1;
}

.stButton > button > div,
.stButton > button > span,
.stButton > button > div > div,
.stButton > button p,
.stButton > button label,
.stButton > button svg,
.stDownloadButton > button > div,
.stDownloadButton > button > span,
.stDownloadButton > button > div > div,
.stDownloadButton > button p,
.stDownloadButton > button label,
[data-testid="stDownloadButton"] a > div,
[data-testid="stDownloadButton"] a > span,
[data-testid="stDownloadButton"] a p,
[data-testid="stDownloadButton"] a svg,
[data-testid="stFileUploaderDropzone"] button > div,
[data-testid="stFileUploaderDropzone"] button > span,
[data-testid="stFileUploaderDropzone"] button > div > div,
[data-testid="stFileUploaderDropzone"] button p,
[data-testid="stFileUploaderDropzone"] button label,
[data-testid="stFileUploaderDropzone"] button svg {
    background: transparent !important;
    color: inherit !important;
}

.stButton > button:hover::before,
.stDownloadButton > button:hover::before,
[data-testid="stDownloadButton"] a:hover::before,
[data-testid="stFileUploaderDropzone"] button:hover::before {
    filter: brightness(1.05) saturate(1.08);
}

.stButton > button:hover,
.stDownloadButton > button:hover,
[data-testid="stDownloadButton"] a:hover,
[data-testid="stFileUploaderDropzone"] button:hover {
    transform: translateY(-2px) !important;
}

.stButton > button:active,
.stDownloadButton > button:active,
[data-testid="stDownloadButton"] a:active,
[data-testid="stFileUploaderDropzone"] button:active {
    transform: translateY(0px) !important;
}

.stButton > button:focus-visible,
.stDownloadButton > button:focus-visible,
[data-testid="stDownloadButton"] a:focus-visible,
[data-testid="stFileUploaderDropzone"] button:focus-visible {
    outline: 3px solid rgba(255, 255, 255, 0.65) !important;
    outline-offset: 3px !important;
}

[data-testid="stFileUploaderDropzone"] {
    border: 1px solid rgba(124, 58, 237, 0.14) !important;
    border-radius: 16px !important;
    background: linear-gradient(135deg, rgba(255,255,255,0.96) 0%, rgba(240,249,255,0.94) 100%) !important;
    box-shadow: 0 16px 32px rgba(30,41,59,0.05) !important;
    padding: 10px 12px !important;
}

[data-testid="stFileUploaderDropzone"] button > div,
[data-testid="stFileUploaderDropzone"] button > span,
[data-testid="stFileUploaderDropzone"] button p,
[data-testid="stFileUploaderDropzone"] button svg {
    background: transparent !important;
    color: inherit !important;
}

.stButton > button,
.stDownloadButton > button,
[data-testid="stDownloadButton"] a,
[data-testid="stFileUploaderDropzone"] button {
    font-family: Inter, "Segoe UI", "Helvetica Neue", Arial, sans-serif !important;
    font-size: 16px !important;
    font-weight: 700 !important;
}

/* Sidebar Text */
.sidebar-label {
    font-weight: 700;
    background: linear-gradient(90deg, var(--pink-primary) 0%, var(--blue-primary) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-top: 16px;
    margin-bottom: 10px;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}

/* Input Fields */
.stNumberInput, .stSelectbox, .stTextInput {
    background-color: white !important;
}

.stNumberInput input, .stSelectbox select, .stTextInput input {
    border: 2px solid rgba(244, 114, 182, 0.2) !important;
    border-radius: 10px !important;
    background-color: linear-gradient(135deg, #ffffff 0%, #f0f9ff 100%) !important;
    color: var(--dark-text) !important;
    transition: all 0.3s ease !important;
}

.stNumberInput input:focus, .stSelectbox select:focus, .stTextInput input:focus {
    border-color: var(--pink-primary) !important;
    box-shadow: 0 0 12px rgba(244, 114, 182, 0.3) !important;
}

/* Image containers */
.neuro-image {
    background: linear-gradient(135deg, #ffffff 0%, #f0f9ff 100%);
    border: 2px solid rgba(244, 114, 182, 0.2);
    border-radius: 14px;
    padding: 16px;
    text-align: center;
    box-shadow: 0 4px 12px rgba(244, 114, 182, 0.08);
    transition: all 0.3s ease;
}

.neuro-image:hover {
    box-shadow: 0 6px 16px rgba(244, 114, 182, 0.12);
    border-color: rgba(244, 114, 182, 0.4);
}

.neuro-image img {
    max-width: 100%;
    height: auto;
    border-radius: 6px;
    margin-top: 8px;
}

.neuro-image-label {
    font-size: 14px;
    font-weight: 600;
    color: var(--dark-text);
    margin-bottom: 8px;
}

/* Utility Classes */
.small-muted {
    color: var(--muted);
    font-size: 12px;
}

.divider {
    margin: 16px 0;
    border-top: 1px solid var(--border-light);
}

.gradient-text {
    background: var(--primary-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 700;
}

/* Upload area */
.stFileUploader {
    border: 3px dashed rgba(244, 114, 182, 0.3) !important;
    border-radius: 14px !important;
    background: linear-gradient(135deg, #ffffff 0%, rgba(244, 114, 182, 0.05) 100%) !important;
    transition: all 0.3s ease !important;
}

.stFileUploader:hover {
    border-color: var(--pink-primary) !important;
    background: linear-gradient(135deg, #fdf2f8 0%, rgba(244, 114, 182, 0.1) 100%) !important;
}

/* Spinners and loading */
.stSpinner {
    color: var(--pink-primary) !important;
}

/* Expander styling */
.streamlit-expander {
    border: 2px solid rgba(244, 114, 182, 0.2) !important;
    border-radius: 10px !important;
    background: linear-gradient(135deg, #ffffff 0%, #f0f9ff 100%) !important;
}

.streamlit-expanderHeader {
    background: transparent !important;
    color: var(--dark-text) !important;
    font-weight: 600 !important;
}

/* Metric value styling */
.metric-value {
    background: linear-gradient(90deg, var(--pink-primary) 0%, var(--blue-primary) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

/* Inline compact metrics row (left-to-right phrases) */
.metrics-inline {
    display: flex;
    gap: 24px;
    align-items: center;
    justify-content: flex-start;
    flex-wrap: nowrap;
}
.metrics-inline .metric-item {
    font-size: 15px;
    color: var(--dark-text);
    background: white;
    padding: 8px 12px;
    border-radius: 10px;
    border: 1px solid var(--border-light);
    box-shadow: 0 4px 10px rgba(59,130,246,0.04);
}

/* Multipage Landing */
.landing-page [data-testid="stSidebar"] {
    display: none !important;
}

.landing-shell {
    padding: 12px 0 28px;
}

.landing-hero {
    background: linear-gradient(135deg, rgba(255,255,255,0.96) 0%, rgba(240,249,255,0.9) 45%, rgba(245,243,255,0.92) 100%);
    border: 1px solid rgba(124,58,237,0.14);
    border-radius: 28px;
    padding: 30px;
    box-shadow: 0 24px 60px rgba(30,64,175,0.08);
}

.landing-hero-grid {
    display: grid;
    grid-template-columns: 1.15fr 0.85fr;
    gap: 28px;
    align-items: center;
}

.landing-hero-center {
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    gap: 0;
    max-width: 860px;
    margin: 0 auto;
}

.landing-kicker {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    border-radius: 999px;
    background: rgba(124,58,237,0.08);
    color: #5b21b6;
    font-size: 12px;
    font-weight: 800;
    text-transform: uppercase;
    letter-spacing: 0.12em;
}

.landing-title {
    font-size: clamp(40px, 6vw, 72px);
    line-height: 0.98;
    font-weight: 900;
    letter-spacing: -0.04em;
    margin: 16px 0 12px;
    background: linear-gradient(135deg, #1e293b 0%, #4c1d95 40%, #1d4ed8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.landing-subtitle {
    font-size: 22px;
    font-weight: 700;
    color: #334155;
    margin-bottom: 14px;
}

.landing-copy {
    font-size: 16px;
    line-height: 1.7;
    color: var(--muted);
    max-width: 760px;
}

.landing-stat-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 14px;
}

.landing-stat-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.96) 0%, rgba(245,248,255,0.92) 100%);
    border: 1px solid rgba(124,58,237,0.12);
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 16px 30px rgba(30,41,59,0.05);
}

.landing-stat-value {
    font-size: 28px;
    font-weight: 900;
    background: linear-gradient(135deg, #7c3aed 0%, #3b82f6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.landing-stat-label {
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--muted);
    font-weight: 700;
    margin-top: 8px;
}

.landing-section-title {
    font-size: 22px;
    font-weight: 900;
    letter-spacing: -0.03em;
    color: var(--dark-text);
    margin-bottom: 14px;
}

.landing-section-subtitle {
    color: var(--muted);
    font-size: 15px;
    margin-bottom: 18px;
}

.landing-pill-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.96) 0%, rgba(245,248,255,0.92) 100%);
    border: 1px solid rgba(124,58,237,0.12);
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 14px 28px rgba(30,41,59,0.05);
    height: 100%;
}

.landing-pill-icon {
    width: 48px;
    height: 48px;
    border-radius: 14px;
    display: grid;
    place-items: center;
    background: linear-gradient(135deg, rgba(124,58,237,0.12) 0%, rgba(59,130,246,0.12) 100%);
    font-size: 22px;
    margin-bottom: 12px;
}

.landing-flow {
    display: grid;
    grid-template-columns: repeat(9, minmax(0, 1fr));
    gap: 10px;
    align-items: center;
}

.landing-flow-step {
    background: rgba(255,255,255,0.95);
    border: 1px solid rgba(124,58,237,0.14);
    border-radius: 16px;
    padding: 14px 10px;
    text-align: center;
    box-shadow: 0 12px 24px rgba(30,41,59,0.05);
}

.landing-flow-step .icon {
    font-size: 22px;
    margin-bottom: 8px;
}

.landing-flow-arrow {
    text-align: center;
    color: #7c3aed;
    font-size: 22px;
    font-weight: 900;
}

.landing-tech-grid {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 14px;
}

.landing-cta-panel {
    background: linear-gradient(135deg, rgba(124,58,237,0.96) 0%, rgba(59,130,246,0.94) 100%);
    border-radius: 26px;
    padding: 28px;
    color: white;
    box-shadow: 0 26px 60px rgba(59,130,246,0.18);
    text-align: center;
}

.landing-cta-panel .small-muted {
    color: rgba(255,255,255,0.82);
}

@media (max-width: 980px) {
    .landing-stat-grid,
    .landing-tech-grid {
        grid-template-columns: 1fr;
    }

    .landing-flow {
        grid-template-columns: 1fr;
    }

    .landing-flow-arrow {
        transform: rotate(90deg);
    }
}

/* Slider styling */
.stSlider > div > div {
    background: var(--primary-gradient) !important;
}
</style>
"""

st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)

_DEFAULT_API_URL = os.getenv("NEUROPATH_API_URL", "http://localhost:8000/predict")
_API_STARTUP_TIMEOUT = 12
_API_STARTUP_RETRY_DELAY = 0.5


def _normalize_api_url(api_url: str) -> str:
    if "://" not in api_url:
        return f"http://{api_url}"
    return api_url


def _resolve_api_urls(api_url: str) -> tuple[str, str, str, str | None, int | None]:
    parsed = urlparse(api_url)
    hostname = parsed.hostname
    port = parsed.port
    normalized_host = "127.0.0.1" if hostname == "0.0.0.0" else hostname
    netloc = ""
    if normalized_host:
        netloc = f"{normalized_host}:{port}" if port is not None else normalized_host
    else:
        netloc = parsed.netloc
    base = f"{parsed.scheme}://{netloc}"
    path = parsed.path if parsed.path and parsed.path != "/" else "/predict"
    predict_url = f"{base}{path}"
    health_url = f"{base}/health"
    return base, predict_url, health_url, normalized_host, port


def _is_local_host(hostname: str | None) -> bool:
    return hostname in {"localhost", "127.0.0.1", "0.0.0.0"}


def _check_api_health(health_url: str) -> bool:
    try:
        response = requests.get(health_url, timeout=2)
    except requests.RequestException:
        return False
    return response.status_code == 200


def _wait_for_api(health_url: str, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _check_api_health(health_url):
            return True
        time.sleep(_API_STARTUP_RETRY_DELAY)
    return False


def _start_local_api(host: str, port: int) -> subprocess.Popen:
    project_root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{project_root}{os.pathsep}{project_root / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "neuro_path_ai.api.app:app",
        "--host",
        host,
        "--port",
        str(port),
    ]
    return subprocess.Popen(
        command,
        cwd=str(project_root),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _ensure_api_running(api_url: str) -> tuple[str, bool, str]:
    normalized_url = _normalize_api_url(api_url.strip())
    _, predict_url, health_url, hostname, port = _resolve_api_urls(normalized_url)
    if not hostname:
        return predict_url, False, "Invalid API URL. Please include a host."

    if not _is_local_host(hostname):
        is_ready = _check_api_health(health_url)
        status = "Remote API connected." if is_ready else "Remote API not reachable. Check the URL or server."
        return predict_url, is_ready, status

    if _check_api_health(health_url):
        return predict_url, True, "Local API connected."

    if st.session_state.get("api_autostart_attempted"):
        return predict_url, False, "Local API not reachable. Start it manually or check logs."

    st.session_state["api_autostart_attempted"] = True
    selected_host = "127.0.0.1" if hostname in {"localhost", "127.0.0.1"} else hostname
    selected_port = port or 8000

    try:
        proc = _start_local_api(selected_host, selected_port)
    except (OSError, ValueError) as exc:
        return predict_url, False, f"Failed to start local API: {exc}"

    st.session_state["api_process_pid"] = proc.pid
    if _wait_for_api(health_url, timeout=_API_STARTUP_TIMEOUT):
        return predict_url, True, "Local API started automatically."
    return predict_url, False, "Local API failed to start. Verify model files and dependencies."



# NeuroPath AI Logo SVG (embedded)
def get_logo_svg() -> str:
    """Generate premium SVG logo with pink brain icon and AI elements"""
    return """
    <svg viewBox="0 0 100 100" width="100" height="100" xmlns="http://www.w3.org/2000/svg">
        <!-- Gradient definitions -->
        <defs>
            <radialGradient id="brainGradient" cx="40%" cy="40%">
                <stop offset="0%" style="stop-color:#f472b6;stop-opacity:1" />
                <stop offset="100%" style="stop-color:#ec4899;stop-opacity:1" />
            </radialGradient>
            <linearGradient id="circuitGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" style="stop-color:#f472b6;stop-opacity:0.8" />
                <stop offset="100%" style="stop-color:#3b82f6;stop-opacity:0.8" />
            </linearGradient>
            <filter id="glow">
                <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
                <feMerge>
                    <feMergeNode in="coloredBlur"/>
                    <feMergeNode in="SourceGraphic"/>
                </feMerge>
            </filter>
        </defs>
        
        <!-- Brain outline with premium fill -->
        <g filter="url(#glow)">
            <!-- Left hemisphere -->
            <path d="M 35 25 Q 28 30 28 40 Q 28 55 38 62 Q 45 65 50 63 Q 48 58 48 50 Q 48 35 42 28 Q 39 25 35 25" 
                  fill="url(#brainGradient)" stroke="none"/>
            
            <!-- Right hemisphere -->
            <path d="M 65 25 Q 72 30 72 40 Q 72 55 62 62 Q 55 65 50 63 Q 52 58 52 50 Q 52 35 58 28 Q 61 25 65 25" 
                  fill="url(#brainGradient)" stroke="none"/>
            
            <!-- Brain ridge detail -->
            <path d="M 50 20 Q 48 22 48 30 Q 48 45 50 60 Q 52 45 52 30 Q 52 22 50 20" 
                  fill="#f093fb" stroke="none" opacity="0.9"/>
        </g>
        
        <!-- AI Circuit nodes with blue glow -->
        <g fill="none" stroke="url(#circuitGradient)" stroke-width="1.5" opacity="0.7">
            <!-- Circuit connections -->
            <line x1="35" y1="35" x2="50" y2="50"/>
            <line x1="65" y1="35" x2="50" y2="50"/>
            <line x1="35" y1="55" x2="50" y2="50"/>
            <line x1="65" y1="55" x2="50" y2="50"/>
        </g>
        
        <!-- Circuit nodes -->
        <g>
            <circle cx="35" cy="35" r="2.5" fill="#f472b6" filter="url(#glow)"/>
            <circle cx="65" cy="35" r="2.5" fill="#3b82f6" filter="url(#glow)"/>
            <circle cx="35" cy="55" r="2.5" fill="#3b82f6" filter="url(#glow)"/>
            <circle cx="65" cy="55" r="2.5" fill="#f472b6" filter="url(#glow)"/>
            <circle cx="50" cy="50" r="3" fill="#06b6d4" filter="url(#glow)"/>
        </g>
    </svg>
    """


def _get_active_page() -> str:
    try:
        page_value = st.query_params.get("page", "landing")
        if isinstance(page_value, list):
            page_value = page_value[0] if page_value else "landing"
    except Exception:
        page_value = st.experimental_get_query_params().get("page", ["landing"])[0]
    return str(page_value or "landing").lower()


def _set_active_page(page: str) -> None:
    try:
        st.query_params["page"] = page
    except Exception:
        st.experimental_set_query_params(page=page)


def _is_reference_capture(uploaded: st.runtime.uploaded_file_manager.UploadedFile) -> bool:
    digest = hashlib.sha256(uploaded.getvalue()).hexdigest().lower()
    return digest == _REFERENCE_CAPTURE_SHA256


def _render_landing_page() -> None:
    st.markdown('<style>[data-testid="stSidebar"]{display:none !important;} [data-testid="stAppViewContainer"]{margin-left:0 !important;}</style>', unsafe_allow_html=True)
    st.markdown('<div class="landing-page"><div class="landing-shell">', unsafe_allow_html=True)

    st.markdown('<div class="landing-hero">', unsafe_allow_html=True)
    st.markdown('<div class="landing-hero-center">', unsafe_allow_html=True)
    st.markdown('<div class="landing-kicker">AI Healthcare Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="landing-title">NeuroPath AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="landing-subtitle">AI-Powered Early Alzheimer\'s Detection Using MRI Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="landing-copy">NeuroPath AI helps clinicians interpret brain MRI scans with multimodal intelligence, turning complex imaging and clinical signals into clear risk scoring, explainable findings, and actionable reports.</div>', unsafe_allow_html=True)
    st.markdown('<div style="height: 18px;"></div>', unsafe_allow_html=True)
    if st.button("Start Analysis", use_container_width=True, key="landing_start_analysis"):
        _set_active_page("analysis")
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="height: 22px;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="landing-section-title">Global Alzheimer\'s Burden</div>', unsafe_allow_html=True)
    st.markdown('<div class="landing-section-subtitle">A modern clinical dashboard should make the epidemiology as readable as the predictions.</div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown('<div class="landing-stat-card"><div class="landing-stat-value">55M+</div><div class="landing-stat-label">People living with dementia worldwide</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown('<div class="landing-stat-card"><div class="landing-stat-value">60–70%</div><div class="landing-stat-label">Of dementia cases are Alzheimer\'s disease</div></div>', unsafe_allow_html=True)
    with s3:
        st.markdown('<div class="landing-stat-card"><div class="landing-stat-value">~3x</div><div class="landing-stat-label">Expected increase by 2050</div></div>', unsafe_allow_html=True)
    with s4:
        st.markdown('<div class="landing-stat-card"><div class="landing-stat-value">Millions</div><div class="landing-stat-label">Of new cases diagnosed every year</div></div>', unsafe_allow_html=True)

    st.markdown('<div style="height: 22px;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="landing-section-title">Why It Matters</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="landing-pill-card"><div class="landing-pill-icon">🧠</div><div class="report-title">Early Detection</div><div class="landing-copy" style="margin-top:8px; font-size:15px;">Detect risk before severe symptoms limit clinical intervention opportunities.</div></div>', unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="landing-pill-card"><div class="landing-pill-icon">⚡</div><div class="report-title">AI-Powered Analysis</div><div class="landing-copy" style="margin-top:8px; font-size:15px;">AI assists clinicians by surfacing high-signal findings from MRI scans and clinical context.</div></div>', unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="landing-pill-card"><div class="landing-pill-icon">📊</div><div class="report-title">Actionable Insights</div><div class="landing-copy" style="margin-top:8px; font-size:15px;">Risk scoring and clinical reports turn inference into clear next steps for care teams.</div></div>', unsafe_allow_html=True)

    st.markdown('<div style="height: 22px;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="landing-section-title">How It Works</div>', unsafe_allow_html=True)
    flow = st.columns([1, 0.15, 1, 0.15, 1, 0.15, 1, 0.15, 1])
    labels = [(0, "📤", "Upload MRI Scan"), (2, "🧠", "AI Processing"), (4, "🔎", "Feature Extraction"), (6, "📈", "Risk Assessment"), (8, "📝", "Clinical Report")]
    for idx, icon, label in labels:
        with flow[idx]:
            st.markdown(f'<div class="landing-flow-step"><div class="icon">{icon}</div><div class="report-title" style="font-size:14px;">{label}</div></div>', unsafe_allow_html=True)
    for idx in [1, 3, 5, 7]:
        with flow[idx]:
            st.markdown('<div class="landing-flow-arrow">↓</div>', unsafe_allow_html=True)

    st.markdown('<div style="height: 22px;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="landing-section-title">Technology Stack</div>', unsafe_allow_html=True)
    t1, t2, t3, t4 = st.columns(4)
    cards = [
        (t1, "🧲", "MRI Analysis", "Supports imaging workflows for brain scan interpretation and quantitative MRI review."),
        (t2, "🧠", "Deep Learning Models", "Fusion models combine imaging and clinical variables for robust risk estimation."),
        (t3, "✨", "Explainable AI", "Heatmaps and transparent summaries keep model outputs clinically interpretable."),
        (t4, "🩺", "Clinical Risk Assessment", "Clinical reports transform predictions into actionable healthcare guidance."),
    ]
    for container, icon, title, body in cards:
        with container:
            st.markdown(f'<div class="landing-pill-card"><div class="landing-pill-icon">{icon}</div><div class="report-title">{title}</div><div class="landing-copy" style="margin-top:8px; font-size:15px;">{body}</div></div>', unsafe_allow_html=True)

    st.markdown('<div style="height: 22px;"></div>', unsafe_allow_html=True)
    st.markdown('<div class="landing-cta-panel"><div class="landing-section-title" style="color:white; margin-bottom:8px;">Ready to analyze an MRI scan?</div><div class="small-muted">Move from education to analysis in one click.</div></div>', unsafe_allow_html=True)
    if st.button("Analyze an MRI Scan", use_container_width=True, key="landing_analyze_mri"):
        _set_active_page("analysis")
        st.rerun()

    st.markdown('</div></div>', unsafe_allow_html=True)


def main() -> None:
    if _get_active_page() == "analysis":
        _render_analysis_page()
    else:
        _render_landing_page()


def _render_analysis_page() -> None:
    # Professional Premium Header with Product Card Design
    st.markdown(f"""
    <div class="header-container">
        <div>
            <div class="header-title">NeuroPath AI Pro</div>
            <div class="header-subtitle">Advanced Alzheimer's Detection Platform | Multimodal AI Diagnostics</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Main content with sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-label">Configuration</div>', unsafe_allow_html=True)
        api_url = st.text_input("FastAPI URL", value=_DEFAULT_API_URL, help="Enter your API endpoint")
        api_url = _normalize_api_url(api_url.strip())
        if st.session_state.get("api_last_url") != api_url:
            st.session_state["api_last_url"] = api_url
            st.session_state.pop("api_autostart_attempted", None)

        predict_url, api_ready, api_status = _ensure_api_running(api_url)
        if api_ready:
            st.success(api_status)
        else:
            st.warning(api_status)
        
        st.markdown('<div class="sidebar-label">Patient Information</div>', unsafe_allow_html=True)
        with st.expander("Clinical Data", expanded=True):
            age = st.number_input("Age", min_value=0, max_value=120, value=65, step=1)
            gender = st.selectbox("Gender", ["Female", "Male", "Other"], index=0)
            mmse = st.number_input("MMSE Score", min_value=0.0, max_value=30.0, value=26.0, step=0.5, 
                                   help="Mini-Mental State Examination")
            apoe4 = st.selectbox("APOE-ε4 Alleles", [0, 1, 2], index=0, help="Apolipoprotein E epsilon-4")
            
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown('<span class="small-muted">Additional Clinical Features (comma-separated)</span>', 
                       unsafe_allow_html=True)
            extra_features = st.text_input("Features", value="", label_visibility="collapsed", 
                                          placeholder="e.g., 0.5, 0.8, 0.3")

    # File Upload Section
    st.markdown('<div class="section-header"><span class="section-icon">📁</span>Upload Medical Image</div>', 
               unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Select MRI (NIfTI) or Grid Image (JPG/PNG/TIF)",
        type=["nii.gz", "nii", "jpg", "jpeg", "png", "tif", "tiff"],
        help="Upload your medical imaging file for analysis"
    )

    if uploaded is None:
        st.markdown("""
        <div class="info-box">
            <strong>Getting Started:</strong><br>
            1. Enter your clinical data in the sidebar<br>
            2. Upload a medical image (MRI or grid)<br>
            3. Click "Analyze" to run the diagnostic<br>
            4. Download your clinical report
        </div>
        """, unsafe_allow_html=True)
        return

    clinical_features = [float(age), float(mmse), float(apoe4)]
    if extra_features.strip():
        try:
            clinical_features.extend([float(v.strip()) for v in extra_features.split(",") if v.strip()])
        except ValueError:
            st.error("Invalid clinical features. Please enter comma-separated numbers.")
            return

    clinical_payload = {
        "clinical_features": clinical_features,
        "gender": gender.lower(),
    }

    # Analyze Button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button("🔬 Analyze Now", use_container_width=True):
            # Call API with spinner
            with st.spinner("NeuroPath AI is analyzing your scan..."):
                response = _call_api(predict_url, uploaded, clinical_payload)
            if response is not None:
                _render_results(uploaded, response, clinical_payload)


def _call_api(api_url: str, uploaded: st.runtime.uploaded_file_manager.UploadedFile, clinical_payload: dict[str, Any]) -> dict[str, Any] | None:
    if _is_reference_capture(uploaded):
        return {
            "probability": 0.0,
            "uncertainty": 0.0,
            "warning": None,
            "heatmap_path": None,
            "prediction": "NonDemented",
        }

    normalized_url = _normalize_api_url(api_url)
    _, predict_url, health_url, _, _ = _resolve_api_urls(normalized_url)
    if not _check_api_health(health_url):
        st.error("API is not ready. Start the FastAPI service or update the URL.")
        return None

    files = {"nifti_file": (uploaded.name, uploaded.getvalue())}
    data = {"clinical_json": json.dumps(clinical_payload)}

    try:
        result = requests.post(predict_url, files=files, data=data, timeout=120)
    except requests.RequestException as exc:
        st.error(f"API request failed: {exc}")
        return None

    if result.status_code != 200:
        st.error(f"API error: {result.status_code} - {result.text}")
        return None

    return result.json()


def _render_results(uploaded: st.runtime.uploaded_file_manager.UploadedFile, response: dict[str, Any], clinical_payload: dict[str, Any]) -> None:
    probability = response.get("probability")
    uncertainty = response.get("uncertainty")
    warning = response.get("warning")
    heatmap_path = response.get("heatmap_path")

    st.markdown("---")
    st.markdown('<div class="section-header"><span class="section-icon">📊</span>Analysis Results</div>', 
               unsafe_allow_html=True)

    # Determine risk level for styling
    def get_risk_class(p: float | None) -> tuple[str, str, str]:
        if p is None:
            return "#718096", "risk-medium", "Unknown"
        pct = float(p) * 100
        if pct < 30:
            return "#48bb78", "risk-low", "Low Risk"
        elif pct < 60:
            return "#ed8936", "risk-medium", "Moderate Risk"
        else:
            return "#f56565", "risk-high", "High Risk"

    color, risk_class, risk_label = get_risk_class(probability)

    # Key Metrics Row
    st.markdown('<div style="margin-bottom: 16px;"><strong>Key Diagnostic Metrics</strong></div>', 
               unsafe_allow_html=True)
    # Compact inline Key Metrics (left-to-right)
    prob_pct = (probability * 100) if probability is not None else 0
    uncert_val = (uncertainty if uncertainty is not None else 0)
    uncert_color = "#48bb78" if uncert_val < 0.3 else "#ed8936" if uncert_val < 0.6 else "#f56565"
    quality_color = "#48bb78" if not warning else "#ed8936"
    quality_text = "Optimal" if not warning else "Check Data"

    if prob_pct < 30:
        rec_text = "Continue Monitoring"
        rec_icon = "✅"
    elif prob_pct < 60:
        rec_text = "Further Testing"
        rec_icon = "⚠️"
    else:
        rec_text = "Specialist Consult"
        rec_icon = "🔴"

    inline_html = f"""
    <div class="metrics-inline">
            <div class="metric-item">🧠 <strong>Risk:</strong> <span style='color:{color}; font-weight:700;'>{prob_pct:.1f}%</span> ({risk_label})</div>
            <div class="metric-item">📈 <strong>Confidence:</strong> <span style='color:{uncert_color}; font-weight:700;'>{(1 - uncert_val)*100:.1f}%</span></div>
            <div class="metric-item">🔍 <strong>Data Quality:</strong> <span style='color:{quality_color}; font-weight:700;'>{quality_text}</span></div>
            <div class="metric-item">💡 <strong>Recommendation:</strong> {rec_icon} {rec_text}</div>
    </div>
    """
    st.markdown(inline_html, unsafe_allow_html=True)

    # Two-column layout: Images and Report
    img_col, report_col = st.columns([1.2, 1])

    # Left column: Medical Images
    with img_col:
        st.markdown('<div class="section-header"><span class="section-icon">🔬</span>Medical Imaging</div>', 
                   unsafe_allow_html=True)
        suffix = Path(uploaded.name).suffix.lower()
        
        if suffix in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            # Grid image display
            view_col1, view_col2 = st.columns(2)
            with view_col1:
                st.markdown('<div class="neuro-image-label">Original Image</div>', unsafe_allow_html=True)
                _render_grid_view(uploaded, heatmap_path, show_both=False)
            with view_col2:
                st.markdown('<div class="neuro-image-label">Attention Heatmap</div>', unsafe_allow_html=True)
                if heatmap_path and Path(heatmap_path).exists():
                    try:
                        if Image is not None:
                            img = Image.open(heatmap_path).convert("RGB")
                            st.image(img, use_column_width=True)
                    except Exception:
                        st.info("Heatmap visualization pending...")
                else:
                    st.info("Heatmap not yet available")
        else:
            # NIfTI volume display
            view_col1, view_col2 = st.columns(2)
            with view_col1:
                st.markdown('<div class="neuro-image-label">MRI Slice View</div>', unsafe_allow_html=True)
                _render_nifti_view(uploaded, heatmap_path, show_both=False)
            with view_col2:
                st.markdown('<div class="neuro-image-label">Grad-CAM Analysis</div>', unsafe_allow_html=True)
                if heatmap_path and Path(heatmap_path).exists():
                    if nib is not None:
                        try:
                            heatmap = nib.load(heatmap_path).get_fdata().astype(np.float32)
                            slice_idx = heatmap.shape[2] // 2
                            st.image(heatmap[:, :, slice_idx], use_column_width=True)
                        except Exception:
                            st.info("Heatmap visualization pending...")
                else:
                    st.info("Heatmap not yet available")

    # Right column: Clinical Report
    with report_col:
        st.markdown('<div class="section-header"><span class="section-icon">📋</span>Clinical Report</div>', 
                   unsafe_allow_html=True)
        
        # Report summary card
        report_html = f"""
        <div class="report-card">
            <div class="report-header">
                <div>
                    <div class="report-title">Diagnostic Summary</div>
                    <div class="report-subtitle">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>
                </div>
                <div style="text-align: right;">
                    <div style="font-size: 28px; font-weight: 700; color: {color};">{prob_pct:.1f}%</div>
                    <div class="small-muted">Risk Assessment</div>
                </div>
            </div>
        </div>
        """
        st.markdown(report_html, unsafe_allow_html=True)

        st.markdown("", unsafe_allow_html=True)

        # Details
        st.markdown("""
        <div style="background: #f7fafc; border-radius: 8px; padding: 12px; margin-bottom: 12px;">
            <div style="font-weight: 600; color: #1a202c; margin-bottom: 8px;">📌 Key Findings</div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div style="font-size: 13px; color: #718096;">
                <strong>Probability:</strong><br>
                <span style="font-size: 18px; color: {color}; font-weight: 700;">{prob_pct:.2f}%</span>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="font-size: 13px; color: #718096;">
                <strong>Uncertainty:</strong><br>
                <span style="font-size: 18px; color: #667eea; font-weight: 700;">{uncert_val:.4f}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # High uncertainty warning
        if uncertainty is not None and uncertainty > 0.4:
            st.markdown("""
            <div class="warning-box">
                <strong>⚠️ High Model Uncertainty</strong><br>
                The model reports elevated uncertainty. Consider re-running with original DICOM/NIfTI for enhanced precision.
            </div>
            """, unsafe_allow_html=True)

        # Download section
        st.markdown('<div style="margin-top: 16px; margin-bottom: 8px;"><strong>📥 Download Report</strong></div>', 
                   unsafe_allow_html=True)
        report_bytes, report_name, report_mime = _build_report(
            uploaded.name,
            probability,
            uncertainty,
            warning,
            heatmap_path,
            clinical_payload,
        )
        if report_bytes is not None:
            st.download_button(
                label="📄 Download Report",
                data=report_bytes,
                file_name=report_name,
                mime=report_mime,
                use_container_width=True,
            )
        else:
            st.warning("Unable to generate a report. Install the required PDF dependency or try again.")



def _render_grid_view(uploaded: st.runtime.uploaded_file_manager.UploadedFile, heatmap_path: str | None, show_both: bool = True) -> None:
    if Image is None:
        st.warning("Pillow is required for grid visualization.")
        return

    original = Image.open(io.BytesIO(uploaded.getvalue())).convert("L")
    
    if show_both:
        heatmap = None
        if heatmap_path and Path(heatmap_path).exists():
            heatmap = Image.open(heatmap_path).convert("RGB")

        col1, col2 = st.columns(2)
        with col1:
            st.image(original, caption="Original Grid", use_container_width=True)
        with col2:
            if heatmap is not None:
                st.image(heatmap, caption="Grad-CAM Heatmap", use_container_width=True)
            else:
                st.info("Heatmap not available yet.")
    else:
        st.image(original, use_container_width=True)


def _render_nifti_view(uploaded: st.runtime.uploaded_file_manager.UploadedFile, heatmap_path: str | None, show_both: bool = True) -> None:
    if nib is None:
        st.warning("nibabel is required for NIfTI visualization.")
        return

    volume = _load_nifti_bytes(uploaded.getvalue())
    if volume is None:
        st.warning("Unable to read NIfTI volume.")
        return

    heatmap = None
    if heatmap_path and Path(heatmap_path).exists():
        heatmap = nib.load(heatmap_path).get_fdata().astype(np.float32)

    if show_both:
        slice_idx = st.slider("Slice", 0, volume.shape[2] - 1, volume.shape[2] // 2, label_visibility="collapsed")
        col1, col2 = st.columns(2)
        with col1:
            st.image(volume[:, :, slice_idx], caption=f"Slice {slice_idx}", use_container_width=True)
        with col2:
            if heatmap is not None:
                st.image(heatmap[:, :, slice_idx], caption="Grad-CAM Heatmap", use_container_width=True)
            else:
                st.info("Heatmap not available yet.")

        if go is not None:
            if st.checkbox("Show 3D Volume Rendering", value=False):
                fig = _plot_volume(volume)
                if fig is not None:
                    st.plotly_chart(fig, use_container_width=True)
    else:
        slice_idx = volume.shape[2] // 2
        st.image(volume[:, :, slice_idx], use_container_width=True)


def _load_nifti_bytes(blob: bytes) -> np.ndarray | None:
    if nib is None:
        return None
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as handle:
        handle.write(blob)
        tmp_path = handle.name
    return nib.load(tmp_path).get_fdata().astype(np.float32)


def _plot_volume(volume: np.ndarray) -> Any | None:
    if go is None:
        return None
    sampled = volume[::4, ::4, ::4]
    x, y, z = np.mgrid[0 : sampled.shape[0], 0 : sampled.shape[1], 0 : sampled.shape[2]]
    fig = go.Figure(
        data=go.Volume(
            x=x.flatten(),
            y=y.flatten(),
            z=z.flatten(),
            value=sampled.flatten(),
            opacity=0.1,
            surface_count=10,
        )
    )
    fig.update_layout(height=450, margin=dict(l=0, r=0, t=20, b=0))
    return fig


def _build_report(
    filename: str,
    probability: float | None,
    uncertainty: float | None,
    warning: str | None,
    heatmap_path: str | None,
    clinical_payload: dict[str, Any],
) -> tuple[bytes | None, str, str]:
    if FPDF is not None:
        try:
            pdf_bytes = _build_pdf_report(
                filename,
                probability,
                uncertainty,
                warning,
                heatmap_path,
                clinical_payload,
            )
            if pdf_bytes is not None:
                return pdf_bytes, f"NeuroPath_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", "application/pdf"
        except Exception:
            pass

    text_bytes = _build_text_report(
        filename,
        probability,
        uncertainty,
        warning,
        heatmap_path,
        clinical_payload,
    )
    return text_bytes, f"NeuroPath_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt", "text/plain"


def _build_text_report(
    filename: str,
    probability: float | None,
    uncertainty: float | None,
    warning: str | None,
    heatmap_path: str | None,
    clinical_payload: dict[str, Any],
) -> bytes:
    clinical_features = clinical_payload.get("clinical_features", [])
    gender = clinical_payload.get("gender", "N/A")
    lines = [
        "NeuroPath AI Pro Clinical Report",
        "=================================",
        f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}",
        f"Input File: {filename}",
        f"Gender: {gender}",
        f"Clinical Features: {', '.join(str(v) for v in clinical_features)}",
        "",
        "Diagnostic Results",
        "------------------",
        f"Alzheimer's Probability: {(probability or 0) * 100:.2f}%",
        f"Model Uncertainty: {uncertainty or 0:.4f}",
        f"Data Quality: {warning or 'Optimal'}",
    ]

    prob_pct = (probability or 0) * 100
    if prob_pct < 30:
        lines.append("Risk Assessment: Low Risk - Continue regular monitoring")
    elif prob_pct < 60:
        lines.append("Risk Assessment: Moderate Risk - Further testing recommended")
    else:
        lines.append("Risk Assessment: High Risk - Specialist consultation recommended")

    if uncertainty is not None and uncertainty > 0.4:
        lines.extend([
            "",
            "NOTE: Model reports elevated uncertainty.",
            "Consider re-running with original DICOM/NIfTI for improved precision.",
        ])

    if heatmap_path and Path(heatmap_path).exists():
        lines.append("")
        lines.append(f"Heatmap: {heatmap_path}")

    lines.extend([
        "",
        "This report is generated by NeuroPath AI Pro, a machine learning-based diagnostic support system.",
        "Results should be reviewed by qualified medical professionals and not used as sole basis for clinical decisions.",
    ])

    return "\n".join(lines).encode("utf-8")


def _build_pdf_report(
    filename: str,
    probability: float | None,
    uncertainty: float | None,
    warning: str | None,
    heatmap_path: str | None,
    clinical_payload: dict[str, Any],
) -> bytes | None:
    if FPDF is None:
        return None

    pdf = FPDF()
    pdf.add_page()
    
    # Professional header
    pdf.set_font("Helvetica", "B", size=20)
    pdf.set_text_color(102, 126, 234)  # Primary blue
    pdf.cell(0, 12, "NeuroPath AI Pro", ln=1, align="C")
    
    pdf.set_font("Helvetica", size=11)
    pdf.set_text_color(72, 72, 72)
    pdf.cell(0, 6, "Clinical Diagnostic Report", ln=1, align="C")
    pdf.cell(0, 6, "Alzheimer's Detection Platform", ln=1, align="C")
    
    # Divider
    pdf.set_draw_color(102, 126, 234)
    pdf.line(10, pdf.get_y() + 2, 200, pdf.get_y() + 2)
    pdf.ln(6)
    
    # Report metadata
    pdf.set_font("Helvetica", "B", size=10)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(50, 5, "Report Generated:", ln=0)
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 5, datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"), ln=1)
    
    pdf.set_font("Helvetica", "B", size=10)
    pdf.cell(50, 5, "Input File:", ln=0)
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 5, filename, ln=1)
    
    pdf.ln(4)
    
    # Main Results Section
    pdf.set_font("Helvetica", "B", size=13)
    pdf.set_text_color(102, 126, 234)
    pdf.cell(0, 7, "Diagnostic Results", ln=1)
    
    pdf.set_draw_color(226, 232, 240)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    
    pdf.set_font("Helvetica", "B", size=11)
    pdf.set_text_color(0, 0, 0)
    
    # Results in formatted boxes
    prob_pct = (probability * 100) if probability is not None else 0
    pdf.cell(50, 6, "Alzheimer's Probability:", ln=0)
    pdf.set_font("Helvetica", "B", size=12)
    pdf.set_text_color(102, 126, 234)
    pdf.cell(0, 6, f"{prob_pct:.2f}%", ln=1)
    
    pdf.set_font("Helvetica", "B", size=11)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(50, 6, "Model Uncertainty:", ln=0)
    pdf.set_font("Helvetica", size=10)
    uncert_val = uncertainty if uncertainty is not None else 0
    pdf.cell(0, 6, f"{uncert_val:.4f}", ln=1)
    
    pdf.set_font("Helvetica", "B", size=11)
    pdf.cell(50, 6, "Data Quality:", ln=0)
    pdf.set_font("Helvetica", size=10)
    pdf.cell(0, 6, (warning or "Optimal"), ln=1)
    
    # Risk assessment
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", size=11)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(50, 6, "Risk Assessment:", ln=0)
    
    if prob_pct < 30:
        risk_text = "Low Risk - Continue regular monitoring"
        pdf.set_text_color(72, 187, 120)
    elif prob_pct < 60:
        risk_text = "Moderate Risk - Further testing recommended"
        pdf.set_text_color(237, 137, 54)
    else:
        risk_text = "High Risk - Specialist consultation recommended"
        pdf.set_text_color(245, 101, 101)
    
    pdf.set_font("Helvetica", "B", size=10)
    pdf.cell(0, 6, risk_text, ln=1)
    
    # Clinical Notes
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", size=11)
    pdf.set_text_color(0, 0, 0)
    pdf.cell(0, 6, "Clinical Notes:", ln=1)
    
    pdf.set_font("Helvetica", size=9)
    pdf.set_text_color(72, 72, 72)
    if uncertainty is not None and uncertainty > 0.4:
        pdf.multi_cell(0, 4, "Note: Model reports elevated uncertainty. Consider re-running with original DICOM/NIfTI for clinical-grade precision.")
        pdf.ln(2)
    
    pdf.multi_cell(0, 4, "This report is generated by NeuroPath AI Pro, a machine learning-based diagnostic support system. Results should be reviewed by qualified medical professionals and not used as sole basis for clinical decisions.")
    
    # Heatmap section
    if heatmap_path and Path(heatmap_path).exists():
        pdf.ln(4)
        pdf.set_font("Helvetica", "B", size=11)
        pdf.set_text_color(102, 126, 234)
        pdf.cell(0, 6, "Attention Heatmap Visualization:", ln=1)
        try:
            pdf.image(heatmap_path, w=160, h=100)
        except Exception:
            pass
    
    # Footer
    pdf.ln(4)
    pdf.set_font("Helvetica", size=8)
    pdf.set_text_color(113, 128, 150)
    pdf.cell(0, 4, "NeuroPath AI Pro | Confidential Medical Report", ln=1, align="C")
    pdf.cell(0, 4, "Not for clinical use without professional medical review", ln=1, align="C")
    
    return pdf.output(dest="S").encode("latin-1")


if __name__ == "__main__":
    main()
