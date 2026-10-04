import streamlit as st
import tempfile
import time
from datetime import datetime, timedelta


def setup_ui():
    # --- PAGE CONFIG ---
    st.set_page_config(
        page_title="Smart Surveillance System",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # Initialize session state
    if "run" not in st.session_state:
        st.session_state.run = False
    if "splash_done" not in st.session_state:
        st.session_state.splash_done = False
    if "current_page" not in st.session_state:
        st.session_state.current_page = "dashboard"
    if "video_source_type" not in st.session_state:
        st.session_state.video_source_type = None
    # Live stats counters updated by detection logic
    if "stat_cameras" not in st.session_state:
        st.session_state.stat_cameras = 0
    if "stat_alerts" not in st.session_state:
        st.session_state.stat_alerts = 0
    if "stat_incidents" not in st.session_state:
        st.session_state.stat_incidents = 0
    if "stat_health" not in st.session_state:
        st.session_state.stat_health = "Offline"

    # Derive health from run state
    if st.session_state.run:
        st.session_state.stat_health = "Online"
        if st.session_state.stat_cameras == 0:
            st.session_state.stat_cameras = 1

    # ------------------------------------------------------------------ CSS
    st.markdown(
        """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

    /* ===== GLOBAL RESET ===== */
    header { visibility: hidden; }
    footer { visibility: hidden; }
    .block-container {
        padding-top: 0 !important;
        padding-bottom: 2rem !important;
        max-width: 100% !important;
    }

    /* ===== LIGHT APP BACKGROUND ===== */
    .stApp {
        background: #f0f4f8;
        color: #0f172a;
        font-family: 'Sora', sans-serif;
    }

    /* Subtle dot grid */
    .stApp::before {
        content: '';
        position: fixed;
        inset: 0;
        background-image: radial-gradient(circle, rgba(14,165,233,0.06) 1px, transparent 1px);
        background-size: 30px 30px;
        pointer-events: none;
        z-index: 0;
    }

    /* ===== SPLASH ANIMATIONS ===== */
    @keyframes typePreempt {
        0%   { width: 0; }
        100% { width: 28ch; }
    }
    @keyframes typeAI {
        0%   { width: 0; }
        100% { width: 2ch; }
    }
    @keyframes cursorBlink {
        0%, 49% { border-color: #0ea5e9; }
        50%, 100% { border-color: transparent; }
    }
    @keyframes fadeUp {
        0%   { opacity: 0; transform: translateY(10px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    @keyframes progressFill {
        0%   { width: 0%; }
        100% { width: 100%; }
    }
    @keyframes splashOut {
        0%   { opacity: 1; }
        100% { opacity: 0; pointer-events: none; }
    }
    @keyframes glowPulse {
        0%, 100% { opacity: 0.28; transform: translate(-50%,-50%) scale(1); }
        50%       { opacity: 0.5;  transform: translate(-50%,-50%) scale(1.05); }
    }
    @keyframes logoPop {
        0%   { opacity: 0; transform: scale(0.72); }
        60%  { transform: scale(1.07); }
        100% { opacity: 1; transform: scale(1); }
    }
    @keyframes statusPulse {
        0%, 100% { opacity: 1; }
        50%       { opacity: 0.55; }
    }

    /* ===== SPLASH OVERLAY ===== */
    .splash-overlay {
        position: fixed;
        inset: 0;
        z-index: 9999;
        background: #f8fafc;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        animation: splashOut 0.55s ease-out 4.2s forwards;
    }

    .splash-glow {
        position: absolute;
        width: 520px; height: 520px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(14,165,233,0.11) 0%, transparent 70%);
        top: 50%; left: 50%;
        transform: translate(-50%,-50%);
        animation: glowPulse 3s ease-in-out infinite;
    }
    .splash-glow2 {
        position: absolute;
        width: 760px; height: 760px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(14,165,233,0.05) 0%, transparent 70%);
        top: 50%; left: 50%;
        transform: translate(-50%,-50%);
        animation: glowPulse 3s ease-in-out 0.6s infinite;
    }

    .splash-inner {
        position: relative;
        z-index: 2;
        display: flex;
        flex-direction: column;
        align-items: center;
    }

    .splash-logo {
        width: 84px; height: 84px;
        border-radius: 22px;
        background: linear-gradient(145deg, #0ea5e9 0%, #0369a1 100%);
        display: flex; align-items: center; justify-content: center;
        margin-bottom: 32px;
        box-shadow: 0 0 0 14px rgba(14,165,233,0.07), 0 20px 60px rgba(14,165,233,0.22);
        animation: logoPop 0.7s cubic-bezier(0.34,1.56,0.64,1) 0.1s both;
    }
    .splash-logo svg {
        width: 42px; height: 42px;
        stroke: #fff; fill: none; stroke-width: 1.6;
    }

    /* Typewriter brand row */
    .splash-brand-row {
        display: flex;
        justify-content: center;   /* ✅ centers horizontally */
        align-items: center;       /* ✅ centers vertically */
        margin-bottom: 14px;
        width: 100%;               /* ensures full width */
    }
    .splash-preempt, .splash-ai {
        text-align: center;
    }
    .splash-preempt {
        font-family: 'Sora', sans-serif;
        font-size: 60px; font-weight: 800;
        letter-spacing: -2.5px; color: #0f172a;
        overflow: hidden; white-space: nowrap;
        width: 0; display: inline-block;
        animation: typePreempt 2s steps(28, end) 0.8s forwards;
    }

    .splash-ai {
        font-family: 'Sora', sans-serif;
        font-size: 60px; font-weight: 800;
        letter-spacing: -2.5px; color: #0ea5e9;
        overflow: hidden; white-space: nowrap;
        width: 0; display: inline-block;
        border-right: 3px solid transparent;
        animation:
            typeAI 0.35s steps(2, end) 1.75s forwards,
            cursorBlink 0.55s step-end 1.75s 5;
    }

    .splash-tagline {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px; letter-spacing: 3.5px;
        text-transform: uppercase; color: #94a3b8;
        margin-bottom: 52px;
        animation: fadeUp 0.6s ease-out 2.3s both;
    }

    .splash-progress-track {
        width: 200px; height: 2px;
        background: #e2e8f0; border-radius: 2px;
        overflow: hidden;
        animation: fadeUp 0.4s ease-out 2.2s both;
    }
    .splash-progress-fill {
        height: 100%;
        background: linear-gradient(90deg, #0ea5e9, #38bdf8);
        border-radius: 2px;
        animation: progressFill 1.8s cubic-bezier(0.4,0,0.2,1) 2.4s both;
    }

    .splash-footer-text {
        position: absolute; bottom: 28px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 9px; letter-spacing: 2.5px; color: #cbd5e1;
        z-index: 2;
    }

    /* ===== TOP NAVBAR ===== */
    .top-navbar {
        position: relative; z-index: 100;
        display: flex; align-items: center; justify-content: space-between;
        padding: 13px 40px;
        background: transparent;
        backdrop-filter: blur(24px);
        border-bottom: none;
        margin: -1rem -1rem 0 -1rem;
        width: calc(100% + 2rem);
        box-sizing: border-box;
        box-shadow: none;
    }

    .navbar-brand { display: flex; align-items: center; gap: 11px; }

    .navbar-logo-icon {
        width: 34px; height: 34px; border-radius: 9px;
        background: linear-gradient(145deg, #0ea5e9, #0369a1);
        display: flex; align-items: center; justify-content: center;
        box-shadow: 0 3px 10px rgba(14,165,233,0.28);
    }
    .navbar-logo-icon svg {
        width: 18px; height: 18px;
        stroke: #fff; fill: none; stroke-width: 2;
    }

    .navbar-brand-text {
        font-family: 'Sora', sans-serif;
        font-size: 19px; font-weight: 700;
        color: #0ea5e9; letter-spacing: -0.4px;
    }
    .navbar-brand-text span { color: #0ea5e9; }

    .navbar-right { display: flex; align-items: center; gap: 12px; }

    .system-status-badge {
        display: flex; align-items: center; gap: 7px;
        padding: 7px 14px; border-radius: 100px;
        background: rgba(34,197,94,0.08);
        border: 1px solid rgba(34,197,94,0.22);
    }
    .status-dot {
        width: 7px; height: 7px; border-radius: 50%;
        background: #22c55e;
        box-shadow: 0 0 7px rgba(34,197,94,0.55);
        animation: statusPulse 2s ease-in-out infinite;
    }
    .status-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10px; font-weight: 500;
        letter-spacing: 1px; color: #16a34a;
    }

    /* ===== NAV BUTTONS ===== */
    .stButton > button {
        background: #fff !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 9px !important;
        color: #475569 !important;
        font-family: 'Sora', sans-serif !important;
        font-size: 13px !important; font-weight: 500 !important;
        padding: 9px 18px !important; height: auto !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05) !important;
    }
    .stButton > button:hover {
        background: #f0f9ff !important;
        border-color: #bae6fd !important;
        color: #0369a1 !important;
        transform: none !important;
        box-shadow: 0 2px 8px rgba(14,165,233,0.1) !important;
    }

    .primary-btn button {
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important;
        border: none !important; color: #fff !important;
        box-shadow: 0 4px 14px rgba(14,165,233,0.28) !important;
        font-weight: 600 !important;
    }
    .primary-btn button:hover {
        box-shadow: 0 6px 20px rgba(14,165,233,0.38) !important;
        transform: translateY(-1px) !important;
    }

    .danger-btn button {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
        border: none !important; color: #fff !important;
        box-shadow: 0 4px 14px rgba(239,68,68,0.22) !important;
        font-weight: 600 !important;
    }
    .danger-btn button:hover {
        box-shadow: 0 6px 20px rgba(239,68,68,0.32) !important;
        transform: translateY(-1px) !important;
    }

    .action-btn button {
        background: #f0f9ff !important;
        border: 1px solid #bae6fd !important;
        color: #0369a1 !important; font-weight: 600 !important;
    }
    .action-btn button:hover {
        background: #e0f2fe !important;
        border-color: #7dd3fc !important; color: #0ea5e9 !important;
    }

    /* ===== WELCOME SECTION ===== */
    .welcome-section { margin: 30px 0 22px; }
    .welcome-eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10px; letter-spacing: 3px;
        text-transform: uppercase; color: #0ea5e9; margin-bottom: 8px;
    }
    .welcome-heading {
        font-family: 'Sora', sans-serif;
        font-size: 29px; font-weight: 700;
        letter-spacing: -0.7px; color: #0f172a; margin-bottom: 5px;
    }
    .welcome-sub {
        font-family: 'Sora', sans-serif;
        font-size: 14px; color: #64748b; font-weight: 400;
    }

    /* ===== STAT CARDS ===== */
    .stat-card-wrap {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 22px 22px 18px;
        transition: all 0.22s ease;
        position: relative; overflow: hidden;
        box-shadow: 0 1px 5px rgba(0,0,0,0.055);
    }
    .stat-card-wrap::after {
        content: '';
        position: absolute; top: 0; left: 0; right: 0;
        height: 3px;
        background: var(--card-accent, #0ea5e9);
        border-radius: 16px 16px 0 0;
        opacity: 0; transition: opacity 0.22s ease;
    }
    .stat-card-wrap:hover {
        border-color: #bae6fd;
        transform: translateY(-3px);
        box-shadow: 0 12px 36px rgba(14,165,233,0.1);
    }
    .stat-card-wrap:hover::after { opacity: 1; }

    .stat-top {
        display: flex; justify-content: space-between;
        align-items: flex-start; margin-bottom: 14px;
    }
    .stat-label {
        font-family: 'Sora', sans-serif;
        font-size: 12px; color: #64748b; font-weight: 400; margin-bottom: 6px;
    }
    .stat-num {
        font-family: 'Sora', sans-serif;
        font-size: 36px; font-weight: 700;
        color: #0f172a; letter-spacing: -2px; line-height: 1;
    }
    .stat-icon-box {
        width: 42px; height: 42px; border-radius: 11px;
        display: flex; align-items: center; justify-content: center;
    }
    .stat-icon-box svg { width: 21px; height: 21px; }

    .stat-change-pill {
        display: inline-flex; align-items: center; gap: 3px;
        padding: 3px 9px; border-radius: 100px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10px; font-weight: 500; margin-top: 10px;
    }
    .stat-change-pill.up      { background: #dcfce7; color: #15803d; }
    .stat-change-pill.down    { background: #fee2e2; color: #dc2626; }
    .stat-change-pill.neutral { background: #f1f5f9; color: #64748b; }
    .stat-change-pill.online  { background: #dcfce7; color: #15803d; }
    .stat-change-pill.offline { background: #f1f5f9; color: #94a3b8; }

    /* ===== SECTION LABEL ===== */
    .section-label {
        font-family: 'Sora', sans-serif;
        font-size: 17px; font-weight: 600;
        color: #0f172a; margin: 30px 0 13px; letter-spacing: -0.2px;
    }

    /* ===== QUICK ACTION CARDS ===== */
    .quick-action-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 16px; padding: 24px;
        transition: all 0.22s ease;
        box-shadow: 0 1px 4px rgba(0,0,0,0.048);
    }
    .quick-action-card:hover {
        border-color: #bae6fd;
        transform: translateY(-3px);
        box-shadow: 0 12px 36px rgba(14,165,233,0.09);
    }
    .qa-icon {
        width: 44px; height: 44px; border-radius: 11px;
        display: flex; align-items: center; justify-content: center;
        margin-bottom: 13px;
    }
    .qa-icon svg { width: 22px; height: 22px; stroke-width: 1.6; fill: none; }
    .qa-title {
        font-family: 'Sora', sans-serif;
        font-size: 15px; font-weight: 600; color: #0f172a; margin-bottom: 6px;
    }
    .qa-desc {
        font-family: 'Sora', sans-serif;
        font-size: 12px; color: #64748b; line-height: 1.55; margin-bottom: 16px;
    }

    /* ===== INFO BANNER ===== */
    .info-banner {
        margin-top: 30px;
        background: #f0f9ff; border: 1px solid #bae6fd;
        border-radius: 16px; padding: 20px 26px;
        display: flex; align-items: center; gap: 18px;
    }
    .info-banner-icon {
        width: 44px; height: 44px; border-radius: 11px;
        background: rgba(14,165,233,0.1);
        display: flex; align-items: center; justify-content: center; flex-shrink: 0;
    }
    .info-banner-icon svg { width: 22px; height: 22px; stroke: #0ea5e9; fill: none; stroke-width: 1.6; }
    .info-banner-title {
        font-family: 'Sora', sans-serif;
        font-size: 13px; font-weight: 600; color: #0369a1; margin-bottom: 3px;
    }
    .info-banner-body {
        font-family: 'Sora', sans-serif;
        font-size: 12px; color: #0369a1; opacity: 0.75; line-height: 1.55;
    }

    /* ===== PAGE HEADER ===== */
    .page-header { margin: 24px 0 20px; }
    .page-title {
        font-family: 'Sora', sans-serif;
        font-size: 23px; font-weight: 700; color: #0f172a;
        letter-spacing: -0.5px; margin-bottom: 4px;
    }
    .page-subtitle { font-family: 'Sora', sans-serif; font-size: 13px; color: #64748b; }

    /* ===== SOURCE CARDS ===== */
    .source-select-card {
        background: #ffffff; border: 1px solid #e2e8f0;
        border-radius: 18px; padding: 28px;
        transition: all 0.22s ease;
        box-shadow: 0 1px 4px rgba(0,0,0,0.048);
    }
    .source-select-card:hover {
        border-color: #7dd3fc; transform: translateY(-3px);
        box-shadow: 0 14px 44px rgba(14,165,233,0.1);
    }
    .source-icon-box {
        width: 50px; height: 50px; border-radius: 13px;
        background: #f0f9ff; border: 1px solid #bae6fd;
        display: flex; align-items: center; justify-content: center; margin-bottom: 16px;
    }
    .source-icon-box svg { width: 24px; height: 24px; stroke: #0ea5e9; fill: none; stroke-width: 1.6; }
    .source-card-title {
        font-family: 'Sora', sans-serif;
        font-size: 16px; font-weight: 600; color: #0f172a; margin-bottom: 8px;
    }
    .source-card-desc {
        font-family: 'Sora', sans-serif;
        font-size: 12px; color: #64748b; line-height: 1.6; margin-bottom: 16px;
    }
    .feature-list { display: flex; flex-direction: column; gap: 7px; margin-bottom: 20px; }
    .feature-row {
        display: flex; align-items: center; gap: 9px;
        font-family: 'Sora', sans-serif; font-size: 12px; color: #475569;
    }
    .feature-pip { width: 5px; height: 5px; border-radius: 50%; background: #0ea5e9; flex-shrink: 0; }

    /* ===== VIDEO CONSOLE ===== */
    .video-shell {
        border: 1px solid #e2e8f0; border-radius: 18px;
        background: #0c1120; overflow: hidden;
        box-shadow: 0 20px 60px rgba(0,0,0,0.16);
        position: relative; margin-top: 18px;
    }
    .video-hud-bar {
        position: absolute; top: 0; left: 0; right: 0; padding: 13px 20px;
        display: flex; justify-content: space-between; align-items: center;
        background: linear-gradient(180deg, rgba(12,17,32,0.9) 0%, transparent 100%);
        z-index: 10; pointer-events: none;
    }
    .hud-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 9px; letter-spacing: 2px; color: rgba(14,165,233,0.55);
    }
    .hud-rec {
        display: flex; align-items: center; gap: 6px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 9px; letter-spacing: 2px; color: rgba(239,68,68,0.8);
    }
    .rec-pip {
        width: 7px; height: 7px; border-radius: 50%; background: #ef4444;
        animation: statusPulse 1s ease-in-out infinite;
    }
    .video-hud-bottom-bar {
        position: absolute; bottom: 0; left: 0; right: 0; padding: 13px 20px;
        display: flex; justify-content: space-between; align-items: center;
        background: linear-gradient(0deg, rgba(12,17,32,0.9) 0%, transparent 100%);
        z-index: 10; pointer-events: none;
    }
    .corner-mark { position: absolute; width: 20px; height: 20px; pointer-events: none; z-index: 11; }
    .corner-mark.tl { top: 12px; left: 12px; border-left: 1.5px solid rgba(14,165,233,0.4); border-top: 1.5px solid rgba(14,165,233,0.4); }
    .corner-mark.tr { top: 12px; right: 12px; border-right: 1.5px solid rgba(14,165,233,0.4); border-top: 1.5px solid rgba(14,165,233,0.4); }
    .corner-mark.bl { bottom: 12px; left: 12px; border-left: 1.5px solid rgba(14,165,233,0.4); border-bottom: 1.5px solid rgba(14,165,233,0.4); }
    .corner-mark.br { bottom: 12px; right: 12px; border-right: 1.5px solid rgba(14,165,233,0.4); border-bottom: 1.5px solid rgba(14,165,233,0.4); }

    .standby-screen { padding: 90px 0; text-align: center; }
    .standby-icon-wrap { margin-bottom: 16px; opacity: 0.14; }
    .standby-icon-wrap svg { width: 50px; height: 50px; stroke: #94a3b8; fill: none; stroke-width: 1; }
    .standby-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px; letter-spacing: 4px; color: #334155; margin: 0 0 6px;
    }
    .standby-sub {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10px; letter-spacing: 2px; color: #475569;
    }

    /* ===== SLIDER ===== */
    .stSlider > div > div > div > div { background: #0ea5e9 !important; }
    .stSlider label { font-family: 'Sora', sans-serif !important; font-size: 12px !important; color: #475569 !important; }

    /* ===== FILE UPLOADER ===== */
    .stFileUploader {
        border: none !important;
        border-radius: 12px !important; background: transparent !important;
    }
    
    /* File uploader button styling */
    .stFileUploader [data-testid="stFileUploaderDropzone"] {
        border: none !important;
    }
    
    .stFileUploader button {
        background: linear-gradient(135deg, #0ea5e9 0%, #0284c7 100%) !important;
        border: none !important;
        color: #fff !important;
        font-weight: 600 !important;
        padding: 12px 24px !important;
        border-radius: 9px !important;
        box-shadow: 0 4px 14px rgba(14,165,233,0.28) !important;
    }
    
    .stFileUploader button:hover {
        box-shadow: 0 6px 20px rgba(14,165,233,0.38) !important;
    }

    /* ===== SELECT ===== */
    .stSelectbox label, .stRadio label {
        font-family: 'Sora', sans-serif !important;
        font-size: 12px !important; color: #475569 !important;
    }
    [data-baseweb="select"] { background: #fff !important; border-color: #e2e8f0 !important; }

    /* ===== METRICS ===== */
    [data-testid="stMetric"] {
        background: #fff; border: 1px solid #e2e8f0;
        border-radius: 12px; padding: 14px 18px !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.045);
    }
    [data-testid="stMetricLabel"] { font-family: 'Sora', sans-serif !important; color: #64748b !important; font-size: 12px !important; }
    [data-testid="stMetricValue"] {
        font-family: 'Sora', sans-serif !important; color: #0f172a !important;
        font-size: 26px !important; font-weight: 700 !important; letter-spacing: -0.8px !important;
    }

    /* ===== HISTORY ROWS ===== */
    .incident-row {
        background: #fff; border: 1px solid #e2e8f0;
        border-radius: 12px; padding: 15px 17px;
        margin-bottom: 8px; transition: all 0.2s ease;
        box-shadow: 0 1px 3px rgba(0,0,0,0.038);
    }
    .incident-row:hover { border-color: #7dd3fc; background: #f8fbff; }
    .alert-badge {
        display: inline-flex; align-items: center; gap: 5px;
        padding: 4px 11px; border-radius: 100px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10px; font-weight: 500; letter-spacing: 1px;
        background: #fee2e2; color: #dc2626;
    }

    /* ===== SIDEBAR HIDE ===== */
    section[data-testid="stSidebar"] { display: none !important; }

    /* ===== FOOTER ===== */
    .app-footer {
        margin-top: 52px; padding: 16px 0;
        border-top: 1px solid #e2e8f0;
        display: flex; justify-content: space-between; align-items: center;
    }
    .app-footer span { font-family: 'IBM Plex Mono', monospace; font-size: 9px; letter-spacing: 2px; color: #cbd5e1; }

    /* ===== EXTRAS ===== */
    .stDeployButton { display: none !important; }
    #MainMenu { visibility: hidden; }
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: #f1f5f9; }
    ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #94a3b8; }
    [data-testid="stVegaLiteChart"], .stLineChart {
        background: #fff !important; border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important; padding: 10px !important;
    }
    [data-testid="stDownloadButton"] button {
        background: #f0f9ff !important; border: 1px solid #bae6fd !important;
        color: #0369a1 !important; font-weight: 600 !important;
    }
    [data-testid="stDownloadButton"] button:hover { background: #e0f2fe !important; }
    [data-testid="stInfo"] {
        background: #f0f9ff !important; border: 1px solid #bae6fd !important;
        border-radius: 10px !important; color: #0369a1 !important;
    }
    /* ===== DATE INPUT LIGHT BACKGROUND ===== */
    [data-testid="stDateInput"] input,
    [data-baseweb="input"] input {
        background: #f8fafc !important;
        border-color: #e2e8f0 !important;
        color: #334155 !important;
        border-radius: 8px !important;
    }
    [data-baseweb="calendar"] {
        background: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
    }
    </style>
    """,
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------------- SPLASH
    if not st.session_state.splash_done:
        st.markdown(
            """
        <style>
        .block-container { padding: 0 !important; max-width: 100% !important; }
        </style>
        <div class="splash-overlay">
            <div class="splash-glow"></div>
            <div class="splash-glow2"></div>
            <div class="splash-inner">
                <div class="splash-logo">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                        <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                    </svg>
                </div>
                <div class="splash-brand-row">
                    <span class="splash-preempt">Smart Surveillance System</span><span class="splash-ai"></span>
                </div>
                <div class="splash-tagline">With Anomaly Detection</div>
                <div class="splash-progress-track">
                    <div class="splash-progress-fill"></div>
                </div>
            </div>
            <div class="splash-footer-text">Smart Surveillance System &nbsp;·&nbsp; v3.2.1 &nbsp;·&nbsp; THREAT DETECTION ENGINE</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        time.sleep(4)
        st.session_state.splash_done = True
        st.rerun()
        return None, None, False, None

    # ---------------------------------------------------------------- NAVBAR
    st.markdown(
        """
    <div class="top-navbar">
        <div class="navbar-brand">
            <div class="navbar-logo-icon">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                    <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>
                </svg>
            </div>
            <div class="navbar-brand-text">Smart <span>Surveillance <span>System </span> </div>
        </div>
        <div class="navbar-right">
            <div class="system-status-badge">
                <div class="status-dot"></div>
                <span class="status-label">SYSTEM ONLINE</span>
            </div>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # ---------------------------------------------------------------- NAV TABS
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    nc1, nc2, nc3, nc_sp = st.columns([1, 1, 1, 6])
    with nc1:
        if st.button("  Dashboard", key="nav_dashboard", use_container_width=True):
            st.session_state.current_page = "dashboard"
            st.rerun()
    with nc2:
        if st.button("  Live Feed", key="nav_live", use_container_width=True):
            st.session_state.current_page = "live_feed"
            st.rerun()
    with nc3:
        if st.button("  History", key="nav_history", use_container_width=True):
            st.session_state.current_page = "history"
            st.rerun()
    st.markdown("<div style='height: 4px;'></div>", unsafe_allow_html=True)

    conf_thresh = 0.5
    video_source = None
    frame_placeholder = None

    # ============================================================== DASHBOARD
    if st.session_state.current_page == "dashboard":

        st.markdown(
            """
        <div class="welcome-section">
            <div class="welcome-eyebrow">COMMAND CENTER</div>
            <div class="welcome-heading">Welcome Smart Surveillance System</div>
            <div class="welcome-sub">Real-time Surveillance &amp; Threat Detection</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # DELETE the 4 stat card columns (sc1–sc4) and replace with this:

        st.markdown(
            """
        <div style="margin: 10px 0 28px;">

        <div style="margin-bottom: 18px;">
            <div class="welcome-eyebrow">SECURITY INTELLIGENCE</div>
        </div>

        <h1 style="font-family:'Sora',sans-serif; font-size:36px; font-weight:800;
                    letter-spacing:-1.2px; color:#0f172a; margin-bottom:18px;
                    animation: fadeUp 0.7s ease-out 0.1s both;">
            Security That <span style="color:#0ea5e9;">Thinks</span> For You
        </h1>

        <p style="font-family:'Sora',sans-serif; font-size:14px; color:#475569;
                    line-height:1.8; max-width:780px;
                    animation: fadeUp 0.7s ease-out 0.4s both;">
            Why rely on manual monitoring when intelligent systems can do it faster, better,
            and more accurately? Our solution continuously processes video feeds, detects anomalies,
            tracks individuals, and analyses movement patterns — all in real time.
            <br><br>
            Designed to handle dynamic and real-world scenarios, the system ensures that suspicious
            activities such as unusual movements, loitering, or potential shoplifting are identified
            instantly. With automated alerts and insights, it helps prevent incidents before they escalate.
        </p>

        <div style="display:flex; flex-direction:column; gap:10px; margin-top:20px;
                    animation: fadeUp 0.7s ease-out 0.75s both;">
            <div style="display:flex; align-items:flex-start; gap:10px; font-family:'Sora',sans-serif;
                        font-size:13px; color:#334155; font-weight:500;">
            <div style="width:7px; height:7px; border-radius:50%; background:#0ea5e9;
                        flex-shrink:0; margin-top:5px;"></div>
            Instant anomaly detection without delays
            </div>
            <div style="display:flex; align-items:flex-start; gap:10px; font-family:'Sora',sans-serif;
                        font-size:13px; color:#334155; font-weight:500;">
            <div style="width:7px; height:7px; border-radius:50%; background:#0ea5e9;
                        flex-shrink:0; margin-top:5px;"></div>
            Reliable tracking across multiple frames
            </div>
            <div style="display:flex; align-items:flex-start; gap:10px; font-family:'Sora',sans-serif;
                        font-size:13px; color:#334155; font-weight:500;">
            <div style="width:7px; height:7px; border-radius:50%; background:#0ea5e9;
                        flex-shrink:0; margin-top:5px;"></div>
            Real-time insights for proactive action
            </div>
        </div>

        <div style="margin-top:22px; font-family:'Sora',sans-serif; font-size:14px;
                    font-weight:600; color:#0ea5e9;
                    animation: fadeUp 0.7s ease-out 1.05s both;">
            👉 Let AI take control of your surveillance.
        </div>

        </div>

        <style>
        @keyframes fadeUp {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
        }
        </style>
        """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="section-label">Quick Actions</div>', unsafe_allow_html=True)
        qa1, qa2, qa3 = st.columns(3)

        with qa1:
            st.markdown(
                """
            <div class="quick-action-card">
                <div class="qa-icon" style="background:#f0f9ff">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" stroke="#0ea5e9">
                        <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
                        <circle cx="12" cy="13" r="4"/>
                    </svg>
                </div>
                <div class="qa-title">Start Live Monitoring</div>
                <div class="qa-desc">Connect your webcam and begin real-time threat detection with bounding box overlays.</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
            st.markdown('<div class="action-btn">', unsafe_allow_html=True)
            if st.button("Launch Live Feed →", key="qa_live", use_container_width=True):
                st.session_state.current_page = "live_feed"
                st.session_state.video_source_type = "webcam"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        with qa2:
            st.markdown(
                """
            <div class="quick-action-card">
                <div class="qa-icon" style="background:#f0fdf4">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" stroke="#22c55e">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                        <polyline points="17 8 12 3 7 8"/>
                        <line x1="12" y1="3" x2="12" y2="15"/>
                    </svg>
                </div>
                <div class="qa-title">Upload Footage</div>
                <div class="qa-desc">Analyse pre-recorded video files for incidents. Supports MP4, AVI, and MOV formats.</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
            st.markdown('<div class="action-btn">', unsafe_allow_html=True)
            if st.button("Upload & Analyse →", key="qa_upload", use_container_width=True):
                st.session_state.current_page = "live_feed"
                st.session_state.video_source_type = "upload"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        with qa3:
            st.markdown(
                """
            <div class="quick-action-card">
                <div class="qa-icon" style="background:#fffbeb">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" stroke="#f59e0b">
                        <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                        <line x1="3" y1="9" x2="21" y2="9"/>
                        <line x1="9" y1="21" x2="9" y2="9"/>
                    </svg>
                </div>
                <div class="qa-title">View History</div>
                <div class="qa-desc">Browse all logged anomaly records, filter by date or person, and download CSV reports.</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
            st.markdown('<div class="action-btn">', unsafe_allow_html=True)
            if st.button("Open History →", key="qa_history", use_container_width=True):
                st.session_state.current_page = "history"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            """
        <div class="info-banner">
            <div class="info-banner-icon">
                <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                    <circle cx="12" cy="12" r="10"/>
                    <line x1="12" y1="8" x2="12" y2="12"/>
                    <line x1="12" y1="16" x2="12.01" y2="16"/>
                </svg>
            </div>
            <div>
                <div class="info-banner-title">AI Detection Engine Ready</div>
                <div class="info-banner-body">
                    Powered by DETR ResNet-50 for object detection, ByteTrack for multi-person tracking,
                    and a dedicated pose-estimation module. All detections are logged in real time.
                </div>
            </div>
        </div>
        <div class="app-footer">
            <span>Smart Surveillance System &nbsp;·&nbsp; v3.2.1</span>
            
        </div>
        """,
            unsafe_allow_html=True,
        )

        return conf_thresh, video_source, st.session_state.run, st.empty()

    # ============================================================== LIVE FEED
    elif st.session_state.current_page == "live_feed":
        st.markdown(
            """
        <div class="page-header">
            <div class="page-title">Live Monitoring</div>
            <div class="page-subtitle">Real-time anomaly detection — choose a video source to begin</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        if st.session_state.video_source_type is None:
            lc1, lc2 = st.columns(2)
            with lc1:
                st.markdown(
                    """
                <div class="source-select-card">
                    <div class="source-icon-box">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                            <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
                            <circle cx="12" cy="13" r="4"/>
                        </svg>
                    </div>
                    <div class="source-card-title">Live Webcam</div>
                    <div class="source-card-desc">Stream from your device camera with real-time bounding box overlays and pose estimation.</div>
                    <div class="feature-list">
                        <div class="feature-row"><div class="feature-pip"></div>Real-time DETR object detection</div>
                        <div class="feature-row"><div class="feature-pip"></div>Pose estimation per person</div>
                        <div class="feature-row"><div class="feature-pip"></div>Anomaly alert generation</div>
                        <div class="feature-row"><div class="feature-pip"></div>Detection log with timestamps</div>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
                st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
                if st.button("Select Webcam →", key="sel_webcam", use_container_width=True):
                    st.session_state.video_source_type = "webcam"
                    st.session_state.stat_cameras = 1
                    st.session_state.stat_health = "Online"
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

            with lc2:
                st.markdown(
                    """
                <div class="source-select-card">
                    <div class="source-icon-box">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                            <polyline points="17 8 12 3 7 8"/>
                            <line x1="12" y1="3" x2="12" y2="15"/>
                        </svg>
                    </div>
                    <div class="source-card-title">Upload a Video</div>
                    <div class="source-card-desc">Upload a pre-recorded video (MP4, AVI, MOV) and process it frame by frame with overlays.</div>
                    <div class="feature-list">
                        <div class="feature-row"><div class="feature-pip"></div>Multi-frame video processing</div>
                        <div class="feature-row"><div class="feature-pip"></div>Shoplifting &amp; loitering detection</div>
                        <div class="feature-row"><div class="feature-pip"></div>Frame-by-frame replay with overlays</div>
                        <div class="feature-row"><div class="feature-pip"></div>Full incident report saved to DB</div>
                    </div>
                </div>
                """,
                    unsafe_allow_html=True,
                )
                st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
                if st.button("Select Upload →", key="sel_upload", use_container_width=True):
                    st.session_state.video_source_type = "upload"
                    st.session_state.stat_cameras = 1
                    st.session_state.stat_health = "Online"
                    st.rerun()
                st.markdown('</div>', unsafe_allow_html=True)

            return conf_thresh, video_source, st.session_state.run, st.empty()

        else:
            if st.button("← Back to Source Selection", key="back_btn"):
                st.session_state.video_source_type = None
                st.session_state.run = False
                st.session_state.stat_cameras = 0
                st.session_state.stat_health = "Offline"
                st.rerun()

            st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
            cc1, cc2, cc3 = st.columns([3, 1, 1])
            with cc1:
                conf_thresh = st.slider("Detection Confidence Threshold", 0.10, 1.00, 0.50, key="conf_slider")
            with cc2:
                st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
                if st.button("▶  START", key="start_btn", use_container_width=True):
                    st.session_state.run = True
                    st.session_state.stat_health = "Online"
                    st.session_state.stat_cameras = 1
                st.markdown('</div>', unsafe_allow_html=True)
            with cc3:
                st.markdown('<div class="danger-btn">', unsafe_allow_html=True)
                if st.button("⏹  STOP", key="stop_btn", use_container_width=True):
                    st.session_state.run = False
                st.markdown('</div>', unsafe_allow_html=True)

            if st.session_state.video_source_type == "upload":
                uploaded_file = st.file_uploader("Upload Video File", type=["mp4", "avi", "mov"], key="video_uploader")
                if uploaded_file:
                    tmp = tempfile.NamedTemporaryFile(delete=False)
                    tmp.write(uploaded_file.read())
                    video_source = tmp.name
            else:
                video_source = 0

            feed_label = "WEBCAM" if st.session_state.video_source_type == "webcam" else "UPLOADED VIDEO"
            run_label  = "LIVE" if st.session_state.run else "STANDBY"

            st.markdown(
                f"""
            <div class="video-shell">
                <div class="corner-mark tl"></div><div class="corner-mark tr"></div>
                <div class="corner-mark bl"></div><div class="corner-mark br"></div>
                <div class="video-hud-bar">
                    <div class="hud-label">FEED 01 &nbsp;&bull;&nbsp; {feed_label}</div>
                    <div class="hud-rec"><div class="rec-pip"></div>{run_label}</div>
                </div>
            """,
                unsafe_allow_html=True,
            )

            frame_placeholder = st.empty()

            if not st.session_state.run:
                st.markdown(
                    """
                <div class="standby-screen">
                    <div class="standby-icon-wrap">
                        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                            <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/>
                            <circle cx="12" cy="12" r="3"/>
                        </svg>
                    </div>
                    <div class="standby-title">AWAITING INPUT</div>
                    <div class="standby-sub">Press START to begin surveillance</div>
                </div>
                """,
                    unsafe_allow_html=True,
                )

            st.markdown(
                f"""
                <div class="video-hud-bottom-bar">
                    <div class="hud-label">CONF: {conf_thresh:.0%}</div>
                    <div class="hud-label">MODEL: DETR-R50</div>
                    <div class="hud-label">TRACKER: BYTETRACK</div>
                </div>
            </div>
            <div class="app-footer">
                <span>Smart Surveillance System &nbsp;·&nbsp; v3.2.1</span>
                <span>&copy; 2024 PROACTIVE INTELLIGENCE FOR SAFER SPACES</span>
            </div>
            """,
                unsafe_allow_html=True,
            )

            return conf_thresh, video_source, st.session_state.run, frame_placeholder

    # ================================================================ HISTORY
    elif st.session_state.current_page == "history":
        from mongo_db import get_all_anomalies
        import pandas as pd

        st.markdown(
            """
        <div class="page-header">
            <div class="page-title">Anomaly History</div>
            <div class="page-subtitle">Review and analyse all past detected incidents</div>
        </div>
        """,
            unsafe_allow_html=True,
        )
        data = get_all_anomalies()
        render_history_page(data, pd)
        return conf_thresh, video_source, st.session_state.run, st.empty()


def render_history_page(data, pd):
    """Render the history page with anomaly data."""
    import io, os, base64
    from datetime import datetime, timedelta

    if len(data) == 0:
        st.info("🔍 No anomalies recorded yet. Start monitoring to capture incidents.")
        return

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    today_count = len(df[df["timestamp"].dt.date == datetime.now().date()])
    st.session_state.stat_incidents = len(df)
    st.session_state.stat_alerts   = today_count

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total Incidents", len(df))
    with c2: st.metric("Today", today_count)
    with c3: st.metric("This Week", len(df[df["timestamp"] >= datetime.now() - timedelta(days=7)]))
    with c4: st.metric("Unique Persons", df["person_id"].nunique())

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    st.markdown("""
    <style>
    [data-testid="stDateInput"] label,
    [data-testid="stSelectbox"] label {
        color: #0ea5e9 !important;
        font-family: 'Sora', sans-serif !important;
        font-size: 12px !important;
        font-weight: 600 !important;
    }
    [data-baseweb="calendar"] * { color: #0ea5e9 !important; }
    [data-baseweb="calendar"] [aria-selected="true"] {
        background: #0ea5e9 !important;
        color: #fff !important;
    }
    [data-baseweb="calendar"] button:hover {
        background: #e0f2fe !important;
        color: #0369a1 !important;
    }
    [data-testid="stDateInput"] input,
    [data-baseweb="input"] input {
        background: #f0f9ff !important;
        border-color: #bae6fd !important;
        color: #0369a1 !important;
        border-radius: 8px !important;
    }
    [data-baseweb="select"] {
        background: #f0f9ff !important;
        border-color: #bae6fd !important;
        border-radius: 8px !important;
    }
    [data-baseweb="select"] div {
        background: #f0f9ff !important;
        color: #0369a1 !important;
        font-family: 'Sora', sans-serif !important;
    }
    </style>
    """, unsafe_allow_html=True)

    f1, f2, f3 = st.columns([2, 2, 1])
    with f1:
        date_range = st.date_input("📅 Date Range",
            [df["timestamp"].min().date(), df["timestamp"].max().date()], key="date_filter")
    with f2:
        person_ids = ["All"] + sorted(df["person_id"].unique().tolist())
        selected_person = st.selectbox("👤 Person ID", person_ids, key="person_filter")
    with f3:
        st.markdown("<div style='height: 26px;'></div>", unsafe_allow_html=True)
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        st.download_button("📥 Export CSV", buf.getvalue(),
            f"preempt_report_{datetime.now().strftime('%Y%m%d')}.csv", "text/csv",
            key="hist_export_csv",
            use_container_width=True)

    filtered = df.copy()
    if len(date_range) == 2:
        filtered = filtered[
            (filtered["timestamp"].dt.date >= date_range[0]) &
            (filtered["timestamp"].dt.date <= date_range[1])]
    if selected_person != "All":
        filtered = filtered[filtered["person_id"] == selected_person]

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-label" style="font-size:15px;margin-top:0">Detection Trend</div>', unsafe_allow_html=True)
    t = filtered.copy()
    t["date"] = t["timestamp"].dt.date
    st.line_chart(t.groupby("date").size())

    st.markdown("<div style='height: 18px;'></div>", unsafe_allow_html=True)
    st.markdown(
        f'<div class="section-label" style="font-size:15px;margin-top:0">Incident Records '
        f'<span style="font-size:12px;font-weight:400;color:#64748b">({len(filtered)} shown)</span></div>',
        unsafe_allow_html=True)

    for _, item in filtered.iterrows():
        person_id  = item.get('person_id', 'N/A')
        timestamp  = item.get('timestamp', 'N/A')
        video_path = item.get('video_path', 'N/A')
        snapshot   = item.get('snapshot', '')

        has_image = snapshot and os.path.exists(str(snapshot))

        if has_image:
            with open(str(snapshot), "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            ext = str(snapshot).split(".")[-1].lower()
            mime = "image/png" if ext == "png" else "image/jpeg"
            img_html = (
                "<img src='data:" + mime + ";base64," + b64 + "' "
                "style='width:88px;height:56px;object-fit:cover;"
                "border-radius:8px;border:1px solid #e2e8f0;display:block;'/>"
            )
        else:
            img_html = (
                "<div style='width:88px;height:56px;background:#f1f5f9;border-radius:8px;"
                "display:flex;align-items:center;justify-content:center;"
                "font-size:17px;border:1px solid #e2e8f0;'>🖼️</div>"
            )

        row_html = (
            '<div class="incident-row" style="display:flex;align-items:center;gap:16px;flex-wrap:wrap;">'
            '<div style="flex-shrink:0;">'
            + img_html +
            '</div>'
            '<div style="flex:1;min-width:140px;">'
            '<p style="font-family:Sora,sans-serif;font-size:11px;color:#94a3b8;margin:0">Person ID</p>'
            '<p style="font-family:Sora,sans-serif;font-size:14px;font-weight:600;color:#0ea5e9;margin:2px 0 6px">' + str(person_id) + '</p>'
            '<p style="font-family:IBM Plex Mono,monospace;font-size:10px;color:#94a3b8;margin:0">' + str(timestamp) + '</p>'
            '</div>'
            '<div style="flex:2;min-width:160px;">'
            '<p style="font-family:Sora,sans-serif;font-size:11px;color:#94a3b8;margin:0">Source</p>'
            '<p style="font-family:Sora,sans-serif;font-size:12px;color:#334155;margin:2px 0 0;word-break:break-all">' + str(video_path) + '</p>'
            '</div>'
            '<div style="flex-shrink:0;padding-top:4px;">'
            '<span class="alert-badge">&#9679; ALERT</span>'
            '</div>'
            '</div>'
        )
        st.markdown(row_html, unsafe_allow_html=True)

    st.markdown(
        """
        <div class="app-footer">
            <span>Smart Surveillance System &nbsp;&middot;&nbsp; v3.2.1</span>
            <span>&copy; 2024 PROACTIVE INTELLIGENCE FOR SAFER SPACES</span>
        </div>
        """,
        unsafe_allow_html=True
    )

