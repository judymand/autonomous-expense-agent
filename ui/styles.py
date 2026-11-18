"""Visual system for the Hebrew expense dashboard."""

APP_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Heebo:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: "Heebo", "Arial Hebrew", "Noto Sans Hebrew", sans-serif;
  }
  .stApp { background: #F8FAFC; color: #0F172A; }
  .block-container { padding-top: 1.1rem; max-width: 1240px; }

  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
  header[data-testid="stHeader"] { background: transparent; }
  [data-testid="stToolbar"],
  [data-testid="stDecoration"],
  .stAppDeployButton { display: none; }

  .hero { direction: rtl; text-align: right; margin-bottom: 0.35rem; }
  .hero h1 {
    margin: 0;
    font-size: 1.65rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: #0F172A;
  }
  .badge {
    display: inline-block;
    margin-top: 0.45rem;
    padding: 0.28rem 0.7rem;
    border-radius: 999px;
    background: #EFF6FF;
    color: #1D4ED8;
    border: 1px solid #DBEAFE;
    font-size: 0.82rem;
    font-weight: 600;
  }
  .panel-title, .section-label {
    direction: rtl;
    text-align: right;
    font-size: 0.95rem;
    font-weight: 700;
    color: #0F172A;
    margin: 0.15rem 0 0.55rem;
  }
  .section-label {
    font-size: 0.78rem;
    font-weight: 600;
    color: #475569;
  }

  div[data-testid="stMetric"] {
    direction: rtl;
    text-align: right;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 0.85rem 1rem 0.7rem;
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
  }
  div[data-testid="stMetric"] label {
    color: #64748B;
    font-weight: 600;
  }
  div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #0F172A;
    font-weight: 700;
  }

  .chat-log { direction: rtl; text-align: right; }
  .bubble {
    direction: rtl;
    text-align: right;
    display: inline-block;
    max-width: 88%;
    border-radius: 16px;
    padding: 0.72rem 0.9rem;
    margin: 0.28rem 0;
    line-height: 1.5;
  }
  .bubble.user {
    background: #2563EB;
    color: #FFFFFF;
    margin-inline-end: 0;
    margin-inline-start: 12%;
  }
  .bubble.assistant {
    background: #FFFFFF;
    color: #0F172A;
    border: 1px solid #E2E8F0;
    box-shadow: 0 4px 14px rgba(15, 23, 42, 0.04);
    margin-inline-start: 0;
    margin-inline-end: 12%;
  }

  .stButton > button {
    border-radius: 10px;
    font-weight: 600;
  }
  .stButton > button[kind="primary"] {
    background: #2563EB;
    border: 1px solid #2563EB;
    color: #FFFFFF;
  }
  .st-key-approve_expense button {
    background: #16A34A !important;
    border: 1px solid #16A34A !important;
    color: #FFFFFF !important;
  }
  .st-key-reject_expense button {
    background: #FEF2F2 !important;
    border: 1px solid #FECACA !important;
    color: #B91C1C !important;
  }
  [data-testid="stFileUploader"] {
    background: #EFF6FF;
    border: 1px dashed #BFDBFE;
    border-radius: 10px;
    padding: 0.35rem 0.5rem 0.6rem;
  }
  [data-testid="stFileUploader"] section {
    border: none;
    background: transparent;
  }
  [data-testid="stFileUploader"] button {
    border-radius: 10px;
    background: #2563EB;
    color: #FFFFFF;
    border: none;
  }

  .approval-card {
    direction: rtl;
    text-align: right;
    background: linear-gradient(180deg, #FFFBEB 0%, #EFF6FF 100%);
    border: 1px solid #FCD34D;
    border-radius: 14px;
    padding: 0.9rem 1rem;
    margin: 0.55rem 0 0.75rem;
    box-shadow: 0 10px 24px rgba(37, 99, 235, 0.08);
  }
  .approval-kicker { color: #B45309; font-size: 0.78rem; font-weight: 700; }
  .approval-merchant { font-size: 1.12rem; font-weight: 700; color: #0F172A; }
  .approval-amount { font-size: 1.45rem; font-weight: 700; color: #1D4ED8; }
  .approval-meta { color: #475569; }

  .pipeline {
    direction: ltr;
    text-align: left;
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0.35rem;
    margin: 0.15rem 0 0.85rem;
  }
  .node {
    border-radius: 999px;
    padding: 0.32rem 0.75rem;
    background: #F1F5F9;
    color: #64748B;
    border: 1px solid #E2E8F0;
    font-size: 0.8rem;
    font-weight: 600;
  }
  .node.done { background: #ECFDF5; color: #047857; border-color: #A7F3D0; }
  .node.now { background: #2563EB; color: #FFFFFF; border-color: #2563EB; }
  .pipe-arrow { color: #94A3B8; font-weight: 700; }

  .gauge { height: 10px; border-radius: 999px; background: #E2E8F0; overflow: hidden; }
  .gauge-fill { height: 100%; border-radius: 999px; }
  .gauge-fill.high { background: #2563EB; }
  .gauge-fill.low { background: #D97706; }
  .gauge-label { direction: rtl; text-align: right; margin-top: 0.35rem; font-weight: 700; color: #0F172A; }

  div[data-testid="stJson"],
  div[data-testid="stCode"],
  div[data-testid="stCodeBlock"],
  pre, code {
    direction: ltr;
    text-align: left;
  }
</style>
"""
