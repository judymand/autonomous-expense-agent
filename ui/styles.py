"""Visual system for the Hebrew expense dashboard."""

APP_CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Heebo:wght@400;500;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: "Heebo", "Arial Hebrew", "Noto Sans Hebrew", sans-serif;
  }
  .stApp { background: #F8FAFC; color: #0F172A; }
  .block-container,
  [data-testid="stMainBlockContainer"] {
    padding-top: 1.1rem;
    padding-left: 24px;
    padding-right: 24px;
    max-width: none;
  }

  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
  header[data-testid="stHeader"] { background: transparent; }
  [data-testid="stToolbar"],
  [data-testid="stDecoration"],
  .stAppDeployButton { display: none; }

  .st-key-page_header {
    direction: rtl;
    width: 100%;
    box-sizing: border-box;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 14px;
    padding: 0.45rem 1.1rem;
    margin-bottom: 0.9rem;
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.05);
  }
  .page-header-copy {
    display: flex;
    align-items: center;
    justify-content: flex-start;
    gap: 12px;
    direction: rtl;
    width: 100%;
  }
  .st-key-page_header .stMarkdown,
  .st-key-page_header .stMarkdown div {
    margin-top: 0 !important;
    margin-bottom: 0 !important;
  }
  .st-key-page_header h1 {
    margin: 0;
    padding: 0;
    font-size: 1.65rem;
    line-height: 1.2;
    font-weight: 700;
    letter-spacing: -0.03em;
    color: #0F172A;
    white-space: nowrap;
  }
  .st-key-page_header [data-testid="stHeaderActionElements"] { display: none; }
  .badge {
    display: inline-block;
    margin: 0;
    padding: 0.28rem 0.7rem;
    border-radius: 999px;
    background: #EFF6FF;
    color: #1D4ED8;
    border: 1px solid #DBEAFE;
    font-size: 0.82rem;
    font-weight: 600;
  }
  .st-key-new_chat {
    width: 100%;
    display: flex;
    direction: ltr;
    justify-content: flex-start;
  }
  .st-key-new_chat button {
    background: #FFFFFF !important;
    color: #334155 !important;
    border: 1px solid #CBD5E1 !important;
    border-radius: 10px;
    font-size: 0.82rem;
    font-weight: 600;
    padding: 0.28rem 0.75rem;
    min-height: 2rem;
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
    min-height: 136px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
  }
  div[data-testid="stMetric"] label {
    color: #64748B;
    font-weight: 600;
  }
  div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: #0F172A;
    font-weight: 700;
  }

  .st-key-chat_transcript {
    padding: 16px 20px;
    box-sizing: border-box;
  }

  .chat-log {
    display: flex;
    flex-direction: column;
    width: 100%;
    direction: ltr;
  }
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
    align-self: flex-end;
    background: #2563EB;
    color: #FFFFFF;
  }
  .bubble.assistant {
    align-self: flex-start;
    background: #F1F5F9;
    color: #0F172A;
  }
  .bubble-label {
    direction: ltr;
    text-align: left;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    color: #64748B;
    margin-bottom: 0.22rem;
  }
  [data-testid="stChatInputTextArea"] {
    direction: rtl;
    text-align: right;
  }
  [data-testid="stChatInputTextArea"]::placeholder {
    direction: rtl;
    text-align: right;
    unicode-bidi: isolate;
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
  .st-key-spend_heading { direction: rtl; }
  .st-key-spend_heading .section-label { margin: 0 0 0.35rem; }

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

  [data-baseweb="tab-list"],
  [data-testid="stTabs"] [role="tablist"] {
    direction: rtl !important;
  }
  [data-baseweb="tab"] p,
  [data-testid="stTab"] p {
    direction: rtl;
    unicode-bidi: isolate;
  }

  .empty-state {
    background: #F8FAFC;
    border: 1px dashed #CBD5E1;
    border-radius: 12px;
    padding: 20px;
    text-align: center;
    color: #64748B;
    direction: rtl;
    margin: 0.2rem 0 0.7rem;
  }

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

  .rule-list {
    display: flex;
    flex-direction: column;
    gap: 0.55rem;
    width: 100%;
  }
  .rule-card {
    direction: rtl;
    text-align: right;
    width: 100%;
    box-sizing: border-box;
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 12px;
    padding: 0.75rem 0.9rem;
    white-space: normal;
    overflow: visible;
  }
  .rule-top {
    display: flex;
    flex-direction: row;
    align-items: center;
    justify-content: space-between;
    gap: 0.6rem;
  }
  .rule-badge {
    display: inline-block;
    background: #EFF6FF;
    color: #1D4ED8;
    border: 1px solid #DBEAFE;
    border-radius: 999px;
    padding: 0.12rem 0.6rem;
    font-size: 0.75rem;
    font-weight: 700;
    white-space: nowrap;
  }
  .rule-merchant {
    font-weight: 700;
    color: #0F172A;
  }
  .rule-text {
    margin: 0.45rem 0 0;
    color: #0F172A;
    line-height: 1.55;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    word-break: break-word;
  }

  div[data-testid="stJson"],
  div[data-testid="stCode"],
  div[data-testid="stCodeBlock"],
  pre, code {
    direction: ltr;
    text-align: left;
  }
</style>
"""
