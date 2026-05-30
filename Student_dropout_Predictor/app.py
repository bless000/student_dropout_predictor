"""
Student Early Predictor System — Streamlit App
Uses Project_Dataset.csv column names: Matric_no, Student_name, Age, Gender, GPA,
Attendance_score, Level, Assignment_score, Test_score, Class_participation, Financial_support, Dropout
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from datetime import datetime

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Early Predictor System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Credentials ───────────────────────────────────────────────────────────────
USERS = {"admin": "admin123", "staff": "staff2024", "demo": "demo"}

# ── Load model ────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "dropout_model.joblib")

@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)

# ── Colours ───────────────────────────────────────────────────────────────────
PURPLE   = "#7C3AED"
PURPLE_L = "#A78BFA"
GREEN    = "#10B981"
RED      = "#EF4444"
AMBER    = "#F59E0B"
BG       = "#F5F3FF"
CARD     = "#FFFFFF"
SIDEBAR  = "#1E1B4B"
TEXT     = "#1E1B4B"

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=Inter:wght@400;500&display=swap');
  html, body, [class*="css"] {{ font-family:'Inter',sans-serif; background:{BG}; color:{TEXT}; }}

  section[data-testid="stSidebar"] {{ background:{SIDEBAR} !important; }}
  section[data-testid="stSidebar"] * {{ color:#E0E7FF !important; }}
  section[data-testid="stSidebar"] .stButton>button {{
      background:transparent; border:1px solid #4338CA; color:#E0E7FF !important;
      border-radius:8px; width:100%; text-align:left; padding:10px 16px;
      font-size:.9rem; transition:background .2s;
  }}
  section[data-testid="stSidebar"] .stButton>button:hover {{ background:#312E81; }}

  .metric-card {{ background:{CARD}; border-radius:16px; padding:24px 28px;
      box-shadow:0 2px 12px rgba(124,58,237,.08); }}
  .metric-icon {{ font-size:2rem; }}
  .metric-value {{ font-family:'Sora',sans-serif; font-size:2rem; font-weight:700; color:{TEXT}; }}
  .metric-label {{ font-size:.8rem; color:#6B7280; font-weight:500; letter-spacing:.05em; text-transform:uppercase; }}
  .metric-delta {{ font-size:.78rem; }}
  .up {{ color:{GREEN}; }} .down {{ color:{RED}; }}

  .section-title {{ font-family:'Sora',sans-serif; font-size:1.15rem; font-weight:700; color:{TEXT}; margin-bottom:4px; }}
  .section-sub {{ font-size:.82rem; color:#9CA3AF; margin-bottom:16px; }}

  .badge {{ display:inline-block; padding:3px 12px; border-radius:20px; font-size:.78rem; font-weight:600; letter-spacing:.04em; }}
  .badge-low    {{ background:#D1FAE5; color:#065F46; }}
  .badge-medium {{ background:#FEF3C7; color:#92400E; }}
  .badge-high   {{ background:#FEE2E2; color:#991B1B; }}
  .badge-active {{ background:#D1FAE5; color:#065F46; }}
  .badge-at-risk{{ background:#FEE2E2; color:#991B1B; }}

  .result-box {{ background:{CARD}; border-radius:16px; padding:28px 32px;
      box-shadow:0 2px 16px rgba(124,58,237,.1); border-left:5px solid {PURPLE}; }}
  .result-label {{ font-family:'Sora',sans-serif; font-size:1.4rem; font-weight:700; }}

  .stButton>button {{ background:{PURPLE}; color:white !important; border:none;
      border-radius:10px; font-weight:600; padding:10px 28px; transition:opacity .2s; }}
  .stButton>button:hover {{ opacity:.88; }}

  #MainMenu, footer {{ visibility:hidden; }}
  .stDeployButton {{ display:none; }}
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def risk_label(prob):
    if prob < 0.35: return "Low",    "badge-low",    "🟢"
    if prob < 0.65: return "Medium", "badge-medium", "🟡"
    return "High", "badge-high", "🔴"

def predict_single(bundle, row_dict):
    df = pd.DataFrame([row_dict])
    df['Gender']            = bundle['le_gender'].transform(df['Gender'])
    df['Financial_support'] = bundle['le_finance'].transform(df['Financial_support'])
    X = df[bundle['feature_cols']].values
    if bundle['uses_scaler']:
        X = bundle['scaler'].transform(X)
    return bundle['model'].predict_proba(X)[0][1]

def batch_predict(bundle, df_raw):
    df = df_raw.copy()
    required = bundle['feature_cols']
    missing  = [c for c in required if c not in df.columns]
    if missing:
        return None, f"Missing columns: {missing}"
    try:
        df['Gender']            = bundle['le_gender'].transform(df['Gender'].str.strip())
        df['Financial_support'] = bundle['le_finance'].transform(df['Financial_support'].str.strip())
    except Exception as e:
        return None, str(e)
    X = df[required].values
    if bundle['uses_scaler']:
        X = bundle['scaler'].transform(X)
    probs = bundle['model'].predict_proba(X)[:,1]
    out = df_raw.copy()
    out['Dropout_Probability'] = probs.round(3)
    out['Risk_Level']          = [risk_label(p)[0] for p in probs]
    return out, None

def stat_card(icon, value, label, delta_text="", delta_up=True):
    arrow = "▲" if delta_up else "▼"
    cls   = "up" if delta_up else "down"
    delta_html = f'<div class="metric-delta {cls}">{arrow} {delta_text}</div>' if delta_text else ""
    st.markdown(f"""
    <div class="metric-card">
      <div class="metric-icon">{icon}</div>
      <div class="metric-value">{value}</div>
      <div class="metric-label">{label}</div>
      {delta_html}
    </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  LOGIN
# ═══════════════════════════════════════════════════════════════════════════════
def login_page():
    st.markdown("""
    <div style="text-align:center;padding-top:40px;"><span style="font-size:3rem;">🎓</span></div>
    <div style="font-family:'Sora',sans-serif;font-size:1.6rem;font-weight:700;text-align:center;margin-top:8px;">Student Early Predictor</div>
    <div style="color:#9CA3AF;text-align:center;font-size:.87rem;margin-bottom:28px;">Sign in to access the dashboard</div>
    """, unsafe_allow_html=True)

    col = st.columns([1,2,1])[1]
    with col:
        with st.form("login_form"):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("Sign In →", use_container_width=True)
        if submitted:
            if username in USERS and USERS[username] == password:
                st.session_state['logged_in'] = True
                st.session_state['username']  = username
                st.rerun()
            else:
                st.error("❌ Invalid credentials. Try admin / admin123")
        st.markdown("""
        <div style="text-align:center;margin-top:16px;font-size:.8rem;color:#9CA3AF;">
          Demo accounts: <b>admin</b> / admin123 &nbsp;|&nbsp; <b>staff</b> / staff2024
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
def render_sidebar():
    with st.sidebar:
        

        pages = [("📊","Dashboard"),("🔍","Predict"),("📁","Upload & Batch"),("📈","Analytics"),("❓","Help")]
        for icon, name in pages:
            if st.button(f"{icon}  {name}", key=f"nav_{name}"):
                st.session_state['page'] = name
                st.rerun()

        st.markdown("<hr style='border-color:#312E81;margin:20px 0;'>", unsafe_allow_html=True)


        user = st.session_state.get('username','user').capitalize()
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;padding:8px 0;">
          <div style="width:34px;height:34px;border-radius:50%;background:{PURPLE};display:flex;align-items:center;
                      justify-content:center;font-weight:700;color:white;font-size:.9rem;">{user[0]}</div>
          <div>
            <div style="color:#E0E7FF;font-weight:600;font-size:.88rem;">{user}</div>
            <div style="color:#818CF8;font-size:.73rem;">School Admin</div>
          </div>
        </div>""", unsafe_allow_html=True)

        if st.button("🚪  Logout", key="logout"):
            st.session_state.clear()
            st.rerun()

    return st.session_state.get('page','Dashboard')


# ═══════════════════════════════════════════════════════════════════════════════
#  DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
def dashboard_page(bundle):
    user = st.session_state.get('username','Admin').capitalize()
    st.markdown(f"""
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:28px;">
      <div>
        <div style="font-family:'Sora',sans-serif;font-size:1.6rem;font-weight:700;">Hello {user} 👋</div>
        <div style="color:#9CA3AF;font-size:.85rem;">{datetime.now().strftime("%A, %d %B %Y")} · Early Warning Dashboard</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Load real stats from dataset
    DATASET_PATH = os.path.join(os.path.dirname(__file__), "Project_Dataset.csv")
    df = pd.read_csv(DATASET_PATH)
    total    = len(df)
    dropouts = int(df['Dropout'].sum())
    active   = total - dropouts
    rate     = f"{dropouts/total*100:.1f}%"

    c1,c2,c3,c4 = st.columns(4)
    with c1: stat_card("🎓", f"{total:,}",  "Total Students",  "Level 200", True)
    with c2: stat_card("⚠️", f"{dropouts}", "At-Risk Students","32% of class", False)
    with c3: stat_card("📉", rate,           "Dropout Rate",    "Current semester", False)
    with c4: stat_card("✅", f"{active:,}",  "Active Students", f"{active/total*100:.1f}% retention", True)

    st.markdown("<div style='margin-top:32px;'></div>", unsafe_allow_html=True)

    left, right = st.columns([1, 1.6])

    with left:
        st.markdown('<div class="section-title">⚡ Quick Predict</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Enter basic info for instant result</div>', unsafe_allow_html=True)
        with st.form("quick_form"):
            gpa  = st.slider("GPA", 0.0, 4.0, 2.5, 0.1)
            att  = st.slider("Attendance Score (1–10)", 1, 10, 6)
            asgn = st.slider("Assignment Score (1–10)", 1, 10, 6)
            test = st.slider("Test Score (1–20)", 1, 20, 12)
            cp   = st.slider("Class Participation (1–10)", 1, 10, 6)
            go   = st.form_submit_button("Predict Risk →", use_container_width=True)
        if go:
            row = dict(Age=18, Gender='Male', GPA=gpa, Attendance_score=att, Level=200,
                       Assignment_score=asgn, Test_score=test, Class_participation=cp,
                       Financial_support='Medium')
            prob  = predict_single(bundle, row)
            label, _, emoji = risk_label(prob)
            color = {"Low":GREEN,"Medium":AMBER,"High":RED}[label]
            st.markdown(f"""
            <div class="result-box" style="margin-top:12px;border-left-color:{color};">
              <div style="color:#9CA3AF;font-size:.8rem;font-weight:600;letter-spacing:.05em;">DROPOUT RISK</div>
              <div class="result-label">{emoji} {label} Risk</div>
              <div style="margin-top:8px;">
                <span style="font-size:.88rem;color:#6B7280;">Probability: </span>
                <strong style="color:{PURPLE};">{prob:.1%}</strong>
              </div>
            </div>""", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-title">📋 Recent Risk Assessments</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Live from Project Dataset</div>', unsafe_allow_html=True)

        # Show real students with predictions
        sample = df.sample(10, random_state=42).reset_index(drop=True)
        _, err = batch_predict(bundle, sample)
        result, _ = batch_predict(bundle, sample)

        hdr = '<div style="display:grid;grid-template-columns:1fr 1.6fr 0.8fr 0.8fr 0.8fr 0.8fr 1fr;gap:6px;padding:6px 10px;background:#F9FAFB;border-radius:8px;font-size:.72rem;font-weight:600;color:#6B7280;text-transform:uppercase;letter-spacing:.04em;margin-bottom:4px;">'
        for h in ["Matric No","Name","GPA","Attend.","Test","Assign.","Risk"]:
            hdr += f"<div>{h}</div>"
        hdr += "</div>"
        st.markdown(hdr, unsafe_allow_html=True)

        for _, row in result.iterrows():
            risk   = row['Risk_Level']
            r_cls  = f"badge-{'low' if risk=='Low' else 'medium' if risk=='Medium' else 'high'}"
            st.markdown(f"""
            <div style="display:grid;grid-template-columns:1fr 1.6fr 0.8fr 0.8fr 0.8fr 0.8fr 1fr;gap:6px;
                        padding:9px 10px;border-bottom:1px solid #F3F4F6;font-size:.8rem;align-items:center;">
              <div style="color:#6B7280;font-size:.72rem;">{row['Matric_no']}</div>
              <div style="font-weight:600;">{row['Student_name']}</div>
              <div>{row['GPA']}</div>
              <div>{row['Attendance_score']}/10</div>
              <div>{row['Test_score']}/20</div>
              <div>{row['Assignment_score']}/10</div>
              <div><span class="badge {r_cls}">{risk}</span></div>
            </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  PREDICT PAGE
# ═══════════════════════════════════════════════════════════════════════════════
def predict_page(bundle):
    st.markdown('<div class="section-title">🔍 Individual Student Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Fill in student details to assess dropout risk</div>', unsafe_allow_html=True)

    with st.form("predict_form"):
        c1,c2,c3 = st.columns(3)
        with c1:
            st.markdown("**Student Info**")
            matric = st.text_input("Matric No", placeholder="e.g. 220501005")
            name   = st.text_input("Student Name", placeholder="e.g. Amaka Okafor")
            age    = st.number_input("Age", 14, 35, 18)
            gender = st.selectbox("Gender", ["Male","Female"])
            fin    = st.selectbox("Financial Support", ["Low","Medium","High"])
        with c2:
            st.markdown("**Academic Performance**")
            gpa    = st.slider("GPA (0.0 – 4.0)", 0.0, 4.0, 2.5, 0.05)
            att    = st.slider("Attendance Score (1–10)", 1, 10, 6)
            asgn   = st.slider("Assignment Score (1–10)", 1, 10, 6)
        with c3:
            st.markdown("**Assessment**")
            test   = st.slider("Test Score (1–20)", 1, 20, 12)
            cp     = st.slider("Class Participation (1–10)", 1, 10, 6)
            st.markdown("<br><br>", unsafe_allow_html=True)
            submitted = st.form_submit_button("🔮 Predict Dropout Risk", use_container_width=True)

    if submitted:
        row = dict(Age=age, Gender=gender, GPA=gpa, Attendance_score=att, Level=200,
                   Assignment_score=asgn, Test_score=test, Class_participation=cp,
                   Financial_support=fin)
        prob  = predict_single(bundle, row)
        label, _, emoji = risk_label(prob)
        color = {"Low":GREEN,"Medium":AMBER,"High":RED}[label]

        st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
        r1, r2 = st.columns([1,2])
        with r1:
            student_display = f"{name} ({matric})" if name else matric or "Student"
            st.markdown(f"""
            <div class="result-box" style="border-left-color:{color};">
              <div style="color:#9CA3AF;font-size:.78rem;font-weight:600;letter-spacing:.06em;">PREDICTION RESULT</div>
              <div style="font-size:.85rem;color:#6B7280;margin:4px 0 8px;">{student_display}</div>
              <div class="result-label" style="color:{color};">{emoji} {label} Risk</div>
              <div style="margin-top:12px;font-size:1.8rem;font-weight:700;color:{PURPLE};">{prob:.1%}</div>
              <div style="font-size:.8rem;color:#9CA3AF;">Dropout Probability</div>
            </div>""", unsafe_allow_html=True)

        with r2:
            tips = {
                "Low":    ("✅ Student appears stable.",    ["Maintain regular check-ins","Encourage participation in activities"]),
                "Medium": ("⚠️ Moderate risk detected.",   ["Schedule a counselling session","Review financial aid options","Monitor attendance closely"]),
                "High":   ("🚨 High dropout risk!",        ["Immediate counsellor intervention","Contact parent/guardian","Enrol in academic support programme","Review financial aid urgently"]),
            }
            msg, actions = tips[label]
            actions_html = "".join(f"<li style='margin-bottom:4px;'>{a}</li>" for a in actions)
            st.markdown(f"""
            <div style="background:{CARD};border-radius:16px;padding:24px 28px;box-shadow:0 2px 12px rgba(124,58,237,.08);">
              <div style="font-weight:600;font-size:1rem;margin-bottom:10px;">{msg}</div>
              <div style="font-size:.83rem;color:#6B7280;font-weight:600;text-transform:uppercase;letter-spacing:.05em;margin-bottom:8px;">Recommended Actions</div>
              <ul style="margin:0;padding-left:18px;font-size:.87rem;color:{TEXT};">{actions_html}</ul>
            </div>""", unsafe_allow_html=True)

        # Score summary bar
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        fig, axes = plt.subplots(1,4, figsize=(12,1.2))
        fig.patch.set_facecolor('none')
        bars_data = [
            ("Attendance", att, 10),
            ("Assignment", asgn, 10),
            ("Test Score", test, 20),
            ("Class Part.", cp, 10),
        ]
        for ax, (lbl, val, mx) in zip(axes, bars_data):
            pct = val/mx
            bar_color = GREEN if pct >= 0.6 else AMBER if pct >= 0.4 else RED
            ax.barh(0, mx, color='#F3F4F6', height=0.5)
            ax.barh(0, val, color=bar_color, height=0.5)
            ax.set_xlim(0, mx)
            ax.axis('off')
            ax.set_title(f"{lbl}: {val}/{mx}", fontsize=9, color=TEXT, pad=4)
        plt.tight_layout(pad=1)
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ═══════════════════════════════════════════════════════════════════════════════
#  UPLOAD & BATCH
# ═══════════════════════════════════════════════════════════════════════════════
def upload_page(bundle):
    st.markdown('<div class="section-title">📁 Batch Prediction via CSV Upload</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Upload your student CSV to get bulk dropout predictions</div>', unsafe_allow_html=True)

    # Download the real project dataset as sample
    DATASET_PATH = os.path.join(os.path.dirname(__file__), "Project_Dataset.csv")
    with open(DATASET_PATH,'rb') as f:
        st.download_button("⬇️ Download Sample CSV (Project Dataset)", f, "Project_Dataset.csv", "text/csv")

    st.markdown(f"""
    <div style="background:#EEF2FF;border-radius:12px;padding:16px 20px;margin:16px 0;font-size:.85rem;color:#4338CA;">
      <strong>Required columns:</strong> Age, Gender, GPA, Attendance_score, Level, Assignment_score, Test_score, Class_participation, Financial_support<br>
      <strong>Score ranges:</strong> Attendance 1–10 · Assignment 1–10 · Test 1–20 · Class Participation 1–10 · GPA 0–4
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload student CSV", type=["csv"])
    if uploaded:
        df_raw = pd.read_csv(uploaded)
        st.markdown(f"**{len(df_raw)} records loaded**")
        st.dataframe(df_raw.head(5), use_container_width=True)

        if st.button("🚀 Run Batch Predictions"):
            with st.spinner("Running predictions..."):
                result, err = batch_predict(bundle, df_raw)
            if err:
                st.error(f"Error: {err}")
            else:
                st.success(f"✅ Predictions complete for {len(result)} students!")
                st.session_state['batch_result'] = result

    if 'batch_result' in st.session_state:
        result = st.session_state['batch_result']
        high   = (result['Risk_Level']=='High').sum()
        medium = (result['Risk_Level']=='Medium').sum()
        low    = (result['Risk_Level']=='Low').sum()

        c1,c2,c3 = st.columns(3)
        with c1: stat_card("🔴", str(high),   "High Risk")
        with c2: stat_card("🟡", str(medium), "Medium Risk")
        with c3: stat_card("🟢", str(low),    "Low Risk")

        st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)

        def colour_risk(v):
            m = {"High":"background:#FEE2E2;color:#991B1B",
                 "Medium":"background:#FEF3C7;color:#92400E",
                 "Low":"background:#D1FAE5;color:#065F46"}
            return m.get(v,"")

        try:
            styled = result.style.map(colour_risk, subset=['Risk_Level'])
        except AttributeError:
            styled = result.style.applymap(colour_risk, subset=['Risk_Level'])

        st.dataframe(styled, use_container_width=True)
        st.download_button("⬇️ Download Results CSV",
                           result.to_csv(index=False).encode(),
                           "dropout_predictions.csv","text/csv")


# ═══════════════════════════════════════════════════════════════════════════════
#  ANALYTICS
# ═══════════════════════════════════════════════════════════════════════════════
def analytics_page(bundle):
    st.markdown('<div class="section-title">📈 Analytics & Insights</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Visual patterns from the Project Dataset</div>', unsafe_allow_html=True)

    DATASET_PATH = os.path.join(os.path.dirname(__file__), "Project_Dataset.csv")
    df = pd.read_csv(DATASET_PATH)
    palette = {0:GREEN, 1:RED}

    fig, axes = plt.subplots(2,3, figsize=(16,9))
    fig.patch.set_facecolor(BG)
    for ax in axes.flat:
        ax.set_facecolor(CARD)
        for s in ax.spines.values(): s.set_edgecolor('#E5E7EB')

    # 1. Dropout pie
    counts = df['Dropout'].value_counts()
    axes[0,0].pie(counts.values, labels=['Active','Dropout'],
                  colors=[GREEN,RED], autopct='%1.1f%%', startangle=90,
                  textprops={'fontsize':10})
    axes[0,0].set_title('Dropout Distribution', fontsize=12, fontweight='bold', color=TEXT)

    # 2. GPA distribution
    for val,color in palette.items():
        subset = df[df['Dropout']==val]['GPA']
        axes[0,1].hist(subset, bins=20, color=color, alpha=0.7, label='Dropout' if val else 'Active')
    axes[0,1].set_title('GPA Distribution', fontsize=12, fontweight='bold', color=TEXT)
    axes[0,1].set_xlabel('GPA'); axes[0,1].legend(fontsize=9)

    # 3. Attendance score
    for val,color in palette.items():
        subset = df[df['Dropout']==val]['Attendance_score']
        axes[0,2].hist(subset, bins=10, color=color, alpha=0.7, label='Dropout' if val else 'Active')
    axes[0,2].set_title('Attendance Score (1–10)', fontsize=12, fontweight='bold', color=TEXT)
    axes[0,2].set_xlabel('Score'); axes[0,2].legend(fontsize=9)

    # 4. Test score by status
    active_test  = df[df['Dropout']==0]['Test_score']
    dropout_test = df[df['Dropout']==1]['Test_score']
    axes[1,0].boxplot([active_test, dropout_test], labels=['Active','Dropout'],
                      patch_artist=True,
                      boxprops=dict(facecolor=GREEN, alpha=0.6),
                      medianprops=dict(color=TEXT, linewidth=2))
    boxes = axes[1,0].patches
    if len(boxes) >= 2:
        boxes[1].set_facecolor(RED)
        boxes[1].set_alpha(0.6)
    axes[1,0].set_title('Test Score by Status (1–20)', fontsize=12, fontweight='bold', color=TEXT)

    # 5. Financial support vs dropout rate
    by_fin = df.groupby('Financial_support')['Dropout'].mean() * 100
    colors_fin = [RED if f=='Low' else AMBER if f=='Medium' else GREEN for f in by_fin.index]
    bars = axes[1,1].bar(by_fin.index, by_fin.values, color=colors_fin)
    axes[1,1].set_title('Dropout Rate by Financial Support', fontsize=12, fontweight='bold', color=TEXT)
    axes[1,1].set_ylabel('%')
    for bar, val in zip(bars, by_fin.values):
        axes[1,1].text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                       f'{val:.1f}%', ha='center', fontsize=9)

    # 6. Class participation vs assignment score scatter
    colors_scatter = [GREEN if d==0 else RED for d in df['Dropout']]
    axes[1,2].scatter(df['Class_participation'], df['Assignment_score'],
                      c=colors_scatter, alpha=0.4, s=20)
    axes[1,2].set_title('Class Participation vs Assignment', fontsize=12, fontweight='bold', color=TEXT)
    axes[1,2].set_xlabel('Class Participation (1–10)')
    axes[1,2].set_ylabel('Assignment Score (1–10)')
    from matplotlib.patches import Patch
    axes[1,2].legend(handles=[Patch(color=GREEN,label='Active'),Patch(color=RED,label='Dropout')], fontsize=9)

    plt.tight_layout(pad=2)
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # Model comparison
    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🤖 Model Performance Comparison</div>', unsafe_allow_html=True)
    rows = [{"Model":k,"Accuracy":f"{v['acc']:.4f}","AUC-ROC":f"{v['auc']:.4f}"}
            for k,v in bundle.get('all_results',{}).items()]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.info(f"✅ Best model in use: **{bundle['model_name']}**")

    # Top at-risk students from real data
    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="section-title">🚨 Top 10 Highest Risk Students</div>', unsafe_allow_html=True)
    result, _ = batch_predict(bundle, df)
    if result is not None:
        top10 = result.sort_values('Dropout_Probability', ascending=False).head(10)
        display_cols = ['Matric_no','Student_name','GPA','Attendance_score',
                        'Test_score','Assignment_score','Dropout_Probability','Risk_Level']
        available = [c for c in display_cols if c in top10.columns]
        st.dataframe(top10[available].reset_index(drop=True), use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  HELP
# ═══════════════════════════════════════════════════════════════════════════════
def help_page():
    st.markdown('<div class="section-title">❓ Help & Documentation</div>', unsafe_allow_html=True)
    st.markdown("""
    ### How to use this system

    **Dashboard** — Overview of real statistics from your Project Dataset, plus a quick predict tool and live student risk table.

    **Predict** — Enter one student's matric number and details to get an instant dropout risk prediction with recommended actions.

    **Upload & Batch** — Upload a CSV to predict all students at once. Download the Project Dataset as a sample to see the correct format.

    **Analytics** — Charts and visual insights from the dataset including GPA, attendance, test scores, and financial support patterns.

    ---
    ### Risk Levels
    | Level | Probability | Suggested Action |
    |-------|-------------|-----------------|
    | 🟢 Low | < 35% | Regular monitoring |
    | 🟡 Medium | 35–65% | Schedule counselling |
    | 🔴 High | > 65% | Immediate intervention |

    ---
    ### Score Ranges (Project Dataset)
    | Column | Range |
    |--------|-------|
    | GPA | 0.0 – 4.0 |
    | Attendance_score | 1 – 10 |
    | Assignment_score | 1 – 10 |
    | Test_score | 1 – 20 |
    | Class_participation | 1 – 10 |
    | Financial_support | Low / Medium / High |
    | Gender | Male / Female |
    | Level | 200 |

    ---
    ### Login Credentials
    | Username | Password |
    |----------|----------|
    | admin | admin123 |
    | staff | staff2024 |
    | demo | demo |
    """)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
    if 'page' not in st.session_state:
        st.session_state['page'] = 'Dashboard'

    if not st.session_state['logged_in']:
        login_page()
        return

    bundle = load_bundle()
    page   = render_sidebar()

    if   page == 'Dashboard':      dashboard_page(bundle)
    elif page == 'Predict':        predict_page(bundle)
    elif page == 'Upload & Batch': upload_page(bundle)
    elif page == 'Analytics':      analytics_page(bundle)
    elif page == 'Help':           help_page()

if __name__ == '__main__':
    main()


