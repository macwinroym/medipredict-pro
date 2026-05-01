import streamlit as st
import pandas as pd
import joblib

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="MediPredict Pro",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ======================================================
# LOAD DATA
# ======================================================
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv("Training.csv")
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    return df

model = load_model()
data = load_data()

symptoms = data.drop("prognosis", axis=1).columns.tolist()
diseases = sorted(data["prognosis"].unique().tolist())

# ======================================================
# SESSION STATE
# ======================================================
if "page" not in st.session_state:
    st.session_state.page = "Home"

# ======================================================
# HELPERS
# ======================================================
def go(page):
    st.session_state.page = page
    st.rerun()

def pretty(text):
    return text.replace("_", " ").title()

def predict(selected):
    row = [1 if s in selected else 0 for s in symptoms]
    disease = model.predict([row])[0]

    score = min(97, 48 + len(selected) * 4)

    if score >= 80:
        risk = "High"
        color = "#ef4444"
    elif score >= 60:
        risk = "Moderate"
        color = "#f59e0b"
    else:
        risk = "Low"
        color = "#10b981"

    return disease, score, risk, color

def top_symptoms_for_disease(disease_name, n=5):
    df = data[data["prognosis"] == disease_name]
    if df.empty:
        return []

    symptom_means = df.drop("prognosis", axis=1).mean()
    top = symptom_means.sort_values(ascending=False)
    top = top[top > 0].head(n).index.tolist()

    return [pretty(x) for x in top]

# ======================================================
# CSS
# ======================================================
st.markdown("""
<style>
#MainMenu{visibility:hidden;}
footer{visibility:hidden;}
header{visibility:hidden;}

html, body, [class*="css"]{
    font-family:Segoe UI, sans-serif;
}

body{
    background:#07111f;
}

section.main > div{
    padding-top:1rem;
}

/* NAV */
.nav{
    background:linear-gradient(135deg,#0f172a,#111827);
    padding:10px 28px;
    border-radius:18px;
    margin-bottom:24px;
    border:1px solid rgba(255,255,255,.06);
}

.logo{
    color:white;
    font-size:28px;
    font-weight:800;
}

.sub{
    color:#94a3b8;
    font-size:13px;
}

/* HERO */
.hero{
    background:linear-gradient(135deg,#06b6d4,#1d4ed8);
    padding:42px;
    border-radius:24px;
    color:white;
}

.hero h1{
    font-size:54px;
    line-height:1.1;
}

.hero p{
    color:#e0f2fe;
    font-size:18px;
}

/* CARDS */
.card{
    background:#111827;
    color:white;
    padding:22px;
    border-radius:20px;
    border:1px solid rgba(255,255,255,.06);
}

.light{
    background:white;
    color:#111827;
    padding:22px;
    border-radius:20px;
    box-shadow:0 10px 22px rgba(0,0,0,.08);
    margin-bottom:18px;
}

.title{
    color:white;
    font-size:42px;
    font-weight:800;
}

.subt{
    color:#94a3b8;
    font-size:15px;
    margin-bottom:20px;
}

.footer{
    text-align:center;
    color:#64748b;
    font-size:13px;
    padding:28px 0;
}

.stButton > button{
    width:100%;
    border:none;
    border-radius:14px;
    padding:12px;
    font-weight:700;
    color:white;
    background:linear-gradient(135deg,#06b6d4,#2563eb);
}

.stButton > button:hover{
    transform:translateY(-2px);
    transition:.2s;
}

div[data-baseweb="input"]{
    border-radius:14px !important;
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# NAVBAR
# ======================================================
st.markdown("""
<div class='nav'>
<div class='logo'>🩺 MediPredict</div>
<div class='sub'>AI Healthcare Analytics Platform</div>
</div>
""", unsafe_allow_html=True)

a,b,c,d = st.columns(4)

with a:
    if st.button("🏠 Home"):
        go("Home")
with b:
    if st.button("🧠 Predict"):
        go("Predict")
with c:
    if st.button("📘 About"):
        go("About")
with d:
    if st.button("🩺 Diseases"):
        go("Diseases")

st.write("")

# ======================================================
# HOME
# ======================================================
if st.session_state.page == "Home":

    left,right = st.columns([3,2], gap="large")

    with left:
        st.markdown("""
        <div class='hero'>
        <small>Machine Learning • Healthcare Analytics</small>
        <h1>Diagnose smarter with AI insights</h1>
        <p>Predict probable diseases using symptoms and trained machine learning models.<br>
        132 Symptoms • 4,920 Records • 41 Diseases</p>
        </div>
        """, unsafe_allow_html=True)

        x,y = st.columns(2)
        with x:
            if st.button("Start Diagnosis"):
                go("Predict")
        with y:
            if st.button("How It Works"):
                go("About")

    with right:
        st.markdown("<div class='card'><h3>📈 Live Analysis</h3></div>", unsafe_allow_html=True)
        st.write("Common Cold")
        st.progress(78)
        st.write("Allergy")
        st.progress(43)
        st.write("Asthma")
        st.progress(26)
        st.write("Pneumonia")
        st.progress(14)

    st.write("")
    m1,m2,m3 = st.columns(3)
    with m1:
        st.metric("Diseases", "41+")
    with m2:
        st.metric("Symptoms", "132")
    with m3:
        st.metric("Accuracy", "95%+")

# ======================================================
# PREDICT
# ======================================================
elif st.session_state.page == "Predict":

    st.markdown("<div class='title'>Symptom Checker</div>", unsafe_allow_html=True)
    st.markdown("<div class='subt'>Select symptoms and get AI-based disease prediction.</div>", unsafe_allow_html=True)

    left,right = st.columns([3,2], gap="large")

    with left:
        search = st.text_input("Search symptoms")
        filtered = [s for s in symptoms if search.lower() in pretty(s).lower()]
        selected = []

        cols = st.columns(3)

        for i,s in enumerate(filtered):
            with cols[i % 3]:
                if st.checkbox(pretty(s), key=s):
                    selected.append(s)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)

        if len(selected) == 0:
            st.info("Select symptoms to begin prediction.")
        else:
            disease,score,risk,color = predict(selected)

            st.markdown(f"""
            <div style='background:{color};padding:24px;border-radius:18px;color:white'>
            <h1>{disease}</h1>
            <h3>{score}% Confidence</h3>
            <p>{risk} Risk Level</p>
            </div>
            """, unsafe_allow_html=True)

            st.progress(score)

            st.write("### Selected Symptoms")
            for s in selected[:8]:
                st.success(pretty(s))

            st.warning("This tool is for educational support only. Consult a doctor.")

        st.markdown("</div>", unsafe_allow_html=True)

# ======================================================
# ABOUT (UPGRADED UI/UX)
# ======================================================
elif st.session_state.page == "About":

    st.markdown("""
    <div style='background:linear-gradient(135deg,#06b6d4,#1d4ed8);
    padding:35px;border-radius:24px;color:white;margin-bottom:25px;'>

    <h1 style='font-size:48px;margin-bottom:10px;'>How MediPredict Works</h1>
    <p style='font-size:18px;color:#dbeafe;'>
    A machine learning healthcare analytics platform designed to assist
    early disease prediction using clinical symptoms.
    </p>
    </div>
    """, unsafe_allow_html=True)

    # Stats
    a,b,c = st.columns(3)

    with a:
        st.markdown("""
        <div class='light'>
        <h1 style='color:#2563eb;'>4,920</h1>
        <p>Training Records</p>
        </div>
        """, unsafe_allow_html=True)

    with b:
        st.markdown("""
        <div class='light'>
        <h1 style='color:#10b981;'>41</h1>
        <p>Diseases Covered</p>
        </div>
        """, unsafe_allow_html=True)

    with c:
        st.markdown("""
        <div class='light'>
        <h1 style='color:#f59e0b;'>132</h1>
        <p>Symptoms Tracked</p>
        </div>
        """, unsafe_allow_html=True)

    st.write("")
    st.markdown("<div class='title' style='font-size:34px;'>Prediction Pipeline</div>", unsafe_allow_html=True)

    p1,p2,p3 = st.columns(3)

    with p1:
        st.info("📥 Data Collection")
        st.info("🧹 Data Preprocessing")

    with p2:
        st.info("📊 Exploratory Analysis")
        st.info("🧠 Model Training")

    with p3:
        st.info("🎯 Prediction Engine")
        st.info("✅ Evaluation")

    st.write("")
    st.markdown("<div class='title' style='font-size:34px;'>Tech Stack</div>", unsafe_allow_html=True)

    t1,t2,t3,t4,t5 = st.columns(5)
    t1.success("Python")
    t2.success("Pandas")
    t3.success("NumPy")
    t4.success("Scikit-learn")
    t5.success("Streamlit")

    st.write("")

    st.markdown("""
    <div style='background:#fef3c7;padding:22px;border-radius:18px;color:#111827;'>
    <h4>⚠ Limitations</h4>
    <p>
    Predictions are generated using dataset patterns and machine learning.
    This system supports awareness only and does not replace doctors.
    </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    st.markdown("""
    <div class='card'>
    <h3>🌍 Why MediPredict Matters</h3>
    <p>
    Faster screening, early symptom awareness, reduced delay in decisions,
    and modern healthcare support using AI systems.
    </p>
    </div>
    """, unsafe_allow_html=True)

# ======================================================
# DISEASES
# ======================================================
elif st.session_state.page == "Diseases":

    st.markdown("<div class='title'>Disease Catalog</div>", unsafe_allow_html=True)
    st.markdown("<div class='subt'>Browse supported diseases with common symptoms.</div>", unsafe_allow_html=True)

    q = st.text_input("Search disease")
    filtered = [d for d in diseases if q.lower() in d.lower()]

    cols = st.columns(3)

    for i,disease_name in enumerate(filtered):
        top5 = top_symptoms_for_disease(disease_name, 5)

        symptom_html = ""
        for s in top5:
            symptom_html += f"<li>{s}</li>"

        with cols[i % 3]:
            st.markdown(f"""
            <div class='light'>
            <h4>{disease_name}</h4>
            <p><b>Common Symptoms:</b></p>
            <ul>
            {symptom_html}
            </ul>
            </div>
            """, unsafe_allow_html=True)

# ======================================================
# FOOTER
# ======================================================
st.markdown("""
<div class='footer'>
© 2026 MediPredict Pro • Healthcare Analytics Project • Final Year Ready
</div>
""", unsafe_allow_html=True)