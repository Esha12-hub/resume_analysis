# app.py
import streamlit as st
from sentence_transformers import SentenceTransformer, util
import PyPDF2

# ================== Hugging Face Model ==================
MODEL_REPO = "maham234/resume-matcher-model"
model = SentenceTransformer(MODEL_REPO)

# ================== Streamlit Config ==================
st.set_page_config(page_title="AI Resume Matcher", page_icon="🤖", layout="wide")

# ================== Custom CSS ==================
st.markdown(
    """
    <style>
    /* Background gradient */
    .stApp {
        background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
        color: #ffffff;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Title Glow */
    h1, h2, h3 {
        color: #00e6e6;
        text-shadow: 0 0 10px #00e6e6, 0 0 20px #00ffff;
    }

    /* Info/Success/Warning/Error Boxes */
    .stSuccess, .stInfo, .stWarning, .stError {
        border-radius: 12px;
        padding: 10px;
        font-weight: bold;
    }

    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(to right, #00e6e6, #007acc);
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: #1c1c1c;
        color: #00e6e6;
        font-weight: bold;
    }

    /* Container styling */
    .stMarkdown {
        background: rgba(255,255,255,0.05);
        padding: 12px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ================== Title ==================
st.title("🤖 Resume Matcher")
st.caption("✨ Advanced AI-powered resume analysis and job matching system")

# ================== Helper Functions ==================
def extract_text_from_pdf(uploaded_file):
    """Extract text from PDF resumes."""
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text.strip()

# Example job descriptions (replace with DB or CSV if needed)
job_descriptions = {
    "Software Engineer": "Develop and maintain software applications using Python, JavaScript, APIs, and cloud platforms.",
    "Data Scientist": "Analyze large datasets, build ML models, and deploy NLP-based AI systems.",
    "DevOps Engineer": "Manage CI/CD pipelines, containerization (Docker, Kubernetes), and cloud deployments.",
    "AI Researcher": "Research deep learning models, computer vision, and reinforcement learning algorithms.",
    "Business Analyst": "Gather requirements, analyze business processes, and generate reports with SQL/Excel."
}

# ================== Upload Resume ==================
uploaded_file = st.file_uploader("📂 Upload Your Resume (PDF)", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("🔎 Extracting resume text..."):
        resume_text = extract_text_from_pdf(uploaded_file)
        st.success("✅ Resume text extracted!")

    with st.expander("📑 View Extracted Resume Text"):
        st.write(resume_text[:1200] + "...")  # preview

    # ================== Compute Match Scores ==================
    resume_embedding = model.encode(resume_text, convert_to_tensor=True)
    job_scores = {}

    for job, description in job_descriptions.items():
        job_embedding = model.encode(description, convert_to_tensor=True)
        similarity = util.cos_sim(resume_embedding, job_embedding).item() * 100
        job_scores[job] = similarity

    # Sort jobs by score
    sorted_jobs = dict(sorted(job_scores.items(), key=lambda x: x[1], reverse=True))

    # ================== Display Results ==================
    st.subheader("📊 AI Analysis Results")
    for idx, (job, score) in enumerate(sorted_jobs.items(), start=1):
        st.markdown(f"### {idx}. {job}")
        st.progress(int(score))  # progress bar
        st.markdown(f"**Compatibility Score:** `{score:.2f}%`")

        # Match interpretation
        if score > 70:
            st.success("✅ Strong Match")
        elif score > 50:
            st.warning("⚡ Partial Match")
        else:
            st.error("❌ Weak Match")

        st.markdown("---")
else:
    st.info("👆 Please upload a PDF resume to begin analysis.")
