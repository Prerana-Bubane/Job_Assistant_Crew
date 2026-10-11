import os
import streamlit as st
import pdfplumber
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process

# 1. Environment & Page Setup
load_dotenv()
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

st.set_page_config(
    page_title="Agentic Job Application Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for modern aesthetic
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(124, 58, 237, 0.4);
    }
    .metric-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 16px;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            extract = page.extract_text()
            if extract:
                text += extract + "\n"
    return text

# --- Sidebar Configuration ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=64)
    st.title("Crew Configuration")
    st.caption("Powered by CrewAI & Google Gemini")
    
    # Securely retrieve the key from environment or Streamlit secrets
    env_api_key = os.getenv("GEMINI_API_KEY", "")
    
    # If a key exists on the server/.env, keep it hidden and mark as connected
    if env_api_key:
        st.success("🔒 System API Key Connected", icon="✅")
        api_key_input = env_api_key
        st.caption("Using backend environment credentials securely.")
    else:
        # Only prompt for a key if none exists on the system
        api_key_input = st.text_input(
            "Gemini API Key", 
            type="password",
            placeholder="AIzaSy...",
            help="Enter your Gemini API key to run the crew."
        )
    
    st.divider()
    st.markdown("### 🤖 Autonomous Agents")
    st.markdown("""
    1. **Extractor**: Job Specs & Keywords
    2. **Matcher**: In-depth Gap Analysis
    3. **Resume Strategist**: ATS Bullet Rewriting
    4. **Cover Letter Writer**: Contextual Pitch
    5. **Reviewer**: Fit Scoring & Quality Gate
    """)
    st.divider()
    st.caption("v1.0 | Multi-Agent Portfolio Project")

# --- Main Page Header ---
st.title("💼 Multi-Agent Job Application Suite")
st.markdown("Orchestrate a specialized team of 5 AI agents to analyze, align, optimize, and audit your job application materials.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Target Job Description")
    job_desc_input = st.text_area(
        "Paste the job posting requirements here:",
        height=260,
        placeholder="Paste role description, required qualifications, tech stack..."
    )

with col2:
    st.subheader("2. Your Resume")
    resume_tab1, resume_tab2 = st.tabs(["📄 Upload PDF", "✍️ Paste Text"])
    
    resume_input = ""
    with resume_tab1:
        uploaded_pdf = st.file_uploader("Upload your current resume (PDF)", type=["pdf"])
        if uploaded_pdf:
            resume_input = extract_text_from_pdf(uploaded_pdf)
            st.success(f"Extracted {len(resume_input.split())} words from PDF.")
            
    with resume_tab2:
        text_resume = st.text_area(
            "Or paste your resume text directly:",
            height=200,
            placeholder="Experience, projects, technical skills..."
        )
        if not resume_input and text_resume:
            resume_input = text_resume

st.markdown("<br>", unsafe_allow_html=True)
run_button = st.button("🚀 Deploy Multi-Agent Crew", use_container_width=True)

# --- Execution & Agent Orchestration ---
if run_button:
    api_key = api_key_input or os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("Please provide a valid Gemini API Key in the sidebar or .env file.")
    elif not job_desc_input.strip() or not resume_input.strip():
        st.warning("Please provide both a Job Description and a Resume to proceed.")
    else:
        os.environ["GEMINI_API_KEY"] = api_key

        with st.status("🤖 Crew in Progress: Collaborating...", expanded=True) as status:
            st.write("Initializing Agent team personas...")

            # 1. Define Agents
            extractor_agent = Agent(
                role="Senior Technical Recruiter",
                goal="Extract key requirements, technical skills, and experience levels from a job description.",
                backstory="Expert recruiter identifying core requirements vs nice-to-haves.",
                verbose=True,
                allow_delegation=False,
             llm="gemini/gemini-3.8-flash"
            )

            matcher_agent = Agent(
                role="Career Coach and Profile Analyst",
                goal="Compare a candidate's resume against job requirements and provide a detailed gap analysis.",
                backstory="Objective career coach providing gap analyses across strong matches, weak matches, and missing skills.",
                verbose=True,
                allow_delegation=False,
              llm="gemini/gemini-3.8-flash"
            )

            resume_writer_agent = Agent(
                role="ATS Resume Optimization Strategist",
                goal="Rewrite and optimize the candidate's resume to highlight relevant strengths and ATS keywords without fabricating experience.",
                backstory="Elite resume strategist transforming bullets into high-impact, ATS-optimized points with transferable skills.",
                verbose=True,
                allow_delegation=False,
            llm="gemini/gemini-3.8-flash"
            )

            cover_letter_agent = Agent(
                role="Executive Career Storyteller",
                goal="Draft an authentic, compelling cover letter connecting candidate strengths to the job description while framing learning curves positively.",
                backstory="Master storyteller crafting tailored, human, and persuasive cover letters.",
                verbose=True,
                allow_delegation=False,
               llm="gemini/gemini-3.8-flash"
            )

            reviewer_agent = Agent(
                role="Senior Hiring Committee Chair & Quality Auditor",
                goal="Audit the final materials against the job description and output a fit score, keyword audit, and interview questions.",
                backstory="High-standards engineering director spotting buzzwords, missing keywords, and weak spots.",
                verbose=True,
                allow_delegation=False,
              llm="gemini/gemini-3.8-flash"
            )

            # 2. Define Tasks
            extract_task = Task(
                description=f"Analyze the following job description and extract core requirements, tech stack, and ATS keywords:\n\n{job_desc_input}",
                expected_output="Structured breakdown of requirements, nice-to-haves, and keywords.",
                agent=extractor_agent
            )

            match_task = Task(
                description=f"Produce a gap analysis comparing the extracted requirements to this resume:\n\n{resume_input}",
                expected_output="Detailed gap analysis categorized into strong matches, weak matches, and missing skills.",
                agent=matcher_agent,
                context=[extract_task]
            )

            resume_task = Task(
                description=f"""Using the gap analysis and original resume below, rewrite and polish the candidate's resume:
1. Reframe bullet points for impact and transferable skills.
2. Weave in critical ATS keywords.
3. DO NOT fabricate false experience.

Original Resume:
{resume_input}""",
                expected_output="An optimized, ATS-friendly markdown resume ready for submission.",
                agent=resume_writer_agent,
                context=[match_task]
            )

            cover_letter_task = Task(
                description=f"""Draft a tailored, authentic 3-4 paragraph cover letter for this role:
1. Reference specific requirements from the job description.
2. Highlight genuine strengths.
3. Positively address learning curves and transferable skills.

Job Description:
{job_desc_input}

Original Resume:
{resume_input}""",
                expected_output="Polished 3-4 paragraph markdown cover letter.",
                agent=cover_letter_agent,
                context=[extract_task, match_task]
            )

            review_task = Task(
                description=f"""Audit and score the complete application package (rewritten resume and cover letter) against the job description.
Include:
1. Fit Score (0-100) with rationale.
2. ATS & Keyword Audit table.
3. Honesty & Hallucination Check.
4. Top 3 tough technical interview questions for this candidate.

Job Description:
{job_desc_input}""",
                expected_output="Comprehensive quality review report with fit score, keyword audit, and interview questions.",
                agent=reviewer_agent,
                context=[extract_task, match_task, resume_task, cover_letter_task]
            )

            # 3. Assemble and Kickoff Crew
            crew = Crew(
                agents=[extractor_agent, matcher_agent, resume_writer_agent, cover_letter_agent, reviewer_agent],
                tasks=[extract_task, match_task, resume_task, cover_letter_task, review_task],
                process=Process.sequential,
                verbose=True
            )

            st.write("Agents are executing sequentially: Extractor ➔ Matcher ➔ Resume Writer ➔ Cover Letter ➔ Reviewer...")
            crew.kickoff()
            status.update(label="✅ All Agents Finished Successfully!", state="complete", expanded=False)

        # --- Interactive Tab Results Display ---
        st.success("🎉 Multi-Agent Pipeline Completed!")
        
        tab_review, tab_resume, tab_cover, tab_gaps, tab_reqs = st.tabs([
            "📊 Final Audit & Score",
            "📄 Tailored Resume",
            "✉️ Cover Letter",
            "🔍 Gap Analysis",
            "📋 Extracted Requirements"
        ])

        with tab_review:
            st.markdown("### Hiring Committee Review & Audit")
            st.markdown(review_task.output.raw)
            st.download_button(
                "📥 Download Audit Report (.md)",
                data=review_task.output.raw,
                file_name="application_audit_report.md",
                mime="text/markdown"
            )

        with tab_resume:
            st.markdown("### Optimized ATS-Friendly Resume")
            st.markdown(resume_task.output.raw)
            st.download_button(
                "📥 Download Resume (.md)",
                data=resume_task.output.raw,
                file_name="optimized_resume.md",
                mime="text/markdown"
            )

        with tab_cover:
            st.markdown("### Tailored Cover Letter")
            st.markdown(cover_letter_task.output.raw)
            st.download_button(
                "📥 Download Cover Letter (.md)",
                data=cover_letter_task.output.raw,
                file_name="tailored_cover_letter.md",
                mime="text/markdown"
            )

        with tab_gaps:
            st.markdown("### Profile Gap Analysis")
            st.markdown(match_task.output.raw)

        with tab_reqs:
            st.markdown("### Extracted Requirements & Keywords")
            st.markdown(extract_task.output.raw)