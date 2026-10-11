import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process

# 1. Setup Environment
load_dotenv()
os.environ["CREWAI_TELEMETRY_OPT_OUT"] = "true"

# --- 2. Define the 5 Agents ---

extractor_agent = Agent(
    role="Senior Technical Recruiter",
    goal="Extract key requirements, technical skills, and experience levels from a job description.",
    backstory="You are an expert recruiter who knows exactly what hiring managers look for. You have a sharp eye for identifying core requirements versus nice-to-haves.",
    verbose=True,
    allow_delegation=False,
    llm="gemini/gemini-3.8-flash"
)

matcher_agent = Agent(
    role="Career Coach and Profile Analyst",
    goal="Compare a candidate's resume against job requirements and provide a detailed gap analysis.",
    backstory="You are a seasoned career coach who helps candidates understand how their skills align with job descriptions. You are honest, analytical, and provide actionable feedback on strong matches and missing skills.",
    verbose=True,
    allow_delegation=False,
    llm="gemini/gemini-3.8-flash"
)

resume_writer_agent = Agent(
    role="ATS Resume Optimization Strategist",
    goal="Rewrite and optimize the candidate's resume to highlight relevant strengths and ATS keywords without fabricating experience.",
    backstory="You are an elite technical resume writer. You transform project descriptions into high-impact, results-driven bullet points using action-oriented frameworks. You emphasize transferable skills to mitigate identified gaps while maintaining strict honesty.",
    verbose=True,
    allow_delegation=False,
    llm="gemini/gemini-3.8-flash"
)

cover_letter_agent = Agent(
    role="Executive Career Storyteller",
    goal="Draft an authentic, compelling cover letter connecting candidate strengths to the job description while framing learning curves positively.",
    backstory="You are a master of professional storytelling. You write concise, persuasive cover letters that avoid generic corporate clichés, demonstrate enthusiasm for the role, and showcase a growth mindset.",
    verbose=True,
    allow_delegation=False,
    llm="gemini/gemini-3.8-flash"
)

reviewer_agent = Agent(
    role="Senior Hiring Committee Chair & Quality Auditor",
    goal="Ruthlessly evaluate the final resume and cover letter against the original job description and provide a fit score with constructive critiques.",
    backstory="You are a discerning engineering hiring manager known for high standards. You spot buzzword stuffing, generic fluff, and fabricated claims instantly. You provide objective scores and highlight exactly what questions this candidate will face in a live technical interview.",
    verbose=True,
    allow_delegation=False,
    llm="gemini/gemini-3.8-flash"
)

# --- 3. Inputs ---

sample_job_desc = """
Looking for a Junior Python Developer. Must have 1-2 years of experience with Python, FastAPI, and Docker. Experience with SQL databases is a plus. Needs strong communication skills.
"""

sample_resume = """
Recent bootcamp graduate. Built multiple projects using Python and Flask. Familiar with SQLite and PostgreSQL. Excellent written communication and teamwork skills. No professional Docker experience yet.
"""

# --- 4. Define the Tasks ---

extract_task = Task(
    description=f"Analyze the following job description and extract the core requirements, required technologies, and ATS keywords.\n\nJob Description:\n{sample_job_desc}",
    expected_output="A structured breakdown of core requirements, nice-to-haves, and key ATS keywords.",
    agent=extractor_agent
)

match_task = Task(
    description=f"Produce a gap analysis highlighting strong matches, weak matches, and missing skills. Compare the extracted job requirements with this candidate resume:\n\n{sample_resume}",
    expected_output="A detailed gap analysis report categorized by strong matches, weak matches, and missing skills.",
    agent=matcher_agent,
    context=[extract_task]
)

resume_task = Task(
    description=f"""Using the gap analysis report from the Matcher and the original resume below, rewrite and polish the candidate's resume:
1. Reframe project bullet points to emphasize relevant backend architecture and transferable skills.
2. Naturally weave in relevant ATS keywords.
3. DO NOT fabricate false experience.

Original Resume:
{sample_resume}""",
    expected_output="An optimized, ATS-friendly markdown resume ready to submit.",
    agent=resume_writer_agent,
    context=[match_task]
)

cover_letter_task = Task(
    description=f"""Draft a tailored, authentic 3-4 paragraph cover letter for this role:
1. Reference specific requirements from the job description.
2. Highlight the candidate's real strengths (Python, databases, collaborative projects).
3. Positively address the transition to FastAPI and eagerness to adopt containerization tools like Docker.

Job Description:
{sample_job_desc}

Original Resume:
{sample_resume}""",
    expected_output="A polished 3-4 paragraph markdown cover letter tailored to this position.",
    agent=cover_letter_agent,
    context=[extract_task, match_task]
)

review_task = Task(
    description=f"""Audit and score the complete job application package (both the rewritten resume and cover letter) against the original job description.

Your review report must include:
1. **Fit Score (0-100)** with a clear rationale.
2. **ATS & Keyword Audit**: Are all non-negotiable keywords addressed effectively and honestly?
3. **Honesty / Hallucination Check**: Verify that no unverified claims or fake employment were introduced.
4. **Interview Readiness**: Top 3 tough technical questions the candidate should anticipate based on their profile gaps.

Original Job Description:
{sample_job_desc}""",
    expected_output="A structured final review report featuring a Fit Score, ATS audit, honesty validation, and interview prep questions.",
    agent=reviewer_agent,
    context=[extract_task, match_task, resume_task, cover_letter_task]  # The reviewer inspects EVERYTHING!
)

# --- 5. Assemble the Crew ---

job_crew = Crew(
    agents=[extractor_agent, matcher_agent, resume_writer_agent, cover_letter_agent, reviewer_agent],
    tasks=[extract_task, match_task, resume_task, cover_letter_task, review_task],
    process=Process.sequential,
    verbose=True
)

# --- 6. Kickoff! ---

if __name__ == "__main__":
    print("Starting full 5-agent multi-agent review crew...\n")
    result = job_crew.kickoff()
    
    print("\n" + "="*50)
    print("FINAL QUALITY AUDIT & FIT SCORE (Reviewer Agent):")
    print("="*50 + "\n")
    print(result)