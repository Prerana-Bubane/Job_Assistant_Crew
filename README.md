# 💼 Multi-Agent Job Application Suite

An autonomous multi-agent system built with **CrewAI**, **Google Gemini**, and **Streamlit** that automates the end-to-end job application preparation lifecycle.

**LIVE Demo** :( https://jobassistantcrew-hr26msjanhiyfddmjcxbvq.streamlit.app/)

## 🏗️ Architecture & Agent Pipeline

The system uses a sequential multi-agent collaborative pipeline:

1. **Senior Technical Recruiter (Extractor)**: Deconstructs the job description into must-have skills, nice-to-haves, and ATS sourcing keywords.
2. **Career Coach & Profile Analyst (Matcher)**: Executes an objective gap analysis between candidate background and job requirements.
3. **ATS Resume Optimization Strategist (Writer)**: Rewrites and reframes resume bullet points using action-oriented frameworks without fabricating experience.
4. **Executive Career Storyteller (Cover Letter Writer)**: Drafts an authentic narrative framing transferable skills and mitigating gaps.
5. **Senior Hiring Committee Chair (Reviewer & Auditor)**: Audits the complete package, calculates a 0-100 fit score, checks for keyword compliance, and generates predictive technical interview questions.

## 🛠️ Tech Stack
- **Agent Orchestration**: CrewAI
- **LLM**: Google Gemini (`gemini-3.8-flash`)
- **Frontend / Deployment**: Streamlit & Streamlit Community Cloud
- **Document Ingestion**: pdfplumber

## 🚀 Key Engineering Highlights
- **Role Grounding & Personas**: Distinct backstories and objectives to avoid model persona drift.
- **Context Passing**: Downstream agents consume synthesized artifacts rather than raw prompts.
- **Adversarial Quality Gate**: Built-in Critic pattern to flag resume puffery and bracket placeholders.
