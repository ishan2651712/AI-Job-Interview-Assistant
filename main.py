from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from crewai import Agent, Task, Crew, LLM
from dotenv import load_dotenv
import os
import traceback

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY is None:
    raise ValueError("GEMINI_API_KEY is not set. Please add it to your .env file.")


gemini_llm = LLM(
    model="gemini/gemini-2.5-pro",  
    api_key=GEMINI_API_KEY,
    temperature=0.4,
)



app = FastAPI(
    title="AI Job Interview Assistant",
    description="Agentic AI app using CrewAI + Gemini + FastAPI to generate interview prep material.",
    version="1.0.0",
)


app.mount("/static", StaticFiles(directory="static"), name="static")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




@app.get("/")
def root():
    
    return FileResponse("static/index.html")


@app.get("/health")
def health_check():
    return {"status": "ok"}




class JobRoleRequest(BaseModel):
    role: str
    experience_level: str | None = "Fresher"  
    tech_stack: str | None = None            




def build_career_agent():
    return Agent(
        role="Career Research Expert",
        goal="Explain job roles, skills, learning paths, and expectations.",
        backstory="You help students understand technical job profiles.",
        llm=gemini_llm,
        verbose=True,
    )


def build_technical_agent():
    return Agent(
        role="Technical Interview Expert",
        goal="Generate technical interview questions and detailed answers based on job role.",
        backstory="You have experience as a technical interviewer across IT companies.",
        llm=gemini_llm,
        verbose=True,
    )


def build_hr_agent():
    return Agent(
        role="HR Interview & Soft Skills Expert",
        goal="Generate HR/behavioral interview questions with model answers.",
        backstory="You have experience as HR evaluating communication, attitude and culture fit.",
        llm=gemini_llm,
        verbose=True,
    )


def build_tips_agent():
    return Agent(
        role="Interview Coach",
        goal="Give practical interview tips, mistakes to avoid, and tricks to impress interviewer.",
        backstory="You mentor freshers for college placements.",
        llm=gemini_llm,
        verbose=True,
    )




@app.post("/generate_overview")
def generate_overview(request: JobRoleRequest):
    role = request.role
    exp = request.experience_level or "Fresher"
    tech = request.tech_stack or "General"

    overview_text = ""

    try:
        career_agent = build_career_agent()

        overview_task = Task(
            description=(
                f"Write a structured plain-text interview overview for the role: {role}.\n"
                f"Experience level: {exp}, Tech stack: {tech}.\n"
                "Include: role summary, key responsibilities, technical skills, soft skills, "
                "and typical interview rounds (online test, technical rounds, HR, etc.).\n"
                "Important: No markdown (*, #, **). Plain clean text only."
            ),
            expected_output="Structured interview overview in plain text.",
            agent=career_agent,
        )

        crew = Crew(
            agents=[career_agent],
            tasks=[overview_task],
            verbose=True,
        )

        result = crew.kickoff()
        overview_text = str(result)
    except Exception as e:
        print("Error in overview generation:", e)
        traceback.print_exc()
        overview_text = f"Error generating overview: {e}"

    return {
        "role": role,
        "experience_level": exp,
        "tech_stack": tech,
        "overview": overview_text,
    }



@app.post("/generate_technical_qa")
def generate_technical_qa(request: JobRoleRequest):
    role = request.role
    exp = request.experience_level or "Fresher"
    tech = request.tech_stack or "General"

    technical_text = ""

    try:
        technical_agent = build_technical_agent()

        technical_task = Task(
            description=(
                f"Generate 10 technical interview questions and answers for the job role: {role}.\n"
                f"Target candidate level: {exp}. Relevant technologies: {tech}.\n"
                "Use ONLY plain text (no markdown).\n"
                "Format STRICTLY like this:\n"
                "Q1: <question text>\n"
                "A1: <answer text>\n"
                "Q2: <question text>\n"
                "A2: <answer text>\n"
                "...\n"
                "Questions should test core fundamentals, problem solving, and role-specific concepts."
            ),
            expected_output="10 technical Q&A pairs in the specified Q/A format.",
            agent=technical_agent,
        )

        crew = Crew(
            agents=[technical_agent],
            tasks=[technical_task],
            verbose=True,
        )

        result = crew.kickoff()
        technical_text = str(result)
    except Exception as e:
        print("Error in technical Q&A generation:", e)
        traceback.print_exc()
        technical_text = f"Error generating technical Q&A: {e}"

    return {
        "role": role,
        "experience_level": exp,
        "tech_stack": tech,
        "technical_qa": technical_text,
    }




@app.post("/generate_hr_qa")
def generate_hr_qa(request: JobRoleRequest):
    role = request.role
    exp = request.experience_level or "Fresher"
    tech = request.tech_stack or "General"

    hr_text = ""

    try:
        hr_agent = build_hr_agent()

        hr_task = Task(
            description=(
                "Generate 10 HR and behavioral interview questions with strong sample answers "
                f"for a candidate applying as {role}.\n"
                "Focus on communication, teamwork, conflicts, strengths/weaknesses, failure, pressure handling, etc.\n"
                "Plain text only. No markdown.\n"
                "Format STRICTLY like this:\n"
                "Q1: <question text>\n"
                "A1: <answer text>\n"
                "Q2: <question text>\n"
                "A2: <answer text>\n"
                "...\n"
                "Answers should sound like a good B.Tech student preparing for placements."
            ),
            expected_output="10 HR Q&A pairs in the specified Q/A format.",
            agent=hr_agent,
        )

        crew = Crew(
            agents=[hr_agent],
            tasks=[hr_task],
            verbose=True,
        )

        result = crew.kickoff()
        hr_text = str(result)
    except Exception as e:
        print("Error in HR Q&A generation:", e)
        traceback.print_exc()
        hr_text = f"Error generating HR Q&A: {e}"

    return {
        "role": role,
        "experience_level": exp,
        "tech_stack": tech,
        "hr_qa": hr_text,
    }




@app.post("/generate_interview_tips")
def generate_interview_tips(request: JobRoleRequest):
    role = request.role
    exp = request.experience_level or "Fresher"
    tech = request.tech_stack or "General"

    tips_text = ""

    try:
        tips_agent = build_tips_agent()

        tips_task = Task(
            description=(
                f"Give interview tips specifically for the role: {role}.\n"
                f"Candidate level: {exp}, Tech: {tech}.\n"
                "Include:\n"
                "- Do's before the interview (preparation, resume, projects).\n"
                "- Do's during the interview (body language, how to answer).\n"
                "- Don'ts (common mistakes students make).\n"
                "- Final motivation / confidence boost.\n"
                "Use plain text, no markdown, and keep it crisp and practical."
            ),
            expected_output="Practical do's, don'ts and success tips in plain text.",
            agent=tips_agent,
        )

        crew = Crew(
            agents=[tips_agent],
            tasks=[tips_task],
            verbose=True,
        )

        result = crew.kickoff()
        tips_text = str(result)
    except Exception as e:
        print("Error in tips generation:", e)
        traceback.print_exc()
        tips_text = f"Error generating interview tips: {e}"

    return {
        "role": role,
        "experience_level": exp,
        "tech_stack": tech,
        "interview_tips": tips_text,
    }
