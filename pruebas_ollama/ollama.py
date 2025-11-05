import os
import configparser
from crewai import Agent, Task, Crew

# --- Configuración ---
config = configparser.ConfigParser()
config.read('../pruebas_openai/config.ini')

OPENAI_MODEL = config['DEFAULT']['OPENAI_MODEL']
OPENAI_API_KEY = config['DEFAULT']['OPENAI_API_KEY']

# 👇 Aquí el cambio clave
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

print(f"Modelo cargado: {OPENAI_MODEL}")

# --- Agentes ---
content_agent = Agent(
    name="ContentBiasAnalyzer",
    role="Content Bias Analyzer",
    goal="Detect bias in topics, focus, or representation in text.",
    backstory="You are an expert in gender studies who detects implicit and explicit gender bias in written content.",
    llm=OPENAI_MODEL
)

language_agent = Agent(
    name="LanguageBiasAnalyzer",
    role="Language Bias Analyzer",
    goal="Detect sexist or gendered language and suggest neutral phrasing.",
    backstory="You are a linguist specialized in gender-inclusive language and discourse analysis.",
    llm=OPENAI_MODEL
)

advisor_agent = Agent(
    name="GenderSensitivityAdvisor",
    role="Inclusive Language Advisor",
    goal="Provide inclusive rewrites and explain why they are better.",
    backstory="You are a communication expert who reformulates biased text into inclusive and neutral expressions.",
    llm=OPENAI_MODEL
)

orchestrator = Agent(
    name="Orchestrator",
    role="Pipeline Manager",
    goal="Coordinate the analysis and combine the agents' results.",
    backstory="You manage a multi-agent analysis pipeline and ensure each agent contributes to a coherent output.",
    llm=OPENAI_MODEL
)

# --- Tasks ---
task_content = Task(
    description="Detect content-level gender bias in the given text.",
    expected_output="List of content biases.",
    agent=content_agent,
    inputs={"text": "She was just a nurse, but he was a surgeon."}
)

task_language = Task(
    description="Detect linguistic sexism or gendered language in the given text.",
    expected_output="List of language biases.",
    agent=language_agent,
    inputs={"text": "She was just a nurse, but he was a surgeon."}
)

task_advisor = Task(
    description="Suggest inclusive rewrites of the text and explain the reasoning.",
    expected_output="List of inclusive alternatives with explanations.",
    agent=advisor_agent,
    inputs={"text": "She was just a nurse, but he was a surgeon."}
)

task_summary = Task(
    description="Combine all results into a final JSON report with keys: content_bias, language_bias, suggestions.",
    expected_output="Final JSON summary.",
    agent=orchestrator,
    context=[task_content, task_language, task_advisor]
)

# --- Crew ---
crew = Crew(
    name="GenderBiasPipeline",
    agents=[orchestrator, content_agent, language_agent, advisor_agent],
    tasks=[task_content, task_language, task_advisor, task_summary],
    verbose=True
)

result = crew.kickoff()
print(result)

