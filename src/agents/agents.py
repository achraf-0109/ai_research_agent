from langchain_ollama import ChatOllama
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.tools.tools import web_search, scrape_url
from dotenv import load_dotenv
import os

# load env variables
load_dotenv()
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")

# build core llm engine
llm = ChatOllama(
    model="gpt-oss:120b",
    base_url="https://ollama.com",
    client_kwargs={
        "headers": {
            "Authorization": f"Bearer {OLLAMA_API_KEY}"
            }
    },
)

# search agent
def search_agent():
    return create_agent(
        model=llm,
        tools=[web_search]
    )

# reader agent
def reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url]
    )

# writer agent
def write_agent():
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
        ("human", """Write a detailed research report on the topic below.
        Topic: {topic}

        Research Gathered:
        {research}

        Structure the report as:
        - Introduction
        - Key Findings (minimum 3 well-explained points)
        - Conclusion
        - Sources (list all URLs found in the research)

        Be detailed, factual and professional.""")
    ])

    return prompt | llm | StrOutputParser()

# critic agent
def critic_agent():
    prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a strict but fair research editor. Review reports critically and give specific, actionable feedback."),
    ("human", """Review the research report below.
    Topic: {topic}

    Research Gathered:
    {research}

    Report to Review:
    {report}

    Evaluate the report on:
    - Accuracy: are the claims supported by the research gathered?
    - Completeness: does it cover the topic well, with at least 3 well-explained key findings?
    - Structure: does it have an Introduction, Key Findings, Conclusion and Sources?
    - Clarity and professionalism of the writing
    - Sources: are all URLs from the research listed correctly?

    Respond in this format:

    Score: (1-10)

    Strengths:
    - ...

    Weaknesses:
    - ...

    Suggested Improvements:
    - ...

    Verdict: (APPROVED or NEEDS REVISION)

    Be specific. Quote or point to the parts of the report you are criticizing."""),
    ])

    return prompt | llm | StrOutputParser()