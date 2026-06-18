# agents.py
# This module defines the agents for the application, including the Search Agent and Reader Agent.

from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import search_web, scrape_url
import os
from dotenv import load_dotenv

load_dotenv()

# Set the GROQ API key from environment variable
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")


# Initialize the language model
llm = ChatGroq(model="qwen/qwen3-32b", temperature=0)


# Agents

# Search Agent: Uses the search_web tool to find relevant information on the web.
def build_search_agent():
    return create_agent(
        model = llm,
        tools = [search_web],
    )

# Reader Agent: Uses the scrape_url tool to extract information from a given URL.
def build_reader_agent():
    return create_agent(
        model = llm,
        tools = [scrape_url],
        )
        

# Writer Chain

writer_prompt = ChatPromptTemplate.from_messages([
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

Be detailed, factual and professional."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()

# Critic Chain
critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()