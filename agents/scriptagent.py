# Agent Definitions
from crewai import Agent
from tools.research_tools import research_tools
from config.llm_config import gemini_llm

class ScriptAgents:
    def __init__(self):
        self.research_agent = self._create_research_agent()
        self.outline_agent = self._create_outline_agent()
        self.script_agent = self._create_script_agent()
        self.qa_agent = self._create_qa_agent()
    
    def _create_research_agent(self) -> Agent:
        return Agent(
            role="Senior Research Analyst",
            goal="Conduct thorough research on the given topic using multiple sources to create a comprehensive topic brief",
            backstory=(
                "You are an expert researcher with years of experience in gathering, analyzing, "
                "and synthesizing information from diverse sources. You know how to find the most "
                "relevant, accurate, and up-to-date information on any topic."
            ),
            tools=research_tools,
            verbose=True,
            allow_delegation=False,
            memory=True,
            llm=gemini_llm
        )
    
    def _create_outline_agent(self) -> Agent:
        return Agent(
            role="Senior Content Strategist",
            goal="Create a well-structured, logical outline for a script based on research materials",
            backstory=(
                "You are a content architect with a talent for organizing complex information "
                "into clear, engaging structures. You understand narrative flow, audience engagement, "
                "and how to build compelling content frameworks."
            ),
            verbose=True,
            allow_delegation=False,
            memory=True,
            llm = gemini_llm
        )
    
    def _create_script_agent(self) -> Agent:
        return Agent(
            role="Senior Script Writer",
            goal="Transform research and outlines into compelling, voiceover-friendly scripts",
            backstory=(
                "You are an accomplished scriptwriter with experience in creating engaging, "
                "natural-sounding scripts for videos, podcasts, and presentations. You know how "
                "to make complex topics accessible and entertaining."
            ),
            verbose=True,
            allow_delegation=False,
            memory=True,
            llm = gemini_llm
        )
    
    def _create_qa_agent(self) -> Agent:
        return Agent(
            role="Quality Assurance Editor",
            goal="Review and optimize scripts for accuracy, flow, engagement, and production readiness",
            backstory=(
                "You are a meticulous editor with an eye for detail and a ear for natural language. "
                "You ensure all content meets the highest standards of quality, accuracy, and "
                "effectiveness before production."
            ),
            verbose=True,
            allow_delegation=False,
            memory=True,
            llm = gemini_llm
        )