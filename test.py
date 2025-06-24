import os
from typing import List, Dict, Optional
from crewai import Agent, Task, Crew, Process
from langchain.tools import tool
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_community.document_loaders import YoutubeLoader
from datetime import datetime
import json
from pydantic import BaseModel, Field
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from config.config import *
from dotenv import load_dotenv
import os


# Loading the environment
load_dotenv()

# Creat the Chat Interface
gemini_llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash",
                           verbose=LLM_VERBOSE,
                           temperature=LLM_TEMPERATURE,
                           google_api_key=os.getenv("GOOGLE_API_KEY"))



# Configuration Model
class ScriptConfig(BaseModel):
    target_audience: str = Field(..., description="Primary audience for the script")
    tone: str = Field("professional", description="Tone of the script")
    length: str = Field("medium", description="Short (1-2 min), Medium (3-5 min), Long (5+ min)")
    style: str = Field("explainer", description="Style of the script (explainer, narrative, persuasive, etc.)")
    complexity: str = Field("intermediate", description="Complexity level (beginner, intermediate, advanced)")
    include_examples: bool = Field(True, description="Whether to include real-world examples")
    include_stats: bool = Field(True, description="Whether to include relevant statistics")
    call_to_action: Optional[str] = Field(None, description="Optional call to action at the end")

# Tools Setup
class ResearchTools:
    
    @tool("Search the web using Google Serper")
    def search_web(query: str) -> str:
        """Performs a Google search using Serper API and returns results."""
        search = GoogleSerperAPIWrapper()
        return json.dumps(search.results(query))
    
    @tool("Get YouTube video transcripts")
    def get_youtube_transcript(video_url: str) -> str:
        """Fetches and returns the transcript of a YouTube video."""
        loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=True)
        transcript = loader.load()
        return "\n".join([doc.page_content for doc in transcript])
    
    @tool("Search Quora for topic insights")
    def search_quora(query: str) -> str:
        """Searches Quora for questions and answers related to a topic."""
        # Note: In production, you'd use Quora's API or a web scraping solution
        url = f"https://www.quora.com/search?q={query.replace(' ', '+')}"
        # This is a placeholder - actual implementation would parse the results
        return f"Quora results for {query} available at: {url}"

# Initialize tools
research_tools = ResearchTools()

# Agent Definitions
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
            tools=[
                research_tools.search_web,
                research_tools.get_youtube_transcript,
                research_tools.search_quora
            ],
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
            llm=gemini_llm
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
            llm=gemini_llm
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
            llm=gemini_llm
        )

agents = ScriptAgents()


# Task Definitions
class ScriptTasks:
    
    def __init__(self, config: ScriptConfig):
        self.config = config
    
    def research_task(self, topic: str) -> Task:
        return Task(
            description=(
                f"Conduct comprehensive research on the topic: {topic}\n"
                f"Audience: {self.config.target_audience}\n"
                f"Key requirements:\n"
                "- Gather information from at least 3 different sources (web, YouTube, Quora)\n"
                "- Identify key facts, statistics, and examples\n"
                "- Note different perspectives on the topic\n"
                "- Highlight any controversies or debates\n"
                "- Find recent developments (last 12 months)\n"
                f"Additional notes: {self._research_style_notes()}"
            ),
            expected_output=(
                "A comprehensive research brief containing:\n"
                "1. Key facts and information about the topic\n"
                "2. Relevant statistics and data points\n"
                "3. Real-world examples and case studies\n"
                "4. Different perspectives or schools of thought\n"
                "5. Recent developments and trends\n"
                "6. Potential gaps in available information\n"
                "Formatted as a detailed markdown document with sources cited"
            ),
            agent=agents.research_agent
        )
    
    def outline_task(self, research_data: str) -> Task:
        return Task(
            description=(
                f"Create a detailed script outline based on the research provided.\n"
                f"Key requirements:\n"
                "- Structure content for optimal flow and engagement\n"
                "- Include hooks and transitions\n"
                "- Balance information with entertainment value\n"
                "- Adapt structure to {self.config.length} length\n"
                "- Use {self.config.style} style\n"
                f"Additional notes: {self._outline_style_notes()}"
            ),
            expected_output=(
                "A structured script outline containing:\n"
                "1. Introduction with hook\n"
                "2. Main points with sub-points\n"
                "3. Transitions between sections\n"
                "4. Supporting evidence/examples for each point\n"
                "5. Conclusion with summary\n"
                "6. Call-to-action (if specified)\n"
                "Formatted as a hierarchical markdown document"
            ),
            agent=agents.outline_agent
        )
    
    def script_task(self, outline: str, research: str) -> Task:
        return Task(
            description=(
                f"Write a full script based on the outline and research.\n"
                f"Key requirements:\n"
                "- Write in a {self.config.tone} tone\n"
                "- Target {self.config.complexity} complexity level\n"
                "- Make it voiceover-friendly with natural pauses\n"
                "- Include {'' if self.config.include_examples else 'no '}examples\n"
                "- Include {'' if self.config.include_stats else 'no '}statistics\n"
                f"Additional notes: {self._script_style_notes()}"
            ),
            expected_output=(
                "A complete script with:\n"
                "1. Natural, conversational language\n"
                "2. Proper pacing indicators (pauses, emphasis)\n"
                "3. Clear section transitions\n"
                "4. Appropriate tone and complexity\n"
                "5. Marked examples and statistics (if included)\n"
                "6. Call-to-action (if specified)\n"
                "Formatted as a professional script with timing estimates"
            ),
            agent=agents.script_agent
        )
    
    def qa_task(self, script: str) -> Task:
        return Task(
            description=(
                f"Review and optimize the script for production readiness.\n"
                f"Key requirements:\n"
                "- Check for factual accuracy against research\n"
                "- Ensure logical flow and narrative coherence\n"
                "- Optimize for audience engagement\n"
                "- Verify tone and style consistency\n"
                "- Check timing and pacing\n"
                "- Suggest improvements for clarity and impact"
            ),
            expected_output=(
                "A production-ready script with:\n"
                "1. All factual inaccuracies corrected\n"
                "2. Improved flow and engagement\n"
                "3. Timing adjustments if needed\n"
                "4. Editor's notes on performance suggestions\n"
                "5. Version history of changes made\n"
                "Formatted as a clean markdown document with change tracking"
            ),
            agent=agents.qa_agent
        )
    
    def _research_style_notes(self) -> str:
        notes = {
            "explainer": "Focus on clear, authoritative sources. Prioritize educational content.",
            "narrative": "Look for compelling stories and case studies.",
            "persuasive": "Find strong arguments and counterarguments.",
            "professional": "Prioritize academic and industry sources.",
            "casual": "Include popular media and informal discussions."
        }
        return notes.get(self.config.style, "")
    
    def _outline_style_notes(self) -> str:
        notes = {
            "short": "Be concise. 1 main point with 1-2 supporting points.",
            "medium": "2-3 main points with 2-3 supporting points each.",
            "long": "3-5 main points with multiple supporting points and examples."
        }
        return notes.get(self.config.length, "")
    
    def _script_style_notes(self) -> str:
        notes = {
            "beginner": "Use simple language. Define all terms. More examples.",
            "intermediate": "Balance technical and accessible language.",
            "advanced": "Can use specialized terminology. Fewer explanations."
        }
        return notes.get(self.config.complexity, "")

# Main Crew Setup
class ScriptCrew:
    
    def __init__(self, config: ScriptConfig):
        self.config = config
        self.agents = agents
        self.tasks = ScriptTasks(config)
    
    def run_pipeline(self, topic: str) -> str:
        # Create tasks
        research_task = self.tasks.research_task(topic)
        outline_task = self.tasks.outline_task("{{research_output}}")  # Placeholder
        script_task = self.tasks.script_task("{{outline_output}}", "{{research_output}}")  # Placeholders
        qa_task = self.tasks.qa_task("{{script_output}}")  # Placeholder
        
        # Assemble crew
        crew = Crew(
            agents=[
                self.agents.research_agent,
                self.agents.outline_agent,
                self.agents.script_agent,
                self.agents.qa_agent
            ],
            tasks=[research_task, outline_task, script_task, qa_task],
            process=Process.sequential,
            verbose=2
        )
        
        # Execute pipeline
        result = crew.kickoff(inputs={'topic': topic})
        return result

# Example Usage
if __name__ == "__main__":
    # Configuration
    config = ScriptConfig(
        target_audience="tech entrepreneurs",
        tone="professional",
        length="medium",
        style="explainer",
        complexity="intermediate",
        include_examples=True,
        include_stats=True,
        call_to_action="Subscribe for more tech insights"
    )
    
    # Initialize and run
    crew = ScriptCrew(config)
    topic = "The impact of AI on small business marketing strategies 2024"
    final_script = crew.run_pipeline(topic)
    
    # Save output
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"script_output_{timestamp}.md"
    with open(filename, "w") as f:
        f.write(final_script)
    
    print(f"Script generation complete. Output saved to {filename}")