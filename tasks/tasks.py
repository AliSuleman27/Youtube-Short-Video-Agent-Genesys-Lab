from config.script_config import ScriptConfig
from crewai import Task
from agents.scriptagent import ScriptAgents

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
