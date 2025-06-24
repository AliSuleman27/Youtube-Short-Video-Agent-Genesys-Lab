from config.script_config import ScriptConfig
from agents.scriptagent import ScriptAgents
from crewai import Crew
from crewai.process import Process
from tasks.tasks import ScriptTasks

# Main Crew Setup
class ScriptCrew:
    
    def __init__(self, config: ScriptConfig):
        self.config = config
        self.agents = ScriptAgents()
        self.tasks = ScriptTasks(config)
    
    def run_pipeline(self, topic: str) -> str:
        # Create tasks
        research_task = self.tasks.research_task(topic)
        outline_task = self.tasks.outline_task(research_task.output.raw_output)
        script_task = self.tasks.script_task(outline_task.output.raw_output, research_task.output.raw_output)
        qa_task = self.tasks.qa_task(script_task.output.raw_output)
        
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
        result = crew.kickoff()
        return result
