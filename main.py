from config.script_config import ScriptConfig
from pipeline import ScriptCrew
from datetime import datetime

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