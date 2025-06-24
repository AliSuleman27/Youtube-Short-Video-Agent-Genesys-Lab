from typing import Optional
from pydantic import BaseModel, Field


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
