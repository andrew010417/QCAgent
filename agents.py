from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


class Reasoning(BaseModel):
    effort: Optional[str] = None
    summary: Optional[str] = None


class ModelSettings(BaseModel):
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    store: Optional[bool] = None
    reasoning: Optional[Reasoning] = None


class RunConfig(BaseModel):
    trace_metadata: Dict[str, Any] = field(default_factory=dict)


class TResponseInputItem(BaseModel):
    role: str
    content: List[Dict[str, Any]]

    def to_input_item(self) -> Dict[str, Any]:
        return self.dict()


class Agent:
    def __init__(
        self,
        name: str,
        instructions: str,
        model: str,
        output_type: Optional[type[BaseModel]] = None,
        model_settings: Optional[ModelSettings] = None,
    ):
        self.name = name
        self.instructions = instructions
        self.model = model
        self.output_type = output_type
        self.model_settings = model_settings

    def __repr__(self) -> str:
        return f"Agent(name={self.name}, model={self.model})"


class AgentRunResult:
    def __init__(self, final_output: BaseModel, new_items: Optional[List[TResponseInputItem]] = None):
        self.final_output = final_output
        self.new_items = new_items or []

    def final_output_as(self, output_type: type) -> Any:
        if output_type is str:
            if hasattr(self.final_output, "json"):
                return self.final_output.json()
            return str(self.final_output)
        return self.final_output


class GenericOutput(BaseModel):
    text: str


class Runner:
    @staticmethod
    async def run(agent: Agent, input: list[Dict[str, Any]], run_config: RunConfig) -> AgentRunResult:
        user_text = ""
        if input and isinstance(input, list):
            for item in reversed(input):
                if isinstance(item, dict) and item.get("role") == "user":
                    content = item.get("content", [])
                    if content and isinstance(content, list) and isinstance(content[0], dict):
                        user_text = content[0].get("text", "")
                        break
        output_text = ""
        if agent.name == "Classify":
            text = user_text.lower()
            if any(k in text for k in ["rna-seq", "rna seq", "transcript", "deg", "gene expression"]):
                category = "RNA-seq"
            elif any(k in text for k in ["wgs", "genome", "assembly", "busco", "qv", "n50"]):
                category = "WGS"
            elif any(k in text for k in ["methylation", "dmr", "cpg", "epic", "450k"]):
                category = "Methylation"
            elif any(k in text for k in ["hifi", "pacs", "ccs", "long read", "high fidelity"]):
                category = "HiFi"
            elif any(k in text for k in ["ont", "nanopore", "minion", "promethion"]):
                category = "ONT"
            elif any(k in text for k in ["illumina", "short read", "ngs", "paired-end"]):
                category = "Illumina"
            elif any(k in text for k in ["hi-c", "chromatin", "contact map"]):
                category = "Hi-C"
            elif any(k in text for k in ["single cell", "scrna", "10x", "cell ranger"]):
                category = "Single-cell"
            elif any(k in text for k in ["atac", "chromatin accessibility", "open chromatin"]):
                category = "ATAC-seq"
            else:
                category = "Unknown"
            final_output = agent.output_type(category=category) if agent.output_type else GenericOutput(text=category)
        else:
            if "Data Classifer" in agent.name:
                final_output = GenericOutput(text="Classified input for routing")
            else:
                final_output = GenericOutput(text=f"Agent {agent.name} response for given input.")
        new_items = [TResponseInputItem(role="assistant", content=[{"type": "output_text", "text": output_text or final_output.json()}])]
        return AgentRunResult(final_output=final_output, new_items=new_items)
