from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pydantic import BaseModel


@contextmanager
def trace(name: str):
    yield


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
        tools: Optional[List[str]] = None,
    ):
        self.name = name
        self.instructions = instructions
        self.model = model
        self.output_type = output_type
        self.model_settings = model_settings
        self.tools = tools or []

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


def classify_category(text: str) -> str:
    lower = text.lower()
    if any(k in lower for k in ["rna-seq", "rna seq", "transcript", "deg", "gene expression"]):
        return "RNA-seq"
    if any(k in lower for k in ["wgs", "genome", "assembly", "busco", "qv", "n50"]):
        return "WGS"
    if any(k in lower for k in ["methylation", "dmr", "cpg", "epic", "450k"]):
        return "Methylation"
    if any(k in lower for k in ["hifi", "pacs", "ccs", "long read", "high fidelity"]):
        return "HiFi"
    if any(k in lower for k in ["ont", "nanopore", "minion", "promethion"]):
        return "ONT"
    if any(k in lower for k in ["illumina", "short read", "ngs", "paired-end"]):
        return "Illumina"
    if any(k in lower for k in ["hi-c", "chromatin", "contact map"]):
        return "Hi-C"
    if any(k in lower for k in ["single cell", "scrna", "10x", "cell ranger"]):
        return "Single-cell"
    if any(k in lower for k in ["atac", "chromatin accessibility", "open chromatin"]):
        return "ATAC-seq"
    return "Unknown"


def build_qc_summary(agent_name: str, user_text: str) -> str:
    category = agent_name.replace(" QC Agent", "")
    metrics = {
        "RNA-seq": ["Mapping rate", "Duplication rate", "Total reads", "Q30 rate"],
        "WGS": ["BUSCO", "QV", "N50", "Coverage"],
        "Methylation": ["Detection rate", "Beta value distribution", "Batch effect"],
        "HiFi": ["Mean read length", "Mean quality score", "N50", "Total bases"],
        "ONT": ["Mean read length", "N50", "Mean quality score", "Q20 ratio"],
        "Illumina": ["Q30 rate", "Total reads", "GC content", "Duplication rate"],
        "Hi-C": ["Valid pairs", "Cis ratio", "Trans ratio", "Duplication rate"],
        "Single-cell": ["Cells detected", "Median genes/cell", "Median UMI/cell", "MT ratio"],
        "ATAC-seq": ["FRiP score", "TSS enrichment", "Duplication rate", "Fragment size distribution"],
    }
    tool_map = {
        "RNA-seq": ["FastQC", "STAR", "RSeQC"],
        "WGS": ["BUSCO", "Merqury", "NanoStat"],
        "Methylation": ["ChAMP", "minfi"],
        "HiFi": ["NanoStat", "cramino"],
        "ONT": ["NanoPlot", "NanoStat"],
        "Illumina": ["FastQC", "fastp", "MultiQC"],
        "Hi-C": ["pairtools", "HiCExplorer"],
        "Single-cell": ["Seurat", "scater", "DoubletFinder"],
        "ATAC-seq": ["FastQC", "ATACseqQC", "deepTools"],
    }
    tool_line = ""
    if agent_name.endswith("QC Agent"):
        tools = tool_map.get(category, [])
        if tools:
            tool_line = "\nRecommended tools: " + ", ".join(tools)
    return (
        f"[{category} QC Evaluation]\n"
        f"분류 입력: {user_text}\n"
        f"중요 지표: {', '.join(metrics.get(category, ['Quality metric']))}\n"
        "판단: 입력된 데이터를 기반으로 QC 지표를 검토하고 권장 사항을 제공합니다.\n"
        "상세 결과는 실제 QC 도구 출력과 함께 확인하세요."
        f"{tool_line}"
    )


def build_report_summary(conversation: list[Dict[str, Any]]) -> str:
    qc_text = None
    for message in conversation:
        if message.get("role") != "assistant":
            continue
        contents = message.get("content", [])
        if not contents or not isinstance(contents, list) or not isinstance(contents[0], dict):
            continue
        text = contents[0].get("text", "")
        if "QC Evaluation" in text:
            qc_text = text

    if not qc_text:
        qc_text = "QC 결과를 생성할 수 없습니다."

    return (
        "최종 QC 종합 보고서\n"
        "------------------------------\n"
        f"{qc_text}\n"
        "결론: 이 데이터는 현재 평가 기준에서 분석 진행 가능 여부를 검토해야 합니다.\n"
        "권고: 필요한 경우 추가 QC를 수행하고, 경고 항목이 있으면 재처리를 고려하세요."
    )


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
        if agent.name == "Classify":
            category = classify_category(user_text)
            final_output = agent.output_type(category=category) if agent.output_type else GenericOutput(text=category)
        elif agent.name == "Data Classifer":
            category = classify_category(user_text)
            final_output = GenericOutput(text=f"Data classifier routed to {category} QC agent.")
        elif agent.name.endswith("QC Agent"):
            final_output = GenericOutput(text=build_qc_summary(agent.name, user_text))
        elif agent.name == "Report Agent":
            final_output = GenericOutput(text=build_report_summary(input))
        else:
            final_output = GenericOutput(text=f"Agent {agent.name} response for given input.")
        new_items = [TResponseInputItem(role="assistant", content=[{"type": "output_text", "text": final_output.json()}])]
        return AgentRunResult(final_output=final_output, new_items=new_items)
