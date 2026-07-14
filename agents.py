from __future__ import annotations
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel
import asyncio
import json
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from config import settings


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
        fallback_builder: Optional[Callable[["Agent", str, List[Dict[str, Any]]], BaseModel]] = None,
    ):
        self.name = name
        self.instructions = instructions
        self.model = model
        self.output_type = output_type
        self.model_settings = model_settings
        self.tools = tools or []
        self.fallback_builder = fallback_builder

    def __repr__(self) -> str:
        return f"Agent(name={self.name}, model={self.model})"


class AgentRunResult:
    def __init__(self, final_output: BaseModel, new_items: Optional[List[TResponseInputItem]] = None):
        self.final_output = final_output
        self.new_items = new_items or []

    def final_output_as(self, output_type: type) -> Any:
        if output_type is str:
            if hasattr(self.final_output, "text"):
                return getattr(self.final_output, "text")
            if hasattr(self.final_output, "model_dump"):
                data = self.final_output.model_dump()
                if isinstance(data, dict) and "text" in data:
                    return data["text"]
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
        "요약: 입력된 주요 QC 지표를 기반으로 검토하였습니다.\n"
        "평가: 현재 데이터는 대략적인 QC 검토에 적합하지만, 실제 분석 전에는 원시 도구 결과를 반드시 확인해야 합니다."
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
            break

    if not qc_text:
        return (
            "최종 QC 종합 보고서\n"
            "------------------------------\n"
            "QC 결과를 생성할 수 없습니다.\n"
            "입력된 데이터를 다시 확인하고, 다시 실행해 주세요."
        )

    report_lines = [
        "최종 QC 종합 보고서",
        "------------------------------",
        "QC 요약:",
        qc_text,
        "",
        "종합 의견:",
        "- 이 보고서는 현재 입력된 QC 요약을 기반으로 생성되었습니다.",
        "- 실제 QC 도구 결과와 원본 파일을 함께 검토해야 합니다.",
        "",
        "권장 사항:",
        "- 필요 시 NanoPlot / NanoStat 결과를 추가로 확인하세요.",
        "- 이상 지표가 발견되면 재분석 또는 시퀀싱 재수행을 고려하세요.",
    ]

    return "\n".join(report_lines)


def _openai_api_key() -> str:
    key = getattr(settings, "OPENAI_API_KEY", "")
    if not key:
        return ""

    key = key.strip()
    if key.startswith("<") and key.endswith(">"):
        key = key[1:-1].strip()

    if not key or key.startswith("<") or key.endswith(">"):
        return ""

    return key


def _build_openai_messages(agent: Agent, input: list[Dict[str, Any]]) -> list[Dict[str, str]]:
    messages = [{"role": "system", "content": agent.instructions}]
    for item in input:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content", [])
        if not content or not isinstance(content, list) or not isinstance(content[0], dict):
            continue
        text = content[0].get("text", "")
        if role in ("user", "assistant"):
            messages.append({"role": role, "content": text})
    return messages


def _ncbi_pubmed_search(query: str, retmax: int = 3) -> list[str]:
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": str(retmax),
    }
    request_url = f"{base_url}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(request_url, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
            ids = data.get("esearchresult", {}).get("idlist", [])
            return [str(pid) for pid in ids if isinstance(pid, str)]
    except Exception as exc:
        print(f"NCBI search failed: {exc}")
        return []


def _ncbi_pubmed_summary(pubmed_ids: list[str]) -> str:
    if not pubmed_ids:
        return ""
    base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pubmed_ids),
        "retmode": "json",
    }
    request_url = f"{base_url}?{urllib.parse.urlencode(params)}"
    try:
        with urllib.request.urlopen(request_url, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
            result = data.get("result", {})
            lines = []
            for pid in pubmed_ids:
                item = result.get(pid, {})
                if item:
                    title = item.get("title", "")
                    pubdate = item.get("pubdate", "")
                    source = item.get("source", "")
                    lines.append(f"- [{pid}] {title} ({source}, {pubdate})")
            return "\n".join(lines)
    except Exception as exc:
        print(f"NCBI summary failed: {exc}")
        return ""


def _build_openai_messages(agent: Agent, input: list[Dict[str, Any]]) -> list[Dict[str, str]]:
    messages = [{"role": "system", "content": agent.instructions}]
    for item in input:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content", [])
        if not content or not isinstance(content, list) or not isinstance(content[0], dict):
            continue
        text = content[0].get("text", "")
        if role in ("user", "assistant"):
            messages.append({"role": role, "content": text})
    return messages


def _openai_request(agent: Agent, input: list[Dict[str, Any]]) -> str | None:
    api_key = _openai_api_key()
    if not api_key:
        return None

    pubmed_query = None
    if agent.name.endswith("QC Agent"):
        pubmed_query = f"{agent.name.replace(' QC Agent', '')} QC thresholds"
    elif agent.name == "Report Agent":
        pubmed_query = "bioinformatics QC best practice"

    literature_note = ""
    if pubmed_query:
        pubmed_ids = _ncbi_pubmed_search(pubmed_query, retmax=2)
        summaries = _ncbi_pubmed_summary(pubmed_ids)
        if summaries:
            literature_note = f"\n[NCBI PubMed reference samples]\n{summaries}\n"

    payload: Dict[str, Any] = {
        "model": agent.model,
        "messages": _build_openai_messages(agent, input),
    }
    if literature_note:
        payload["messages"].append({"role": "system", "content": f"Use the following PubMed reference summaries when relevant:{literature_note}"})
    if agent.model_settings and agent.model_settings.temperature is not None:
        payload["temperature"] = agent.model_settings.temperature
    if agent.model_settings and agent.model_settings.top_p is not None:
        payload["top_p"] = agent.model_settings.top_p
    if agent.model_settings and agent.model_settings.max_tokens is not None:
        payload["max_tokens"] = agent.model_settings.max_tokens

    request = urllib.request.Request(
        url="https://api.openai.com/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = response.read().decode("utf-8")
            data = json.loads(body)
            if isinstance(data, dict):
                choices = data.get("choices", [])
                if choices and isinstance(choices[0], dict):
                    message = choices[0].get("message", {})
                    if isinstance(message, dict):
                        return message.get("content", "").strip()
    except urllib.error.HTTPError as exc:
        try:
            error_body = exc.read().decode("utf-8", errors="ignore")
        except Exception:
            error_body = str(exc)
        print(f"OpenAI HTTPError: {exc.code} {error_body}")
    except urllib.error.URLError as exc:
        print(f"OpenAI URLError: {exc}")
    except Exception as exc:
        print(f"OpenAI request failed: {exc}")
    return None


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

        openai_text = None
        if _openai_api_key():
            openai_text = await __import__("asyncio").to_thread(_openai_request, agent, input)

        if openai_text is not None:
            if agent.output_type:
                try:
                    parsed = json.loads(openai_text)
                    final_output = agent.output_type(**parsed)
                except Exception:
                    if agent.fallback_builder:
                        final_output = agent.fallback_builder(agent, user_text, input)
                    else:
                        raise
            else:
                final_output = GenericOutput(text=openai_text)
        else:
            if agent.fallback_builder:
                final_output = agent.fallback_builder(agent, user_text, input)
            else:
                final_output = GenericOutput(text=f"Agent {agent.name} response for given input.")

        output_text = None
        if isinstance(final_output, GenericOutput):
            output_text = final_output.text
        else:
            output_data = final_output.model_dump() if hasattr(final_output, "model_dump") else None
            if isinstance(output_data, dict) and "text" in output_data:
                output_text = output_data["text"]
            elif hasattr(final_output, "json"):
                output_text = final_output.json()
            else:
                output_text = str(final_output)

        new_items = [
            TResponseInputItem(
                role="assistant",
                content=[{"type": "output_text", "text": output_text}],
            )
        ]
        return AgentRunResult(final_output=final_output, new_items=new_items)
