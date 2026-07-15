# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A CLI tool that takes a free-text description of an omics QC workflow (e.g. `RNA-seq fastq, 50M reads, mapping 90%, Q30 92%`), classifies the data type, recommends QC tools, then lets the user paste in actual QC metrics for an LLM-driven evaluation and a synthesized Korean-language report. Interaction and report output are in Korean; code/comments are in English.

## Running it

```bash
python bioQcAgent.py
```

This is a thin `sys.path` shim around `app.py:main()`; running `python app.py` directly works the same way. There is no build step, package manifest, lint config, or test suite in this repo — the only third-party dependency is `pydantic` (everything else is stdlib: `sqlite3`, `urllib`, `xml.etree.ElementTree`, `asyncio`).

### API keys

`config.py` reads `OPENAI_API_KEY` / `NCBI_API_KEY` from the environment first, falling back to constants in `api_key.py` (gitignored, not committed — create it locally with those two variables if you want live OpenAI/NCBI calls). Without a valid key, the tool still runs end-to-end using deterministic rule-based fallbacks (see below), so it's testable with no network access.

## Architecture

The important thing to understand here: **`agents.py` is a hand-rolled local shim, not the real `openai-agents` SDK.** It re-implements just enough of that SDK's surface (`Agent`, `Runner`, `RunConfig`, `trace`, `ModelSettings`, `Reasoning`, `TResponseInputItem`) for this project to work without the actual package installed. Don't go looking for SDK docs/behavior beyond what's in this file — its `Runner.run` is a plain function that:

1. Extracts the latest user message text from the input list.
2. If an OpenAI key is configured, calls `chat/completions` directly via `urllib` (`_openai_request`), first enriching the prompt with PubMed literature (see below).
3. If no key is configured, or the request fails (or the model's JSON doesn't parse into `output_type`), falls back to deterministic logic — but *which* logic is not decided by string-matching `agent.name`. It's driven by two `Agent` fields:
   - `output_type` (a pydantic model class, e.g. `ClassifySchema`): when set, `Runner.run` tries `json.loads(openai_text)` then `output_type(**parsed)`; when unset, the raw text is wrapped in `GenericOutput`.
   - `fallback_builder` (`Callable[[Agent, str, list], BaseModel]`): called with `(agent, user_text, input)` whenever there's no OpenAI text to use, or `output_type` parsing failed. Each `Agent` in `agents_definition.py` supplies its own (`_classify_fallback`, `_data_classifier_fallback`, `_qc_agent_fallback`, `_report_agent_fallback`), built on top of the same rule-based helpers (`classify_category`, `build_qc_summary`, `build_report_summary`).

   `report_agent` uses `output_type=QCReportSchema` (`agents_definition.py`): `category`, `verdict`, `summary`, a structured `metrics: list[QCMetricResult]` (metric/user_value/standard/status/recommendation), `recommendations: list[str]`, plus a `text` field holding the full Korean Markdown report. `AgentRunResult.final_output_as(str)` special-cases any output model with a `text` attribute and returns it directly, so `workflow.py`/`app.py`/`db.py` keep consuming a plain Markdown string unchanged while callers that want structured data (e.g. for a future visualization layer) can read `final_output` directly instead of going through `final_output_as(str)`. In fallback mode (`_report_agent_fallback`), only `text` (via `build_report_summary`) and `category` are derived from the real QC output; `metrics`/`verdict` are coarse placeholders (all `WARNING`, `user_value`/`standard` as `"-"`) since the rule-based fallback can't do real threshold evaluation — only the OpenAI path fills in real per-metric values and statuses.
   
   Adding structured/fallback behavior for a new agent means setting these two fields on that `Agent`, not editing `Runner.run`.

**NCBI PubMed integration** also lives in `agents.py`: before calling OpenAI for a QC or Report agent, `_openai_request` queries NCBI eutils (`esearch` then `esummary`) for a couple of relevant papers and injects them as an extra system message ("Use the following PubMed reference summaries...").

**Two-stage workflow** (`workflow.py`), driven interactively by `app.py`:
- `prepare_workflow`: runs `data_classifer` (rough free-text classifier) then `classify` (strict, schema-constrained via `ClassifySchema` → one of the fixed categories) to pick a category, then looks up the matching QC agent and recommended tools. Returned to the user before they type in actual experiment numbers.
- `evaluate_workflow`: takes the user's pasted QC metrics, runs the category's specialist QC agent, and — if the category is a known one — also runs `report_agent` to synthesize a final Korean Markdown report with a PASS/WARNING/FAIL verdict table.

**Agent/category routing** (`agents_definition.py`): one `Agent` per omics data type (RNA-seq, WGS, Methylation, HiFi, ONT, Illumina, Hi-C, Single-cell, ATAC-seq), each with its own instructions, model name, and recommended CLI tools baked into the prompt (FastQC, STAR, BUSCO, ChAMP, NanoPlot, Seurat, etc.). `QC_AGENT_MAP` and `TOOL_MAP` dicts key off the category string produced by `classify`; `get_qc_agent`/`get_recommended_tools`/`should_run_report` are the lookup helpers everything else calls — add a new omics category by adding an `Agent` plus entries in both maps.

**Persistence** (`db.py`): a single SQLite table `workflow_runs` at `./data/workflow_results.db` (path from `Settings.STORAGE_PATH`), storing input text, classified category, QC output, and report output per run. `init_db()` runs on every `app.py` startup.

## Working in this codebase

- When changing agent behavior, the instructions/prompts in `agents_definition.py` *are* the behavior — there's no separate prompt-template layer.
- Since `agents.py` silently falls back to rule-based output when there's no API key, changes there should be checked in both modes (with and without `OPENAI_API_KEY` set) to make sure the fallback path still produces sane output.
- New omics categories need four touch points kept in sync (all in `agents_definition.py`): the `CATEGORIES`/rules text in `classify`/`data_classifer` instructions, a new `Agent` definition (with `fallback_builder=_qc_agent_fallback` if it's a QC agent), and entries in `QC_AGENT_MAP`/`TOOL_MAP`.
- `Runner.run` in `agents.py` never branches on `agent.name` — behavior is entirely determined by each `Agent`'s `output_type`/`fallback_builder` fields. Keep it that way; if a new agent needs special-cased behavior, give it its own `fallback_builder` in `agents_definition.py` rather than adding a name check back into `Runner.run`.
