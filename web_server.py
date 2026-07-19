"""Web frontend for bioQCAgent.

A thin HTTP layer over the exact same two-stage workflow the CLI (app.py)
drives — nothing below this file changes. Uses only stdlib http.server so the
repo keeps its "no build step, pydantic-only" property; the async workflow
functions are driven with asyncio.run() per request (ThreadingHTTPServer, so
concurrent requests each get their own event loop, and db.py opens a fresh
sqlite connection per call).

Routes:
  GET  /                      -> the single-page UI (static/index.html)
  POST /api/prepare           -> stage 1: classify + recommend tools
  POST /api/evaluate          -> stage 2: QC eval + report + PDF
  GET  /api/report/{id}/pdf   -> download the generated PDF for a run

Run:  python web_server.py   (serves on http://127.0.0.1:8000)
"""
from __future__ import annotations

import asyncio
import json
import mimetypes
import re
from datetime import date
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from db import init_db, save_workflow_result
from workflow import prepare_workflow, evaluate_workflow
from agents_definition import WorkflowInput
from pdf_report import save_qc_report_pdf

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
ASSETS_DIR = BASE_DIR / "assets"
REPORTS_DIR = BASE_DIR / "data" / "reports"

# Mirrors app.py's per-category analysis-purpose placeholder examples so the
# web form can prompt the user with the same hints the CLI does.
ANALYSIS_PURPOSE_EXAMPLES = {
    "RNA-seq": "차등발현 분석, 바이오마커 발굴, 대안적 스플라이싱 분석",
    "WGS": "SNP calling, 구조 변이 분석, T2T 어셈블리",
    "Methylation": "DMR 분석, 에피게놈 프로파일링, 암 바이오마커 발굴",
    "HiFi": "De novo assembly, SV 분석, T2T genome 구축",
    "ONT": "De novo assembly, SV 분석, 메틸레이션 검출, 전사체 분석",
    "Illumina": "WGS variant calling, RNA-seq, ChIP-seq",
    "Hi-C": "TAD 분석, 염색체 스캐폴딩, 3D 게놈 구조 분석",
    "Single-cell": "세포 타입 분류, 궤적 분석, 희귀 세포 발굴",
    "ATAC-seq": "열린 염색질 분석, 전사인자 결합 예측, 피크 calling",
}
DEFAULT_PURPOSE_EXAMPLE = "de novo assembly"

_PDF_RUN_RE = re.compile(r"^/api/report/(\d+)/pdf$")


def _metric_to_dict(metric) -> dict:
    if hasattr(metric, "model_dump"):
        return metric.model_dump()
    if isinstance(metric, dict):
        return metric
    return {}


class Handler(BaseHTTPRequestHandler):
    server_version = "bioQCAgent/1.0"

    # --- helpers -------------------------------------------------------------
    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_bytes(self, data: bytes, content_type: str, status: int = 200, extra_headers: dict | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        for key, value in (extra_headers or {}).items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            return {}

    # --- routing -------------------------------------------------------------
    def do_GET(self) -> None:
        if self.path in ("/", "/index.html"):
            index = STATIC_DIR / "index.html"
            if not index.exists():
                self._send_json({"error": "index.html not found"}, status=500)
                return
            self._send_bytes(index.read_bytes(), "text/html; charset=utf-8")
            return

        pdf_match = _PDF_RUN_RE.match(self.path)
        if pdf_match:
            self._serve_pdf(int(pdf_match.group(1)))
            return

        if self.path.startswith("/assets/"):
            self._serve_asset(self.path[len("/assets/"):])
            return

        self._send_json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        if self.path == "/api/prepare":
            self._handle_prepare()
        elif self.path == "/api/evaluate":
            self._handle_evaluate()
        else:
            self._send_json({"error": "not found"}, status=404)

    # --- endpoints -----------------------------------------------------------
    def _handle_prepare(self) -> None:
        data = self._read_json()
        input_text = (data.get("input_as_text") or "").strip()
        if not input_text:
            self._send_json({"error": "input_as_text is required"}, status=400)
            return

        try:
            result = asyncio.run(prepare_workflow(WorkflowInput(input_as_text=input_text)))
        except Exception as exc:  # surface failures to the UI rather than a bare 500
            self._send_json({"error": f"prepare failed: {exc}"}, status=500)
            return

        category = result["category"]
        self._send_json({
            "category": category,
            "qc_agent_name": result["qc_agent_name"],
            "recommended_tools": result["recommended_tools"],
            "purpose_example": ANALYSIS_PURPOSE_EXAMPLES.get(category, DEFAULT_PURPOSE_EXAMPLE),
        })

    def _handle_evaluate(self) -> None:
        data = self._read_json()
        input_text = (data.get("input_as_text") or "").strip()
        category = (data.get("category") or "").strip()
        analysis_purpose = (data.get("analysis_purpose") or "").strip()
        experiment_text = (data.get("experiment_text") or "").strip()

        if not input_text or not category:
            self._send_json({"error": "input_as_text and category are required"}, status=400)
            return
        if not experiment_text:
            self._send_json({"error": "experiment_text is required"}, status=400)
            return

        # Same prefixing the CLI applies so the QC agent sees the goal inline.
        if analysis_purpose:
            experiment_text = f"분석 목적: {analysis_purpose}\n\n{experiment_text}"

        try:
            result = asyncio.run(evaluate_workflow(
                WorkflowInput(input_as_text=input_text),
                experiment_text=experiment_text,
                category=category,
            ))
        except Exception as exc:
            self._send_json({"error": f"evaluate failed: {exc}"}, status=500)
            return

        qc_output = result["qc_result"]["output_text"]
        report_result = result["report_result"]
        report_output = report_result["output_text"] if report_result else None

        run_id = save_workflow_result(
            input_text=input_text,
            category=category,
            qc_output=qc_output,
            report_output=report_output,
        )

        response: dict = {
            "run_id": run_id,
            "qc_output": qc_output,
            "report": None,
            "pdf_url": None,
        }

        if report_result:
            metric_objs = report_result.get("metrics") or []
            metrics_json = [_metric_to_dict(m) for m in metric_objs]
            response["report"] = {
                "category": report_result.get("category", category),
                "verdict": report_result.get("verdict"),
                "summary": report_result.get("summary"),
                "metrics": metrics_json,
                "recommendations": report_result.get("recommendations") or [],
                "text": report_output,
            }

            # Generate the downloadable PDF eagerly (same as the CLI does), so a
            # later GET /api/report/{id}/pdf just serves the file.
            try:
                REPORTS_DIR.mkdir(parents=True, exist_ok=True)
                pdf_path = REPORTS_DIR / f"qc_report_run{run_id}.pdf"
                save_qc_report_pdf(
                    category=report_result.get("category", category),
                    analysis_purpose=analysis_purpose,
                    run_date=date.today().isoformat(),
                    report={
                        "verdict": report_result.get("verdict"),
                        "summary": report_result.get("summary"),
                        "metrics": metric_objs,
                        "recommendations": report_result.get("recommendations"),
                        "text": report_output,
                    },
                    output_path=pdf_path,
                )
                response["pdf_url"] = f"/api/report/{run_id}/pdf"
            except Exception as exc:
                # PDF is a nice-to-have; don't fail the whole evaluation over it.
                response["pdf_error"] = str(exc)

        self._send_json(response)

    def _serve_asset(self, rel: str) -> None:
        # Serve files from ./assets at runtime (e.g. the gitignored company
        # logo the PDF letterhead also reads) — kept off the tracked HTML on
        # purpose, and confined to ASSETS_DIR to prevent path traversal.
        rel = rel.split("?", 1)[0].split("#", 1)[0]
        target = (ASSETS_DIR / rel).resolve()
        try:
            target.relative_to(ASSETS_DIR.resolve())
        except ValueError:
            self._send_json({"error": "forbidden"}, status=403)
            return
        if not target.is_file():
            self._send_json({"error": "not found"}, status=404)
            return
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        self._send_bytes(target.read_bytes(), content_type)

    def _serve_pdf(self, run_id: int) -> None:
        pdf_path = REPORTS_DIR / f"qc_report_run{run_id}.pdf"
        if not pdf_path.exists():
            self._send_json({"error": "report not found"}, status=404)
            return
        self._send_bytes(
            pdf_path.read_bytes(),
            "application/pdf",
            extra_headers={"Content-Disposition": f'inline; filename="qc_report_run{run_id}.pdf"'},
        )

    def log_message(self, fmt, *args) -> None:  # concise one-line access log
        print(f"[web] {self.address_string()} {fmt % args}")


def main(host: str = "127.0.0.1", port: int = 8000) -> None:
    init_db()
    server = ThreadingHTTPServer((host, port), Handler)
    print(f"bioQCAgent web UI running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
