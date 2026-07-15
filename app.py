import asyncio
from pathlib import Path

from db import init_db, save_workflow_result
from workflow import evaluate_workflow, prepare_workflow
from agents_definition import WorkflowInput
from visualize import save_qc_chart


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


def main():
    init_db()

    input_text = input("Enter QC workflow text: ").strip()
    if not input_text:
        print("No input provided. Exiting.")
        return

    workflow_input = WorkflowInput(input_as_text=input_text)

    prepare_result = asyncio.run(prepare_workflow(workflow_input))
    print("\n--- Tool Recommendation Stage ---")
    print(f"Category: {prepare_result['classification']['output_parsed']['category']}")
    print(f"Selected QC Agent: {prepare_result['qc_agent_name']}")
    print("Recommended tools:")
    for tool in prepare_result['recommended_tools']:
        print(f"- {tool}")

    category = prepare_result["category"]
    purpose_example = ANALYSIS_PURPOSE_EXAMPLES.get(category, "de novo assembly")
    analysis_purpose = input(f"\n분석 목적을 입력하세요 (예: {purpose_example}): ").strip()

    print("\nEnter experiment/test data details for evaluation. Type END on its own line to finish:")
    lines = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)

    experiment_text = "\n".join(lines).strip()
    if not experiment_text:
        print("No experiment data provided. Exiting after recommendation stage.")
        return

    if analysis_purpose:
        experiment_text = f"분석 목적: {analysis_purpose}\n\n{experiment_text}"

    evaluate_result = asyncio.run(
        evaluate_workflow(
            workflow_input,
            experiment_text=experiment_text,
            category=category,
        )
    )

    print("\n--- Evaluation Result ---")
    print("QC Result:")
    print(evaluate_result["qc_result"]["output_text"])
    if evaluate_result["report_result"]:
        print("\nReport Result:")
        print(evaluate_result["report_result"]["output_text"])

    run_id = save_workflow_result(
        input_text=input_text,
        category=category,
        qc_output=evaluate_result["qc_result"]["output_text"],
        report_output=evaluate_result["report_result"]["output_text"] if evaluate_result["report_result"] else None,
    )
    print(f"Saved run id: {run_id}")

    report_result = evaluate_result["report_result"]
    metrics = report_result.get("metrics") if report_result else None
    if metrics:
        charts_dir = Path("./data/charts")
        charts_dir.mkdir(parents=True, exist_ok=True)
        chart_path = charts_dir / f"qc_chart_run{run_id}.png"
        save_qc_chart(metrics, chart_path)
        print(f"차트 저장됨: {chart_path}")


if __name__ == "__main__":
    main()
