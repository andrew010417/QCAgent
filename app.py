import asyncio
from db import init_db, save_workflow_result
from workflow import evaluate_workflow, prepare_workflow
from agents_definition import WorkflowInput


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

    print("\nEnter experiment/test data details for evaluation. Finish with an empty line:")
    lines = []
    while True:
        line = input()
        if not line.strip():
            break
        lines.append(line)

    experiment_text = "\n".join(lines).strip()
    if not experiment_text:
        print("No experiment data provided. Exiting after recommendation stage.")
        return

    evaluate_result = asyncio.run(
        evaluate_workflow(
            workflow_input,
            experiment_text=experiment_text,
            category=prepare_result["category"],
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
        category=prepare_result["category"],
        qc_output=evaluate_result["qc_result"]["output_text"],
        report_output=evaluate_result["report_result"]["output_text"] if evaluate_result["report_result"] else None,
    )
    print(f"Saved run id: {run_id}")


if __name__ == "__main__":
    main()
