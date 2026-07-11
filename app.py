import asyncio
from db import init_db, save_workflow_result
from workflow import run_workflow
from agents_definition import WorkflowInput


def main():
    init_db()

    input_text = input("Enter QC workflow text: ")
    workflow_input = WorkflowInput(input_as_text=input_text)

    result = asyncio.run(run_workflow(workflow_input))
    print("\n--- Workflow Result ---")
    print(result)

    run_id = save_workflow_result(
        input_text=input_text,
        category=result["classification"]["output_parsed"]["category"],
        qc_output=result["qc_result"]["output_text"],
        report_output=result["report_result"]["output_text"] if result["report_result"] else None,
    )
    print(f"Saved run id: {run_id}")


if __name__ == "__main__":
    main()
