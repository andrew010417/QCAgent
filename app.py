import asyncio
from db import init_db, save_workflow_result
from workflow import run_workflow
from agents_definition import WorkflowInput


def main():
    init_db()

    file_path = input("Enter path to QC workflow file (leave empty to type text manually): ").strip()
    if file_path:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                input_text = f.read().strip()
        except FileNotFoundError:
            print(f"File not found: {file_path}")
            return
        except Exception as exc:
            print(f"Failed to read file: {exc}")
            return

        if not input_text:
            print("The file is empty. Please provide valid QC workflow text.")
            return
    else:
        input_text = input("Enter QC workflow text: ").strip()
        if not input_text:
            print("No input provided. Exiting.")
            return

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
