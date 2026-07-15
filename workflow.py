from agents import Runner, RunConfig, trace, TResponseInputItem
from agents_definition import (
    classify,
    data_classifer,
    report_agent,
    get_qc_agent,
    get_recommended_tools,
    should_run_report,
    QCReportSchema,
    WorkflowInput,
)


def build_conversation(workflow_input: WorkflowInput, experiment_text: str | None = None) -> list[dict]:
    conversation: list[dict] = [
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": workflow_input.input_as_text,
                }
            ],
        }
    ]
    if experiment_text:
        conversation.append(
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": experiment_text,
                    }
                ],
            }
        )
    return conversation


async def prepare_workflow(workflow_input: WorkflowInput):
    with trace("Prepare workflow"):
        conversation_history = build_conversation(workflow_input)

        data_classifer_result_temp = await Runner.run(
            data_classifer,
            input=conversation_history,
            run_config=RunConfig(trace_metadata={
                "__trace_source__": "agent-builder",
                "workflow_id": "wf_prepare"
            }),
        )

        conversation_history.extend([item.to_input_item() for item in data_classifer_result_temp.new_items])

        classify_input = workflow_input.input_as_text
        classify_result_temp = await Runner.run(
            classify,
            input=conversation_history,
            run_config=RunConfig(trace_metadata={
                "__trace_source__": "agent-builder",
                "workflow_id": "wf_prepare"
            }),
        )

        classify_result = {
            "output_text": classify_result_temp.final_output_as(str),
            "output_parsed": classify_result_temp.final_output.model_dump(),
        }
        classify_category = classify_result["output_parsed"]["category"]
        qc_agent = get_qc_agent(classify_category)

        return {
            "classification": classify_result,
            "category": classify_category,
            "qc_agent_name": qc_agent.name,
            "recommended_tools": get_recommended_tools(classify_category),
            "conversation_history": conversation_history,
        }


async def evaluate_workflow(workflow_input: WorkflowInput, experiment_text: str, category: str):
    with trace("Evaluate workflow"):
        conversation_history = build_conversation(workflow_input, experiment_text)
        qc_agent = get_qc_agent(category)

        qc_agent_result_temp = await Runner.run(
            qc_agent,
            input=conversation_history,
            run_config=RunConfig(trace_metadata={
                "__trace_source__": "agent-builder",
                "workflow_id": "wf_evaluate"
            }),
        )

        conversation_history.extend([item.to_input_item() for item in qc_agent_result_temp.new_items])

        report_agent_result = None
        if should_run_report(category):
            report_agent_result_temp = await Runner.run(
                report_agent,
                input=conversation_history,
                run_config=RunConfig(trace_metadata={
                    "__trace_source__": "agent-builder",
                    "workflow_id": "wf_evaluate"
                }),
            )
            conversation_history.extend([item.to_input_item() for item in report_agent_result_temp.new_items])
            final_output = report_agent_result_temp.final_output
            report_agent_result = {
                "output_text": report_agent_result_temp.final_output_as(str),
                "metrics": final_output.metrics if isinstance(final_output, QCReportSchema) else [],
            }

        return {
            "qc_result": {
                "output_text": qc_agent_result_temp.final_output_as(str)
            },
            "report_result": report_agent_result,
        }
