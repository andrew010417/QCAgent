from agents import Runner, RunConfig, trace, TResponseInputItem
from agents_definition import classify, data_classifer, report_agent, get_qc_agent, should_run_report, WorkflowInput


async def run_workflow(workflow_input: WorkflowInput):
    with trace("New agent"):
        workflow = workflow_input.model_dump()
        conversation_history: list[TResponseInputItem] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": workflow["input_as_text"]
                    }
                ]
            }
        ]

        data_classifer_result_temp = await Runner.run(
            data_classifer,
            input=[*conversation_history],
            run_config=RunConfig(trace_metadata={
                "__trace_source__": "agent-builder",
                "workflow_id": "wf_6a50f034871481909b98a72891ab270e066080d6bd50e43b"
            })
        )

        conversation_history.extend([item.to_input_item() for item in data_classifer_result_temp.new_items])

        classify_input = workflow["input_as_text"]
        classify_result_temp = await Runner.run(
            classify,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": f"{classify_input}"
                        }
                    ]
                }
            ],
            run_config=RunConfig(trace_metadata={
                "__trace_source__": "agent-builder",
                "workflow_id": "wf_6a50f034871481909b98a72891ab270e066080d6bd50e43b"
            })
        )

        classify_result = {
            "output_text": classify_result_temp.final_output_as(str),
            "output_parsed": classify_result_temp.final_output.model_dump()
        }
        classify_category = classify_result["output_parsed"]["category"]

        qc_agent = get_qc_agent(classify_category)
        qc_agent_result_temp = await Runner.run(
            qc_agent,
            input=[*conversation_history],
            run_config=RunConfig(trace_metadata={
                "__trace_source__": "agent-builder",
                "workflow_id": "wf_6a50f034871481909b98a72891ab270e066080d6bd50e43b"
            })
        )

        conversation_history.extend([item.to_input_item() for item in qc_agent_result_temp.new_items])

        qc_agent_result = {
            "output_text": qc_agent_result_temp.final_output_as(str)
        }

        report_agent_result = None
        if should_run_report(classify_category):
            report_agent_result_temp = await Runner.run(
                report_agent,
                input=[*conversation_history],
                run_config=RunConfig(trace_metadata={
                    "__trace_source__": "agent-builder",
                    "workflow_id": "wf_6a50f034871481909b98a72891ab270e066080d6bd50e43b"
                })
            )
            conversation_history.extend([item.to_input_item() for item in report_agent_result_temp.new_items])
            report_agent_result = {
                "output_text": report_agent_result_temp.final_output_as(str)
            }

        return {
            "classification": classify_result,
            "qc_result": qc_agent_result,
            "report_result": report_agent_result
        }
