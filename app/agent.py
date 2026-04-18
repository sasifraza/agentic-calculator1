import anthropic
import math
import json
from dotenv import load_dotenv  

load_dotenv(override=False)

TOOLS = [
    {
        "name": "calculator",
        "description": "Evaluate a math expression. Supports +, -, *, /, ** (power), sqrt(), etc.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A valid Python math expression e.g. '2 ** 10'"
                }
            },
            "required": ["expression"]
        }
    }
]

def run_calculator(expression: str) -> str:
    try:
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed["abs"] = abs
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

def run_agent(task: str) -> dict:
    client = anthropic.Anthropic()
    messages = [{"role": "user", "content": task}]
    steps = []  # track every tool call for logging

    while True:
        response = client.messages.create(
            model="claude-opus-4-5",
            max_tokens=1024,
            tools=TOOLS,
            messages=messages
        )

        messages.append({"role": "assistant", "content": response.content})

        # Claude finished — return answer + steps
        if response.stop_reason == "end_turn":
            final = next((b.text for b in response.content if hasattr(b, "text")), "Done.")
            return {"answer": final, "steps": steps}

        # Claude calls a tool
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = run_calculator(block.input["expression"])
                steps.append({
                    "tool": "calculator",
                    "expression": block.input["expression"],
                    "result": result
                })
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })

        messages.append({"role": "user", "content": tool_results})