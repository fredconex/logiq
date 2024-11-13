"""
author: Alfredo Fernandes
description: Problems -> code, to help solving problems.

Insert the prefix /logiq when you want to enable it for current message

Example:
/logiq How many r's are in the word stratosphere?

Requires: 
Open WebUI 0.3.31 or above as it requires "Expandable Content Markdown Support"
Only compatible with ollama at moment!
"""

from pydantic import BaseModel, Field
from typing import Optional
import re
import json
import time
import requests
from open_webui.utils.misc import get_last_user_message
import ast
import builtins
import contextlib
import math
import random
import sys
from io import StringIO


class Filter:
    class Valves(BaseModel):
        ollama_url: str = Field(
            default="http://host.docker.internal:11434",
            description="URL for the Ollama API.",
        )
        temperature: float = Field(
            default=0.1,
            description="Temperature for token prediction.",
        )
        max_retries: int = Field(
            default=6,
            description="Maximum number of retry attempts for code generation and execution.",
        )
        show_code_run_errors: bool = Field(
            default=False,
            description="Enable/disable code run errors in the output.",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.code_output = None
        self.safe_builtins = {
            "abs",
            "all",
            "any",
            "bool",
            "chr",
            "dict",
            "divmod",
            "enumerate",
            "filter",
            "float",
            "int",
            "len",
            "list",
            "map",
            "max",
            "min",
            "pow",
            "print",
            "range",
            "round",
            "set",
            "sorted",
            "str",
            "sum",
            "zip",
        }
        self.safe_modules = {"math": math, "random": random, "re": re, "time": time}

    def make_api_call(self, system, prompt, model):
        try:
            response = requests.post(
                f"{self.valves.ollama_url}/api/generate",
                json={
                    "model": model,
                    "system": system,
                    "prompt": prompt,
                    "stream": False,
                },
            )
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            return f"Error: Failed to generate response. Error: {str(e)}"

    def generate_code(self, problem, model):
        allowed_builtins = ", ".join(self.safe_builtins)
        allowed_modules = ", ".join(self.safe_modules.keys())

        rule = """Generate compact Python program to solve this problem. Use natural language on output of print function.
        
        Ensure the code is complete and use logic processing to solve the problem, syntactically correct, and doesn't require user input or imports.
        Provide only the Python code within a code block, no explanations."""

        prompt = f"""
<problem>{problem}</problem>
<allowed-built-ins>{allowed_builtins}</allowed-built-ins>
<allowed-modules>{allowed_modules}</allowed-modules>

{rule} """

        return self.make_api_call(
            rule,
            prompt,
            model,
        )

    def extract_code(self, text):
        pattern = r"```python\n(.*?)\n```"
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else text

    def validate_code(self, code):
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def execute_code(self, code):
        if not self.validate_code(code):
            return (
                "Error: Invalid syntax" if self.valves.show_code_run_errors else "Error"
            )

        output = ""
        original_stdout = sys.stdout
        sys.stdout = StringIO()

        safe_builtins = {name: getattr(builtins, name) for name in self.safe_builtins}
        safe_globals = {"__builtins__": safe_builtins, **self.safe_modules}

        try:
            with contextlib.redirect_stdout(sys.stdout):
                exec(code, safe_globals)
            output = sys.stdout.getvalue()
        except Exception as e:
            output = f"Error: {str(e)}" if self.valves.show_code_run_errors else "Error"
        finally:
            sys.stdout = original_stdout

        return output

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        messages = body.get("messages", [])
        model = body.get("model", "llama2")
        self.logiq_output = None

        if messages:
            last_message = get_last_user_message(body["messages"])
            if last_message.startswith("/logiq"):
                last_message = last_message.replace("/logiq", "", 1).strip()
                for attempt in range(self.valves.max_retries):
                    generated_code_raw = self.generate_code(last_message, model)
                    generated_code = self.extract_code(generated_code_raw)
                    code_output = self.execute_code(generated_code)

                    self.logiq_output = f"<details>\n<summary>LogiQ</summary>\n```python\n{generated_code}\n```\n</details>\n{code_output}"

                    if not code_output.startswith("Error"):
                        body["messages"] = [
                            {"role": "user", "content": """Output "processing..\"."""}
                        ]
                        break
                    elif attempt == self.valves.max_retries - 1:
                        body["messages"][-1][
                            "content"
                        ] = "I apologize, but I couldn't generate a valid response after multiple attempts. Could you please rephrase your question or provide more details?"
                        self.logiq_output = None

        return body

    async def outlet(self, body: dict) -> dict:
        if self.logiq_output is not None:
            body["messages"][-1]["content"] = self.logiq_output
        return body
