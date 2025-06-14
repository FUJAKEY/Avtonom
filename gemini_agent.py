import os
import subprocess
import json
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "env"))
import google.generativeai as genai
from google.generativeai import protos

from dotenv import load_dotenv

WORKSPACE_DIR = os.path.abspath("./")

load_dotenv()
API_KEY = os.getenv("API")

if not API_KEY:
    raise RuntimeError("API key not found in .env")

genai.configure(api_key=API_KEY)

# Planning state
plan_steps = []

# Tool functions

def read_file(path: str) -> str:
    abs_path = os.path.abspath(os.path.join(WORKSPACE_DIR, path))
    if not abs_path.startswith(WORKSPACE_DIR):
        raise ValueError("Access outside workspace is not allowed")
    with open(abs_path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(path: str, content: str) -> str:
    abs_path = os.path.abspath(os.path.join(WORKSPACE_DIR, path))
    if not abs_path.startswith(WORKSPACE_DIR):
        raise ValueError("Access outside workspace is not allowed")
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    with open(abs_path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Wrote {len(content)} bytes to {path}"


def run_command(command: str) -> str:
    result = subprocess.run(command, shell=True, cwd=WORKSPACE_DIR,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True)
    return result.stdout


def create_plan(steps: str) -> str:
    """Register a plan consisting of a JSON list of steps."""
    global plan_steps
    plan_steps = json.loads(steps)
    return f"Plan with {len(plan_steps)} steps saved."


def show_plan() -> str:
    return json.dumps(plan_steps, indent=2, ensure_ascii=False)


def execute_plan() -> str:
    outputs = []
    for step in plan_steps:
        fn = step.get("function")
        args = step.get("args", {})
        tool = FUNCTION_MAP.get(fn)
        if not tool:
            outputs.append(f"Unknown function {fn}")
            continue
        outputs.append(tool(**args))
    return "\n".join(outputs)


TOOLS = [read_file, write_file, run_command, create_plan, show_plan, execute_plan]

FUNCTION_MAP = {
    "read_file": read_file,
    "write_file": write_file,
    "run_command": run_command,
    "create_plan": create_plan,
    "show_plan": show_plan,
    "execute_plan": execute_plan,
}

SYSTEM_PROMPT = (
    "Ты автономный агент, работающий в каталоге /workspace/Avtonom. "
    "Ты умеешь читать и изменять файлы, выполнять команды и строить планы. "
    "Когда задача требует нескольких действий, предлагай план через функцию "
    "create_plan, жди подтверждения пользователя и затем выполняй его целиком "
    "с помощью execute_plan."
)

model = genai.GenerativeModel("gemini-2.0-flash", tools=TOOLS)
chat = model.start_chat(history=[{"role": "system", "parts": [SYSTEM_PROMPT]}])

while True:
    try:
        user_msg = input(">>> ")
    except EOFError:
        break
    response = chat.send_message(user_msg)
    while True:
        acted = False
        for part in response.candidates[0].content.parts:
            if hasattr(part, "function_call") and part.function_call:
                acted = True
                fn_name = part.function_call.name
                args = {}
                if part.function_call.args:
                    try:
                        args = json.loads(part.function_call.args)
                    except json.JSONDecodeError:
                        args = {"steps": part.function_call.args}
                result = FUNCTION_MAP[fn_name](**args)
                response = chat.send_message(
                    protos.FunctionResponse(name=fn_name, response={"result": result})
                )
                break
        if not acted:
            print(response.text)
            break

