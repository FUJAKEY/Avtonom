import os
import subprocess
import json
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "env"))
import google.generativeai as genai
from google.generativeai import protos

WORKSPACE_DIR = os.path.abspath("./")

# Load API key from .env
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("API")

if not API_KEY:
    raise RuntimeError("API key not found in .env")

genai.configure(api_key=API_KEY)

# Define tool functions for function calling

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

# Map for function calling
TOOLS = [read_file, write_file, run_command]

# Map tool name to function
FUNCTION_MAP = {
    "read_file": read_file,
    "write_file": write_file,
    "run_command": run_command,
}

SYSTEM_PROMPT = (
    "Ты автономный агент, помогающий создавать проекты. "
    "Работай строго в каталоге /workspace/Avtonom. "
    "У тебя есть функции чтения и записи файлов и выполнения команд. "
    "Следуй задачам пользователя и постепенно формируй нужные файлы."
)

model = genai.GenerativeModel("gemini-2.0-flash", tools=TOOLS)
chat = model.start_chat(history=[{"role": "system", "parts": [SYSTEM_PROMPT]}])

user_task = (
    "Создай минимальный пример проекта на Python со структурой "
    "каталогов и инструкцией по запуску"
)
response = chat.send_message(user_task)

while True:
    acted = False
    for part in response.candidates[0].content.parts:
        if hasattr(part, "function_call") and part.function_call:
            acted = True
            fn_name = part.function_call.name
            args = json.loads(part.function_call.args)
            result = FUNCTION_MAP[fn_name](**args)
            response = chat.send_message(protos.FunctionResponse(name=fn_name, response={"result": result}))
            break
    if not acted:
        print(response.text)
        break
