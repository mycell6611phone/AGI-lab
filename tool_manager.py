# tool_manager.py

import re

TOOLS = {}

def register_tool(name, func):
    TOOLS[name] = func

def extract_tool_calls(text):
    # Looks for lines like: CALL: tool_name("args")
    pattern = r'CALL:\s*(\w+)\((.*?)\)'
    return re.findall(pattern, text)

def execute_tool_call(name, arg_str):
    func = TOOLS.get(name)
    if not func:
        return f"[ERROR: Tool '{name}' not registered]"
    # You can parse arg_str further if needed (for now assume a simple string arg)
    return func(arg_str.strip('"'))
