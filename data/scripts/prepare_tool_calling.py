import pandas as pd
import json
import re
from pathlib import Path

path = "../raw/glaive_function_calling_v2.parquet"
df = pd.read_parquet(path)

print(f"Всего примеров: {len(df)}")
print("\nКолонки:", df.columns.tolist())

print("\n--- Пример system ---")
print(df['system'][0])
print("\n--- Пример chat ---")
print(df['chat'][0])


def parse_system(system_text):
    system_text = system_text.replace("SYSTEM", "").strip()
    json_match = re.search(r'\{[\s\S]*\}', system_text)
    tools = []
    system_clean = "You are a helpful assistant with access to the following functions. Use them if required. "

    if json_match:
        try:
            tool = json.loads(json_match.group())
            tools.append({"type": "function", "function": tool})
        except:
            pass

    return system_clean, tools

def parse_chat(chat_text):
    turns = []

    parts = re.split(r"(USER:|ASSISTANT:|FUNCTION RESPONCE:)", chat_text)
    parts = [p.strip() for p in parts if p.strip()]

    i = 0
    while i < len(parts):
        role_tag = parts[i]
        content = parts[i+1].replace("<|endoftext|>", "").strip() if i+1 < len(parts) else ""
        if role_tag == "USER:":
            turns.append({"role": "user", "content": content})
        elif role_tag == "ASSISTANT:":
            tool_match = re.search(r"<functioncall>([\s\S]*?)(?=\n|$)", content)
            if tool_match:
                try:
                    tool_call = json.loads(tool_match.group(1).strip())
                    turns.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "type": "function",
                            "function": {
                                "name": tool_call.get("name", ""),
                                "arguments": json.dumps(tool_call.get("arguments", {}))
                            }
                        }]
                    })
                except:
                    turns.append({"role": "assistant", content: content})
            else:
                turns.append({"role": "assistant", "content": content})
        elif role_tag == "FUNCTION RESPONCE:":
            turns.append({"role": "tool", "content": content})

        i += 2

    return turns

def convert_row(row):
    system_clean, tools = parse_system(row["system"])
    messages = parse_chat(row["chat"])

    return {
        "system": system_clean,
        "tools": tools,
        "messages": messages
    }

print("Конвертируем датасет...")
converted = []
errors = 0

for idx, row in df.iterrows():
    try:
        converted.append(convert_row(row))
    except Exception as e:
        errors += 1

print(f"Успешно: {len(converted)}, Ошибок: {errors}")

print("\n--- Пример результата ---")
print(json.dumps(converted[0], indent=2, ensure_ascii=False))

output_path = Path("../processed/tool_calling.json")
output_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(converted, f, ensure_ascii=False, indent=2)

print(f"\nСохранено в {output_path}")