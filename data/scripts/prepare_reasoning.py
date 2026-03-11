import pandas as pd
import json
from pathlib import Path

path = "../raw/Sky-T1_data_17k.json"
df = pd.read_json(path)

print(f"Всего примеров: {len(df)}")
print("\nКолонки:", df.columns.tolist())

print("\n--- Пример system ---")
print(df['system'][0])
print("\n--- Пример chat ---")
print(df['conversations'][0])

converted = []
errors = 0

for idx, row in df.iterrows():
    try:
        ROLE_MAP = {"human": "user", "gpt": "assistant", "user": "user", "assistant": "assistant"}

        converted.append({
            "system": row["system"],
            "messages": [
                {"role": ROLE_MAP.get(item["from"], item["from"]), "content": item["value"]}
                for item in row["conversations"]
            ],
        })
    except Exception as e:
        print(e)
        errors += 1

output_path = Path("../processed/reasoning.json")
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(converted, f, ensure_ascii=False, indent=2)

print(f"Processed and saved successfully | errors: {errors} | processed: {len(converted)}")
