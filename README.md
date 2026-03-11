# Qwen2.5-3B Reasoning & Tool Calling
A fine-tuned version of Qwen2.5-3B-Instruct trained on tool calling and reasoning datasets. The model knows when to call a tool instead of making things up, and can think through complex problems step by step.

## Eval Results
Trained for ~6000 steps on Function Calling(112k samples) + reasoning dataset(17k samples)

| Metric | Score | What it means |
|---|---|---|
| Tool Accuracy | 80% | Correctly picks the right tool |
| Arg Match | 40% | Gets the arguments right |
| Exact Match | 100% | Answers factual questions correctly |
| ROUGE-L | 0.5269 | Response quality vs reference |
| BLEU | 0.0639 | N-gram overlap with expected output |

## Tools
The model has access to 7 tools:

| Tool | What it does                  |
|---|-------------------------------|
| `get_weather` | Current weather               |
| `convert_currency` | Live exchange rates between any currencies |
| `calculate` | Safe math expression evaluator|
| `statistics_analysis` | Mean, median, stdev, outliers, primes on a list of numbers |
| `search_web` | DuckDuckGo search             |
| `scrape_url` | Extracts text, links or title from a webpage |
| `wikipedia_summary` | Wikipedia summary for any topic |

## Reasoning Mode
The model uses `<|begin_of_thought|>` / `<|begin_of_solution|>` tags to think through harder problems before answering. It won't use reasoning for simple questions — only kicks in when the task actually needs it.

## Training
- **Base model**: Qwen/Qwen2.5-3B-Instruct
- **Method**: QLoRA (4-bit, nf4, r=16)
- **Tool calling dataset**: [Glaive Function Calling v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2)
- **Reasoning dataset**: [Sky-T1](https://huggingface.co/datasets/NovaSky-AI/Sky-T1_data_17k)
- **Hardware**: NVIDIA RTX 3070 8GB

<img width="918" height="586" alt="image" src="https://github.com/user-attachments/assets/9ece2884-f979-4582-88ad-83f274330e07" />
<img width="844" height="509" alt="image" src="https://github.com/user-attachments/assets/00d2fdcd-9740-46b5-ab47-cd074cb489fe" />
