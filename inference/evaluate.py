import json
import re
import torch
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from training.sft import Model

TOOL_CALLING_TESTS = [
    {
        "input": "What is the weather in Almaty?",
        "expected_tool": "get_weather",
        "expected_args": {"city": "Almaty"},
    },
    {
        "input": "Convert 100 USD to EUR",
        "expected_tool": "convert_currency",
        "expected_args": {"amount": 100, "from_currency": "USD", "to_currency": "EUR"},
    },
    {
        "input": "Calculate sqrt(256) + 10^2",
        "expected_tool": "calculate",
        "expected_args": {"expression": "sqrt(256) + 10**2"},
    },
    {
        "input": "Search for latest news about DeepSeek",
        "expected_tool": "search_web",
        "expected_args": {"query": "DeepSeek latest news"},
    },
    {
        "input": "Analyze numbers [4, 8, 15, 16, 23, 42] find mean and outliers",
        "expected_tool": "statistics_analysis",
        "expected_args": {"numbers": [4, 8, 15, 16, 23, 42], "operations": ["mean", "outliers"]},
    },
]

REASONING_TESTS = [
    {
        "input": "What is the sum of angles in a triangle?",
        "expected_answer": "180 degrees",
    },
    {
        "input": "If a train travels 60 km/h for 2 hours, how far does it go?",
        "expected_answer": "120 km",
    },
    {
        "input": "What is the capital of France?",
        "expected_answer": "Paris",
    },
    {
        "input": "What is 15% of 200?",
        "expected_answer": "30",
    },
]


def generate(model, tokenizer, prompt: str, tools=None) -> str:
    messages = [
        {"role": "system", "content": "You are a helpful assistant with access to tools. Use tools when needed."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to("cuda")

    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=256)

    generated = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated)]
    return tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()


def parse_tool_call(response: str) -> dict | None:
    try:
        return json.loads(response.strip().split('\n')[0])
    except json.JSONDecodeError:
        match = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass
    return None


def args_match(expected: dict, actual: dict) -> float:
    if not actual:
        return 0.0
    matches = sum(
        1 for k, v in expected.items()
        if str(actual.get(k, "")).lower() == str(v).lower()
    )
    return matches / len(expected)


def rouge_score(prediction: str, reference: str) -> dict:
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = scorer.score(reference.lower(), prediction.lower())
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }


def bleu_score(prediction: str, reference: str) -> float:
    ref_tokens = reference.lower().split()
    pred_tokens = prediction.lower().split()
    smoothie = SmoothingFunction().method1
    return round(sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie), 4)


def evaluate_tool_calling(model, tokenizer, tools_schema):
    print("TOOL CALLING EVALUATION")

    results = []
    for test in TOOL_CALLING_TESTS:
        response = generate(model, tokenizer, test["input"], tools=tools_schema)
        tool_call = parse_tool_call(response)

        called_tool = tool_call.get("name") if tool_call else None
        called_args = tool_call.get("arguments", {}) if tool_call else {}

        tool_correct = called_tool == test["expected_tool"]
        arg_score = args_match(test["expected_args"], called_args) if tool_correct else 0.0

        results.append({
            "input": test["input"],
            "expected_tool": test["expected_tool"],
            "called_tool": called_tool,
            "tool_correct": tool_correct,
            "arg_score": arg_score,
        })

        status = "ok" if tool_correct else "approximately"
        print(f"{status} [{test['expected_tool']}] → called: {called_tool} | args: {arg_score:.0%} match")

    tool_accuracy = sum(r["tool_correct"] for r in results) / len(results)
    avg_arg_score = sum(r["arg_score"] for r in results) / len(results)

    print(f"\nTool Accuracy:     {tool_accuracy:.0%} ({sum(r['tool_correct'] for r in results)}/{len(results)})")
    print(f"Avg Arg Match:     {avg_arg_score:.0%}")

    return {"tool_accuracy": tool_accuracy, "avg_arg_score": avg_arg_score}


def evaluate_reasoning(model, tokenizer):
    print("REASONING EVALUATION")

    results = []
    for test in REASONING_TESTS:
        response = generate(model, tokenizer, test["input"])
        expected = test["expected_answer"]

        exact = expected.lower() in response.lower()
        rouge = rouge_score(response, expected)
        bleu = bleu_score(response, expected)

        results.append({
            "input": test["input"],
            "expected": expected,
            "response": response,
            "exact_match": exact,
            "rouge1": rouge["rouge1"],
            "rougeL": rouge["rougeL"],
            "bleu": bleu,
        })

        status = "ok" if exact else "approximately"
        print(f"{status} Q: {test['input'][:50]}")
        print(f"  expected: {expected}")
        print(f"  got:      {response[:80]}")
        print(f"  ROUGE-1: {rouge['rouge1']} | ROUGE-L: {rouge['rougeL']} | BLEU: {bleu}")

    exact_match = sum(r["exact_match"] for r in results) / len(results)
    avg_rouge1 = sum(r["rouge1"] for r in results) / len(results)
    avg_rougeL = sum(r["rougeL"] for r in results) / len(results)
    avg_bleu = sum(r["bleu"] for r in results) / len(results)

    print(f"\nExact Match:  {exact_match:.0%}")
    print(f"Avg ROUGE-1:  {avg_rouge1:.4f}")
    print(f"Avg ROUGE-L:  {avg_rougeL:.4f}")
    print(f"Avg BLEU:     {avg_bleu:.4f}")

    return {"exact_match": exact_match, "rouge1": avg_rouge1, "rougeL": avg_rougeL, "bleu": avg_bleu}


if __name__ == "__main__":
    from tools.registry import tools as tools_schema

    tc = Model()
    model = tc.get_feature("model")
    tokenizer = tc.get_feature("tokenizer")

    model.eval()

    tool_metrics = evaluate_tool_calling(model, tokenizer, tools_schema)
    reasoning_metrics = evaluate_reasoning(model, tokenizer)

    print("\n")
    print("SUMMARY")
    print(f"Tool Accuracy:  {tool_metrics['tool_accuracy']:.0%}")
    print(f"Arg Match:      {tool_metrics['avg_arg_score']:.0%}")
    print(f"Exact Match:    {reasoning_metrics['exact_match']:.0%}")
    print(f"ROUGE-L:        {reasoning_metrics['rougeL']:.4f}")
    print(f"BLEU:           {reasoning_metrics['bleu']:.4f}")

    with open("eval_results.json", "w") as f:
        json.dump({**tool_metrics, **reasoning_metrics}, f, indent=2)
    print("\nSaved to eval_results.json")