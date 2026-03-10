from training.sft import TCModel
import json
import re
import requests


def get_latest_news(category: str = "technology", count: int = 3) -> list:
    top_stories_url = "https://hacker-news.firebaseio.com/v0/topstories.json"

    story_ids = requests.get(top_stories_url).json()[:count]

    results = []
    for story_id in story_ids:
        story = requests.get(f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json").json()
        results.append({
            "title": story.get("title"),
            "url": story.get("url", "no url"),
            "score": story.get("score"),
            "by": story.get("by")
        })

    return results


tool_map = {
    "get_latest_news": lambda **kwargs: get_latest_news(**kwargs)
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_latest_news",
            "description": "Retrieves the latest news headlines and summaries by category. Use this tool whenever the user asks about news, current events, recent developments, or what is happening in the world. Do NOT make up news — always call this function to get real headlines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["tech", "science", "general"],
                        "description": "News category: 'tech' for technology, 'science' for science, 'general' for world news"
                    },
                    "count": {
                        "type": "integer",
                        "description": "Number of news items to return (1-4)",
                        "minimum": 1,
                        "maximum": 4
                    }
                },
                "required": ["category"]
            }
        }
    }
]


class Chat:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate(self, tools=None):
        messages = [
            {"role": "system", "content": """You are a function-calling assistant. You have access to tools listed below.

            STRICT RULES:
            - You MUST respond with a raw JSON tool call when the user's request can be handled by a tool
            - Format: {"name": "tool_name", "arguments": {...}}
            - Do NOT write any text before or after the JSON
            - Do NOT answer from your own knowledge if a tool exists for the task
            - Do NOT say "I will fetch" or "please wait" — just output the JSON immediately"""}
        ]

        while True:
            prompt = input("user: ")
            messages.append({"role": "user", "content": prompt})

            text = self.tokenizer.apply_chat_template(
                messages,
                tools=tools,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = self.tokenizer([text], return_tensors="pt").to("cuda")
            generated_ids = self.model.generate(**model_inputs, max_new_tokens=512)
            generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in
                             zip(model_inputs.input_ids, generated_ids)]
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            print(f"raw: {response}")

            tool_call = None
            try:
                first_line = response.strip().split('\n')[0]
                tool_call = json.loads(first_line)
            except json.JSONDecodeError:
                tool_match = re.search(r'<tool_call>\s*(.*?)\s*</tool_call>', response, re.DOTALL)
                if tool_match:
                    try:
                        tool_call = json.loads(tool_match.group(1))
                    except json.JSONDecodeError:
                        pass

            if tool_call and tool_call.get("name") in tool_map:
                args = tool_call.get("arguments", {})
                result = tool_map[tool_call["name"]](**args)
                messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})
                print(f"[tool called: {tool_call['name']}] → {result}")
                continue

            messages.append({"role": "assistant", "content": response})
            print(f"assistant: {response}")


tc = TCModel()
tokenizer = tc.get_feature("tokenizer")
chat = Chat(tc.get_feature("model"), tokenizer)
chat.generate(tools)
