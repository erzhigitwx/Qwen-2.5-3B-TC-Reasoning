from training.sft import Model
from tools.registry import tool_map, tools
import json
import re
import warnings
warnings.filterwarnings("ignore")

BLUE = "\033[94m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def user_print(text): print(f"{BLUE}user: {text}{RESET}")
def assistant_print(text): print(f"{GREEN}assistant: {text}{RESET}")
def tool_print(name, args, result): print(f"{YELLOW}[tool] {name}({args}) → {result}{RESET}")


class Chat:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def _generate_response(self, messages):
        text = self.tokenizer.apply_chat_template(
            messages, tools=tools, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to("cuda")
        generated_ids = self.model.generate(**model_inputs, max_new_tokens=512)
        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]
        return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    def _parse_tool_call(self, response):
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

    def generate(self):
        messages = [
            {"role": "system", "content": """You are a helpful assistant with access to tools.

RULES:
- ALWAYS use a tool if the request involves: current data, prices, weather, calculations, web content, currency, search, or facts that may change over time
- NEVER invent data — if a tool exists for the task, call it
- After getting tool results, give a clear formatted answer
- Use reasoning (<|begin_of_thought|> and <|begin_of_solution|>) for complex multi-step tasks
- Only answer from your own knowledge if no tool is relevant"""}
        ]

        while True:
            prompt = input(f"{BLUE}user: {RESET}")
            user_print(prompt) if False else None
            messages.append({"role": "user", "content": prompt})

            while True:
                response = self._generate_response(messages)
                tool_call = self._parse_tool_call(response)

                if tool_call and tool_call.get("name") in tool_map:
                    args = tool_call.get("arguments", {})
                    result = tool_map[tool_call["name"]](**args)
                    result_str = json.dumps(result, ensure_ascii=False)
                    tool_print(tool_call["name"], args, result_str)
                    messages.append({"role": "assistant", "content": response})
                    messages.append({"role": "tool", "content": result_str})
                else:
                    messages.append({"role": "assistant", "content": response})
                    assistant_print(response)
                    break


tc = Model()
tokenizer = tc.get_feature("tokenizer")
chat = Chat(tc.get_feature("model"), tokenizer)
chat.generate()