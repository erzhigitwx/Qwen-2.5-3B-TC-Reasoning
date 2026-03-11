from tools.exchange import convert_currency
from tools.math import calculate, statistics_analysis
from tools.search import search_web
from tools.weather import get_weather
from tools.web import scrape_url
from tools.wikipedia import wikipedia_summary

tool_map = {
    "scrape_url": lambda **kwargs: scrape_url(**kwargs),
    "calculate": lambda **kwargs: calculate(**kwargs),
    "statistics_analysis": lambda **kwargs: statistics_analysis(**kwargs),
    "get_weather": lambda **kwargs: get_weather(**kwargs),
    "search_web": lambda **kwargs: search_web(**kwargs),
    "convert_currency": lambda **kwargs: convert_currency(**kwargs),
    "wikipedia_summary": lambda **kwargs: wikipedia_summary(**kwargs),
}

tools = [
    {
        "type": "function",
        "function": {
            "name": "scrape_url",
            "description": "Scrapes content from a webpage. Use when the user provides a URL and wants to extract text, links, or title from it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Full URL to scrape"},
                    "extract": {"type": "string", "enum": ["text", "links", "title"], "description": "What to extract"}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluates a mathematical expression safely. Use for arithmetic, algebra, trigonometry. Do NOT calculate manually.",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression e.g. '2 + 2', 'sqrt(144)', 'sin(3.14/2)'"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "statistics_analysis",
            "description": "Performs statistical analysis on a list of numbers: mean, median, stdev, variance, min, max, sum, normalize, outliers, primes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "numbers": {"type": "array", "items": {"type": "number"}, "description": "List of numbers"},
                    "operations": {
                        "type": "array",
                        "items": {"type": "string", "enum": ["mean", "median", "stdev", "variance", "min", "max", "sum", "normalize", "outliers", "primes"]},
                        "description": "Operations to perform"
                    }
                },
                "required": ["numbers", "operations"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Gets current weather for any city. Use when user asks about weather, temperature, wind, humidity.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name e.g. 'Almaty', 'London'"},
                    "units": {"type": "string", "enum": ["metric", "imperial"], "description": "metric=Celsius, imperial=Fahrenheit"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Searches the web using DuckDuckGo. Use when user asks about current events, facts, or anything requiring web search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "count": {"type": "integer", "description": "Number of results (1-10)", "minimum": 1, "maximum": 10}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "convert_currency",
            "description": "Converts an amount from one currency to another using live exchange rates. Use when user asks to convert money.",
            "parameters": {
                "type": "object",
                "properties": {
                    "amount": {"type": "number", "description": "Amount to convert"},
                    "from_currency": {"type": "string", "description": "Source currency code e.g. USD, EUR, KZT"},
                    "to_currency": {"type": "string", "description": "Target currency code e.g. USD, EUR, KZT"}
                },
                "required": ["amount", "from_currency", "to_currency"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wikipedia_summary",
            "description": "Gets a summary of any topic from Wikipedia. Use when user asks to explain or describe something.",
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "Topic to look up e.g. 'quantum computing', 'Almaty'"},
                    "sentences": {"type": "integer", "description": "Number of sentences to return", "minimum": 1, "maximum": 10},
                    "lang": {"type": "string", "description": "Language code: 'en', 'ru', 'kk'"}
                },
                "required": ["topic"]
            }
        }
    }
]