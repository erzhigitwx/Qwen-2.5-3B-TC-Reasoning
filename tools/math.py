import math
import statistics
import ast
import operator


SAFE_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.USub: operator.neg,
    ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv,
}

def _safe_eval(node):
    if isinstance(node, ast.Constant):
        return node.n
    elif isinstance(node, ast.BinOp):
        return SAFE_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    elif isinstance(node, ast.UnaryOp):
        return SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    elif isinstance(node, ast.Call):
        fn_map = {
            "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "abs": abs, "round": round, "ceil": math.ceil, "floor": math.floor,
        }
        fn_name = node.func.id
        if fn_name in fn_map:
            args = [_safe_eval(a) for a in node.args]
            return fn_map[fn_name](*args)
    raise ValueError(f"Unsupported expression")


def calculate(expression: str) -> dict:
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"expression": expression, "error": str(e)}


def statistics_analysis(numbers: list, operations: list) -> dict:
    op_map = {
        "mean": lambda x: statistics.mean(x),
        "median": lambda x: statistics.median(x),
        "stdev": lambda x: statistics.stdev(x) if len(x) > 1 else 0,
        "variance": lambda x: statistics.variance(x) if len(x) > 1 else 0,
        "min": min, "max": max, "sum": sum,
        "normalize": lambda x: [(i - min(x)) / (max(x) - min(x)) if max(x) != min(x) else 0 for i in x],
        "outliers": lambda x: [i for i in x if abs(i - statistics.mean(x)) > 2 * (statistics.stdev(x) if len(x) > 1 else 1)],
        "primes": lambda x: [i for i in x if i > 1 and all(i % j != 0 for j in range(2, int(math.sqrt(i)) + 1))],
    }
    return {op: op_map[op](numbers) for op in operations if op in op_map}