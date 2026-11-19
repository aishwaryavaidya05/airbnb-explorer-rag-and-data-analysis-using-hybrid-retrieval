import ast, builtins, re
from functools import lru_cache
import numpy as np, pandas as pd
from llm import llm, BIG, FAST

@lru_cache(1)
def load():
    return pd.read_parquet("data/listings.parquet"), pd.read_parquet("data/reviews.parquet")

TEXT_COLS = {"name", "comments"}

def describe(name, d):
    out = [f"DataFrame `{name}` ({len(d):,} rows)"]
    for c in d.columns:
        s = d[c]
        if c in TEXT_COLS:
            info = "free text"
        elif pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            info = f"min {s.min():.2f}, max {s.max():.2f}, mean {s.mean():.2f}, nulls {s.isna().sum()}"
        elif pd.api.types.is_datetime64_any_dtype(s):
            info = f"{s.min().date()} to {s.max().date()}"
        else:
            info = f"{s.nunique()} distinct; most common: {list(s.value_counts().index[:12])}"
        out.append(f"- {c} [{s.dtype}]: {info}")
    sample = d.drop(columns=[c for c in TEXT_COLS if c in d.columns]).head(3).to_string()
    out.append("Sample rows:\n" + sample)
    return "\n".join(out)

@lru_cache(1)
def schema():
    L, R = load()
    return (describe("listings", L) + "\n\n" + describe("reviews", R) +
            "\n\nNote: `reviews` is a random SAMPLE of reviews, so review counts are not totals.")

CODE_SYSTEM = """You write pandas code to answer questions about Airbnb data in Edinburgh.
You have two DataFrames already loaded: `listings` and `reviews`, plus `pd` and `np`.
Rules:
- No imports, no file or network access. Use only listings, reviews, pd, np.
- Put the final answer in a variable named `result` (scalar, Series, or DataFrame). Keep it under 30 rows and round floats to 2 decimals.
- For name filters use case-insensitive str.contains. Use category values exactly as shown in the schema.
- Prices are nightly, in local currency.
- Reply with ONE ```python code block and nothing else.

Example:
Q: Average price by room type
````python
result = listings.groupby('room_type')['price'].mean().round(2).sort_values(ascending=False)
```"""

BANNED_NAMES = {"eval", "exec", "compile", "open", "getattr", "setattr", "delattr", "globals",
                "locals", "vars", "input", "breakpoint", "os", "sys", "subprocess"}
BANNED_ATTRS = {"eval", "system", "popen", "to_csv", "to_pickle", "to_parquet", "to_sql", "to_excel",
                "to_json", "to_hdf", "to_feather", "to_clipboard", "to_html", "to_latex", "to_xml", "to_stata"}
SAFE = {k: getattr(builtins, k) for k in
        ["len", "range", "sum", "min", "max", "sorted", "round", "abs", "enumerate", "zip", "list",
         "dict", "set", "tuple", "str", "int", "float", "bool", "any", "all", "map", "filter",
         "reversed", "isinstance"]}

def validate(code):
    tree = ast.parse(code)
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            raise ValueError("imports are not allowed")
        if isinstance(n, ast.Name) and (n.id in BANNED_NAMES or n.id.startswith("__")):
            raise ValueError(f"name '{n.id}' is not allowed")
        if isinstance(n, ast.Attribute) and (n.attr.startswith("_") or n.attr in BANNED_ATTRS
                                             or n.attr.startswith("read_")):
            raise ValueError(f"attribute '{n.attr}' is not allowed")
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and "__" in n.value:
            raise ValueError("suspicious string")
    return tree

def run(code, L, R):
    tree = validate(code)
    env = {"__builtins__": SAFE, "pd": pd, "np": np, "listings": L.copy(), "reviews": R.copy()}
    exec(compile(tree, "<generated>", "exec"), env)
    if "result" not in env:
        raise ValueError("code did not set `result`")
    return env["result"]

def _extract(txt):
    m = re.search(r"```(?:python)?\s*(.*?)```", txt, re.S)
    return (m.group(1) if m else txt).strip()

def to_frame(res):
    if isinstance(res, pd.Series):
        res = res.rename(res.name or "value").reset_index()
    elif isinstance(res, pd.DataFrame):
        if not isinstance(res.index, pd.RangeIndex):
            res = res.reset_index()
    else:
        return None
    for c in res.columns:
        if str(res[c].dtype).startswith("period"):
            res[c] = res[c].astype(str)
    return res.head(200)

def answer_analysis(question):
    L, R = load()
    msgs = [{"role": "system", "content": CODE_SYSTEM + "\n\n" + schema()},
            {"role": "user", "content": question}]
    code, err, raw = "", None, None
    for _ in range(2):  # one retry with the error fed back
        code = _extract(llm(msgs, BIG))
        try:
            raw = run(code, L, R)
            err = None
            break
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            msgs += [{"role": "assistant", "content": f"```python\n{code}\n```"},
                     {"role": "user", "content": f"That failed with: {err}. Fix the code."}]
    if err:
        return {"content": f"I couldn't compute that ({err}). Try rephrasing.",
                "code": code, "table": None, "raw": None}
    table = to_frame(raw)
    shown = table.head(30).to_string(index=False) if table is not None else str(raw)
    summary = llm([{"role": "system", "content": "Explain the result in 1-3 sentences using only the numbers shown. Do not invent anything."},
                   {"role": "user", "content": f"Question: {question}\n\nResult:\n{shown}"}], FAST)
    return {"content": summary, "code": code, "table": table, "raw": raw}
