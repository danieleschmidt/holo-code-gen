# holo-code-gen

Generate code from structured specifications. No LLMs. No external dependencies. Pure Python.

Define *what* a function, class, API endpoint, data model, or test suite should do — holo-code-gen generates a correct, annotated stub ready for implementation.

---

## How It Works

```
CodeSpec  →  TemplateEngine  →  generated code  →  SpecValidator
   ↑                                                      ↓
(you write this)                               (validates the output)
```

1. **`CodeSpec`** — structured description of what you want to build
2. **`TemplateEngine`** — selects the right template (5 types × 3 languages)
3. **`HoloCodeGenerator`** — orchestrates the full pipeline
4. **`SpecValidator`** — uses `ast` to verify the generated code matches the spec

---

## Quick Start

```python
from holo_code_gen import CodeSpec, InputParam, OutputParam, HoloCodeGenerator

gen = HoloCodeGenerator()

spec = CodeSpec(
    name="calculate_tax",
    template="function",
    language="python",
    description="Calculate sales tax for a given amount and rate.",
    inputs=[
        InputParam("amount", "float", "Pre-tax amount in dollars"),
        InputParam("rate", "float", "Tax rate as a decimal (e.g., 0.08)"),
    ],
    output=OutputParam("float", "Tax amount"),
    constraints=["amount must be >= 0", "rate must be between 0 and 1"],
)

result = gen.generate(spec)
print(result.code)
```

Output:
```python
def calculate_tax(amount: float, rate: float) -> float:
    """
    Calculate sales tax for a given amount and rate.

    Args:
        amount (float) - Pre-tax amount in dollars
        rate (float) - Tax rate as a decimal (e.g., 0.08)

    Returns:
        float: Tax amount

    Constraints:
        - amount must be >= 0
        - rate must be between 0 and 1
    """
    return 0
```

---

## Templates

| Template        | Description                              | Python | JavaScript | TypeScript |
|-----------------|------------------------------------------|--------|------------|------------|
| `function`      | Standalone function with full docstring  | ✅     | ✅         | ✅         |
| `class`         | Class with `__init__` and method stubs   | ✅     | ✅         | ✅         |
| `rest_endpoint` | FastAPI (Python) / Express (JS/TS) route | ✅     | ✅         | ✅         |
| `data_model`    | Dataclass (Python) / Interface (TS)      | ✅     | ✅         | ✅         |
| `test_suite`    | pytest (Python) / Jest (JS/TS) tests     | ✅     | ✅         | ✅         |

---

## CodeSpec Fields

| Field          | Type                | Required | Description                                         |
|----------------|---------------------|----------|-----------------------------------------------------|
| `name`         | `str`               | ✅       | Identifier (alphanumeric/underscore)                |
| `template`     | `str`               | ✅       | One of the 5 template types above                   |
| `language`     | `str`               | ✅       | `python`, `javascript`, or `typescript`             |
| `description`  | `str`               | ✅       | What this code unit does                            |
| `inputs`       | `List[InputParam]`  |          | Input parameters with name, type, description       |
| `output`       | `OutputParam`       |          | Return value type and description                   |
| `constraints`  | `List[str]`         |          | Behavioral constraints (documented in docstring)    |
| `examples`     | `List[Example]`     |          | Concrete input/output pairs (used in tests/docs)    |
| `dependencies` | `List[str]`         |          | Required imports (checked by validator)             |
| `tags`         | `List[str]`         |          | Metadata tags                                       |

---

## Validation

The `SpecValidator` parses generated Python code with the `ast` module and checks:

- Function/class definition exists with the expected name
- Parameter list matches the spec inputs
- Return type annotation matches spec output (warning if different)
- `@dataclass` decorator present for data models
- All declared dependencies are imported
- Test suites have at least one `test_*` function

For JavaScript/TypeScript, structural text checks are used.

Errors are level `"error"` (hard failures) or `"warning"` (soft mismatches).

---

## From Dict / JSON

```python
import json
from holo_code_gen import HoloCodeGenerator

gen = HoloCodeGenerator()

data = {
    "name": "parse_date",
    "template": "function",
    "language": "python",
    "description": "Parse an ISO 8601 date string into a date object.",
    "inputs": [{"name": "date_str", "type": "str", "description": "ISO 8601 date string"}],
    "output": {"type": "datetime.date", "description": "Parsed date"},
}

result = gen.generate_from_dict(data)
result = gen.generate_from_json(json.dumps(data))
```

---

## Running the Demo

```bash
python3 examples/demo.py
```

Generates 5 different code units: a Python function, TypeScript class, Python REST endpoint, TypeScript data model, and Python test suite.

---

## Running Tests

```bash
python3 -m pytest tests/ -v
```

60 tests. No external dependencies required.

---

## Project Structure

```
holo_code_gen/
├── __init__.py       # Public API
├── spec.py           # CodeSpec, InputParam, OutputParam, Example
├── templates.py      # TemplateEngine — 5 templates × 3 languages
├── validator.py      # SpecValidator — ast-based validation
└── generator.py      # HoloCodeGenerator — full pipeline orchestration

tests/
├── test_spec.py
├── test_templates.py
├── test_validator.py
└── test_generator.py

examples/
└── demo.py           # Generates 5 code units end-to-end
```

---

## License

MIT
