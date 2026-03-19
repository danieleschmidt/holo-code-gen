"""
TemplateEngine: Jinja2-free template system using Python string formatting.

Templates are structured blueprints for each code unit type. The engine
selects the right template, fills in spec fields, and emits clean code stubs.
"""

from __future__ import annotations
from typing import Dict, List, Optional
from .spec import CodeSpec, InputParam


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _indent(text: str, spaces: int = 4) -> str:
    pad = " " * spaces
    return "\n".join(pad + line if line.strip() else line for line in text.split("\n"))


def _py_param_str(params: List[InputParam]) -> str:
    """Build a Python parameter list string from a list of InputParam."""
    parts = []
    for p in params:
        annotation = f": {p.type}" if p.type else ""
        default = f" = {repr(p.default)}" if p.default is not None else ""
        parts.append(f"{p.name}{annotation}{default}")
    return ", ".join(parts)


def _js_param_str(params: List[InputParam]) -> str:
    return ", ".join(p.name for p in params)


def _ts_param_str(params: List[InputParam]) -> str:
    return ", ".join(f"{p.name}: {p.type}" for p in params)


def _docstring_body(spec: CodeSpec) -> str:
    lines = [spec.description]
    if spec.inputs:
        lines.append("")
        lines.append("Args:")
        for p in spec.inputs:
            desc = f" - {p.description}" if p.description else ""
            lines.append(f"    {p.name} ({p.type}){desc}")
    if spec.output:
        lines.append("")
        lines.append("Returns:")
        lines.append(f"    {spec.output.type}: {spec.output.description}")
    if spec.constraints:
        lines.append("")
        lines.append("Constraints:")
        for c in spec.constraints:
            lines.append(f"    - {c}")
    if spec.examples:
        lines.append("")
        lines.append("Examples:")
        for ex in spec.examples:
            lines.append(f"    >>> # {ex.description}")
            lines.append(f"    >>> {spec.name}({', '.join(f'{k}={repr(v)}' for k, v in ex.inputs.items())})")
            lines.append(f"    {repr(ex.expected_output)}")
    return "\n".join(lines)


def _ret_type_py(spec: CodeSpec) -> str:
    return f" -> {spec.output.type}" if spec.output else ""


def _ret_type_ts(spec: CodeSpec) -> str:
    return f": {spec.output.type}" if spec.output else ""


def _return_placeholder(spec: CodeSpec, language: str) -> str:
    """Return a sensible placeholder return statement."""
    if not spec.output:
        return ""
    t = spec.output.type.lower()
    if language == "python":
        if "str" in t:
            return 'return ""'
        if "int" in t or "float" in t:
            return "return 0"
        if "bool" in t:
            return "return False"
        if "list" in t or "List" in t:
            return "return []"
        if "dict" in t or "Dict" in t:
            return "return {}"
        return "raise NotImplementedError"
    else:  # js / ts
        if t in ("string", "str"):
            return 'return "";'
        if t in ("number", "int", "float"):
            return "return 0;"
        if t == "boolean":
            return "return false;"
        if "array" in t or "[]" in t:
            return "return [];"
        if "object" in t or "{" in t:
            return "return {};"
        return "throw new Error('Not implemented');"


# ---------------------------------------------------------------------------
# Template implementations
# ---------------------------------------------------------------------------

class TemplateEngine:
    """
    Selects and renders code templates from a CodeSpec.

    Supported templates:
        function      - standalone function / free function
        class         - class with __init__ and stub methods
        rest_endpoint - FastAPI-style (Python) or Express-style (JS/TS) handler
        data_model    - dataclass (Python) or interface (TS) or plain object (JS)
        test_suite    - pytest (Python) or Jest (JS/TS) test file
    """

    # Registry maps (template, language) -> render method
    def render(self, spec: CodeSpec) -> str:
        """
        Render code from a CodeSpec.
        Returns the generated source code as a string.
        """
        key = (spec.template, spec.language)
        dispatch = {
            ("function", "python"):        self._py_function,
            ("function", "javascript"):    self._js_function,
            ("function", "typescript"):    self._ts_function,
            ("class", "python"):           self._py_class,
            ("class", "javascript"):       self._js_class,
            ("class", "typescript"):       self._ts_class,
            ("rest_endpoint", "python"):   self._py_rest_endpoint,
            ("rest_endpoint", "javascript"): self._js_rest_endpoint,
            ("rest_endpoint", "typescript"): self._ts_rest_endpoint,
            ("data_model", "python"):      self._py_data_model,
            ("data_model", "javascript"):  self._js_data_model,
            ("data_model", "typescript"):  self._ts_data_model,
            ("test_suite", "python"):      self._py_test_suite,
            ("test_suite", "javascript"):  self._js_test_suite,
            ("test_suite", "typescript"):  self._ts_test_suite,
        }
        renderer = dispatch.get(key)
        if renderer is None:
            raise ValueError(f"No template for ({spec.template}, {spec.language})")
        return renderer(spec)

    # ------------------------------------------------------------------
    # FUNCTION templates
    # ------------------------------------------------------------------

    def _py_function(self, spec: CodeSpec) -> str:
        params = _py_param_str(spec.inputs)
        ret = _ret_type_py(spec)
        doc = _docstring_body(spec)
        body = _return_placeholder(spec, "python") or "pass"
        imports = "\n".join(f"import {d}" for d in spec.dependencies)
        header = f"{imports}\n\n" if imports else ""
        return (
            f"{header}"
            f"def {spec.name}({params}){ret}:\n"
            f'    """\n'
            + _indent(doc, 4) + "\n"
            f'    """\n'
            f"    {body}\n"
        )

    def _js_function(self, spec: CodeSpec) -> str:
        params = _js_param_str(spec.inputs)
        body = _return_placeholder(spec, "javascript") or "// TODO: implement"
        doc = _docstring_body(spec)
        jsdoc = "/**\n" + "\n".join(f" * {l}" for l in doc.split("\n")) + "\n */"
        return (
            f"{jsdoc}\n"
            f"function {spec.name}({params}) {{\n"
            f"  {body}\n"
            f"}}\n\n"
            f"module.exports = {{ {spec.name} }};\n"
        )

    def _ts_function(self, spec: CodeSpec) -> str:
        params = _ts_param_str(spec.inputs)
        ret = _ret_type_ts(spec)
        body = _return_placeholder(spec, "typescript") or "// TODO: implement"
        doc = _docstring_body(spec)
        jsdoc = "/**\n" + "\n".join(f" * {l}" for l in doc.split("\n")) + "\n */"
        return (
            f"{jsdoc}\n"
            f"export function {spec.name}({params}){ret} {{\n"
            f"  {body}\n"
            f"}}\n"
        )

    # ------------------------------------------------------------------
    # CLASS templates
    # ------------------------------------------------------------------

    def _py_class(self, spec: CodeSpec) -> str:
        params = _py_param_str(spec.inputs)
        init_assigns = "\n".join(
            f"        self.{p.name} = {p.name}" for p in spec.inputs
        ) or "        pass"
        imports = "\n".join(f"import {d}" for d in spec.dependencies)
        header = f"{imports}\n\n" if imports else ""
        methods = self._py_class_methods(spec)
        return (
            f"{header}"
            f"class {spec.name}:\n"
            f'    """{spec.description}"""\n\n'
            f"    def __init__(self, {params}) -> None:\n"
            f"{init_assigns}\n"
            f"{methods}"
        )

    def _py_class_methods(self, spec: CodeSpec) -> str:
        # Generate method stubs from examples or a generic process method
        if spec.examples:
            methods = []
            for ex in spec.examples:
                mname = ex.description.lower().replace(" ", "_") or "process"
                methods.append(
                    f"\n    def {mname}(self) -> None:\n"
                    f'        """TODO: implement {mname}"""\n'
                    f"        raise NotImplementedError\n"
                )
            return "".join(methods)
        ret = _ret_type_py(spec)
        body = _return_placeholder(spec, "python") or "raise NotImplementedError"
        return (
            f"\n    def process(self){ret}:\n"
            f'        """{spec.description}"""\n'
            f"        {body}\n"
        )

    def _js_class(self, spec: CodeSpec) -> str:
        params = _js_param_str(spec.inputs)
        assigns = "\n".join(f"    this.{p.name} = {p.name};" for p in spec.inputs) or "    // no fields"
        body = _return_placeholder(spec, "javascript") or "// TODO: implement"
        return (
            f"class {spec.name} {{\n"
            f"  /** {spec.description} */\n"
            f"  constructor({params}) {{\n"
            f"{assigns}\n"
            f"  }}\n\n"
            f"  process() {{\n"
            f"    {body}\n"
            f"  }}\n"
            f"}}\n\n"
            f"module.exports = {{ {spec.name} }};\n"
        )

    def _ts_class(self, spec: CodeSpec) -> str:
        params = _ts_param_str(spec.inputs)
        fields = "\n".join(f"  {p.name}: {p.type};" for p in spec.inputs)
        assigns = "\n".join(f"    this.{p.name} = {p.name};" for p in spec.inputs)
        ret = _ret_type_ts(spec)
        body = _return_placeholder(spec, "typescript") or "throw new Error('Not implemented');"
        return (
            f"/** {spec.description} */\n"
            f"export class {spec.name} {{\n"
            f"{fields}\n\n"
            f"  constructor({params}) {{\n"
            f"{assigns}\n"
            f"  }}\n\n"
            f"  process(){ret} {{\n"
            f"    {body}\n"
            f"  }}\n"
            f"}}\n"
        )

    # ------------------------------------------------------------------
    # REST ENDPOINT templates
    # ------------------------------------------------------------------

    def _py_rest_endpoint(self, spec: CodeSpec) -> str:
        """FastAPI-style endpoint."""
        params = _py_param_str(spec.inputs)
        ret = _ret_type_py(spec)
        doc = spec.description
        route = spec.name.replace("_", "-")
        body = _return_placeholder(spec, "python") or '{"status": "ok"}'
        imports = "\n".join(f"import {d}" for d in spec.dependencies)
        base_imports = "from fastapi import FastAPI, HTTPException\nfrom pydantic import BaseModel"
        all_imports = f"{base_imports}\n{imports}" if imports else base_imports
        return (
            f"{all_imports}\n\n"
            f"app = FastAPI()\n\n\n"
            f"@app.post('/{route}')\n"
            f"async def {spec.name}({params}){ret}:\n"
            f'    """{doc}"""\n'
            f"    {body}\n"
        )

    def _js_rest_endpoint(self, spec: CodeSpec) -> str:
        """Express.js-style route handler."""
        route = spec.name.replace("_", "-")
        body = "res.json({ status: 'ok' });"
        return (
            f"const express = require('express');\n"
            f"const router = express.Router();\n\n"
            f"/**\n * {spec.description}\n */\n"
            f"router.post('/{route}', async (req, res) => {{\n"
            f"  try {{\n"
            f"    const {{ {', '.join(p.name for p in spec.inputs)} }} = req.body;\n"
            f"    {body}\n"
            f"  }} catch (err) {{\n"
            f"    res.status(500).json({{ error: err.message }});\n"
            f"  }}\n"
            f"}});\n\n"
            f"module.exports = router;\n"
        )

    def _ts_rest_endpoint(self, spec: CodeSpec) -> str:
        """Express + TypeScript route handler."""
        route = spec.name.replace("_", "-")
        param_destructure = ", ".join(p.name for p in spec.inputs)
        param_types = "; ".join(f"{p.name}: {p.type}" for p in spec.inputs)
        return (
            f"import {{ Router, Request, Response }} from 'express';\n\n"
            f"interface {spec.name}Body {{ {param_types} }}\n\n"
            f"const router = Router();\n\n"
            f"/**\n * {spec.description}\n */\n"
            f"router.post('/{route}', async (req: Request, res: Response) => {{\n"
            f"  const {{ {param_destructure} }}: {spec.name}Body = req.body;\n"
            f"  res.json({{ status: 'ok' }});\n"
            f"}});\n\n"
            f"export default router;\n"
        )

    # ------------------------------------------------------------------
    # DATA MODEL templates
    # ------------------------------------------------------------------

    def _py_data_model(self, spec: CodeSpec) -> str:
        """Python dataclass."""
        fields = "\n".join(
            f"    {p.name}: {p.type}"
            + (f" = {repr(p.default)}" if p.default is not None else "")
            for p in spec.inputs
        ) or "    pass"
        return (
            f"from dataclasses import dataclass\n\n\n"
            f"@dataclass\n"
            f"class {spec.name}:\n"
            f'    """{spec.description}"""\n'
            f"{fields}\n"
        )

    def _js_data_model(self, spec: CodeSpec) -> str:
        """JSDoc-annotated factory function."""
        props = "\n".join(f"   * @property {{{p.type}}} {p.name} - {p.description}" for p in spec.inputs)
        defaults = ",\n    ".join(
            f"{p.name}: {repr(p.default) if p.default is not None else 'undefined'}"
            for p in spec.inputs
        )
        return (
            f"/**\n"
            f" * {spec.description}\n"
            f" *\n"
            f"{props}\n"
            f" */\n"
            f"function create{spec.name}({{{defaults}}}) {{\n"
            f"  return {{ {', '.join(p.name for p in spec.inputs)} }};\n"
            f"}}\n\n"
            f"module.exports = {{ create{spec.name} }};\n"
        )

    def _ts_data_model(self, spec: CodeSpec) -> str:
        """TypeScript interface + factory."""
        fields = "\n".join(
            f"  {p.name}{'?' if not p.required else ''}: {p.type};"
            for p in spec.inputs
        )
        param_str = _ts_param_str(spec.inputs)
        assigns = ", ".join(p.name for p in spec.inputs)
        return (
            f"/** {spec.description} */\n"
            f"export interface {spec.name} {{\n"
            f"{fields}\n"
            f"}}\n\n"
            f"export function create{spec.name}({param_str}): {spec.name} {{\n"
            f"  return {{ {assigns} }};\n"
            f"}}\n"
        )

    # ------------------------------------------------------------------
    # TEST SUITE templates
    # ------------------------------------------------------------------

    def _py_test_suite(self, spec: CodeSpec) -> str:
        """pytest test file."""
        test_cases = []
        if spec.examples:
            for i, ex in enumerate(spec.examples):
                args = ", ".join(repr(v) for v in ex.inputs.values())
                expected = repr(ex.expected_output)
                desc = ex.description or f"case_{i+1}"
                import re
                safe_desc = re.sub(r"[^a-z0-9_]", "_", desc.lower().replace(" ", "_"))
                safe_desc = re.sub(r"_+", "_", safe_desc).strip("_")
                test_cases.append(
                    f"def test_{spec.name}_{safe_desc}():\n"
                    f'    """Test: {desc}"""\n'
                    f"    result = {spec.name}({args})\n"
                    f"    assert result == {expected}\n"
                )
        else:
            test_cases.append(
                f"def test_{spec.name}_basic():\n"
                f'    """Basic smoke test for {spec.name}"""\n'
                f"    result = {spec.name}()\n"
                f"    assert result is not None\n"
            )
        tests_str = "\n\n".join(test_cases)
        return (
            f"import pytest\n"
            f"# TODO: import {spec.name} from your module\n\n\n"
            f"{tests_str}\n"
        )

    def _js_test_suite(self, spec: CodeSpec) -> str:
        """Jest test file."""
        test_cases = []
        if spec.examples:
            for ex in spec.examples:
                args = ", ".join(repr(v) for v in ex.inputs.values())
                expected = repr(ex.expected_output)
                desc = ex.description or "basic"
                test_cases.append(
                    f"  test('{desc}', () => {{\n"
                    f"    expect({spec.name}({args})).toEqual({expected});\n"
                    f"  }});\n"
                )
        else:
            test_cases.append(
                f"  test('basic', () => {{\n"
                f"    expect({spec.name}()).toBeDefined();\n"
                f"  }});\n"
            )
        tests_str = "\n".join(test_cases)
        return (
            f"const {{ {spec.name} }} = require('./{spec.name}');\n\n"
            f"describe('{spec.name}', () => {{\n"
            f"{tests_str}"
            f"}});\n"
        )

    def _ts_test_suite(self, spec: CodeSpec) -> str:
        """Jest + TypeScript test file."""
        test_cases = []
        if spec.examples:
            for ex in spec.examples:
                args = ", ".join(repr(v) for v in ex.inputs.values())
                expected = repr(ex.expected_output)
                desc = ex.description or "basic"
                test_cases.append(
                    f"  test('{desc}', () => {{\n"
                    f"    expect({spec.name}({args})).toEqual({expected});\n"
                    f"  }});\n"
                )
        else:
            test_cases.append(
                f"  test('basic', () => {{\n"
                f"    expect({spec.name}()).toBeDefined();\n"
                f"  }});\n"
            )
        tests_str = "\n".join(test_cases)
        return (
            f"import {{ {spec.name} }} from './{spec.name}';\n\n"
            f"describe('{spec.name}', () => {{\n"
            f"{tests_str}"
            f"}});\n"
        )
