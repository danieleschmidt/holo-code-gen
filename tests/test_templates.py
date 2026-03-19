"""Tests for TemplateEngine — covers all 5 template types × Python."""
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holo_code_gen.spec import CodeSpec, InputParam, OutputParam, Example
from holo_code_gen.templates import TemplateEngine


engine = TemplateEngine()


def spec(template, language="python", **kwargs):
    defaults = dict(
        name="my_unit",
        template=template,
        language=language,
        description="Test unit",
        inputs=[InputParam("x", "int"), InputParam("y", "str")],
        output=OutputParam("bool", "result"),
    )
    defaults.update(kwargs)
    return CodeSpec(**defaults)


# --------------------------------------------------------- function
class TestFunctionTemplate:
    def test_python_has_def(self):
        code = engine.render(spec("function"))
        assert "def my_unit(" in code

    def test_python_params_present(self):
        code = engine.render(spec("function"))
        assert "x: int" in code
        assert "y: str" in code

    def test_python_return_annotation(self):
        code = engine.render(spec("function"))
        assert "-> bool" in code

    def test_python_docstring(self):
        code = engine.render(spec("function"))
        assert '"""' in code

    def test_javascript_function(self):
        code = engine.render(spec("function", "javascript"))
        assert "function my_unit(" in code
        assert "module.exports" in code

    def test_typescript_export(self):
        code = engine.render(spec("function", "typescript"))
        assert "export function my_unit(" in code


# --------------------------------------------------------- class
class TestClassTemplate:
    def test_python_class_def(self):
        code = engine.render(spec("class"))
        assert "class my_unit:" in code

    def test_python_init_params(self):
        code = engine.render(spec("class"))
        assert "def __init__(self, x: int, y: str)" in code

    def test_typescript_class(self):
        code = engine.render(spec("class", "typescript"))
        assert "export class my_unit" in code
        assert "x: int;" in code

    def test_javascript_class(self):
        code = engine.render(spec("class", "javascript"))
        assert "class my_unit" in code


# --------------------------------------------------------- rest_endpoint
class TestRestEndpointTemplate:
    def test_python_has_route(self):
        code = engine.render(spec("rest_endpoint"))
        assert "@app.post" in code
        assert "async def my_unit(" in code

    def test_python_fastapi_import(self):
        code = engine.render(spec("rest_endpoint"))
        assert "fastapi" in code.lower() or "FastAPI" in code

    def test_javascript_express(self):
        code = engine.render(spec("rest_endpoint", "javascript"))
        assert "router.post" in code

    def test_typescript_express(self):
        code = engine.render(spec("rest_endpoint", "typescript"))
        assert "Router" in code or "router.post" in code


# --------------------------------------------------------- data_model
class TestDataModelTemplate:
    def test_python_dataclass(self):
        code = engine.render(spec("data_model"))
        assert "@dataclass" in code
        assert "class my_unit:" in code
        assert "x: int" in code

    def test_typescript_interface(self):
        code = engine.render(spec("data_model", "typescript"))
        assert "interface my_unit" in code or "export interface my_unit" in code

    def test_javascript_factory(self):
        code = engine.render(spec("data_model", "javascript"))
        assert "create" in code


# --------------------------------------------------------- test_suite
class TestTestSuiteTemplate:
    def test_python_pytest_functions(self):
        code = engine.render(spec("test_suite"))
        assert "def test_" in code

    def test_python_with_examples(self):
        s = spec(
            "test_suite",
            examples=[
                Example({"x": 1, "y": "a"}, True, "case one"),
                Example({"x": 2, "y": "b"}, False, "case two"),
            ],
        )
        code = engine.render(s)
        assert code.count("def test_") == 2

    def test_javascript_jest(self):
        code = engine.render(spec("test_suite", "javascript"))
        assert "test(" in code or "describe(" in code

    def test_typescript_jest(self):
        code = engine.render(spec("test_suite", "typescript"))
        assert "test(" in code


# --------------------------------------------------------- unknown template
def test_unknown_template_raises():
    s = CodeSpec(
        name="x", template="function", language="python", description="x"
    )
    s.template = "wizard"  # bypass validation
    with pytest.raises(ValueError, match="No template"):
        engine.render(s)
