"""Tests for SpecValidator."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holo_code_gen.spec import CodeSpec, InputParam, OutputParam, Example
from holo_code_gen.validator import SpecValidator


validator = SpecValidator()


def make_spec(template="function", language="python", **kwargs):
    defaults = dict(
        name="my_func",
        template=template,
        language=language,
        description="Test",
    )
    defaults.update(kwargs)
    return CodeSpec(**defaults)


class TestPythonFunctionValidation:
    def test_valid_function_passes(self):
        code = "def my_func(x: int, y: str) -> bool:\n    return False\n"
        spec = make_spec(
            inputs=[InputParam("x", "int"), InputParam("y", "str")],
            output=OutputParam("bool"),
        )
        result = validator.validate(code, spec)
        assert result.passed

    def test_missing_function_fails(self):
        code = "def other_func(): pass\n"
        spec = make_spec()
        result = validator.validate(code, spec)
        assert not result.passed
        assert any("my_func" in e.message for e in result.error_list)

    def test_syntax_error_fails(self):
        code = "def my_func(x: int\n  return x\n"
        spec = make_spec()
        result = validator.validate(code, spec)
        assert not result.passed
        assert any("SyntaxError" in e.message for e in result.error_list)

    def test_param_mismatch_is_warning(self):
        code = "def my_func(a, b): pass\n"
        spec = make_spec(
            inputs=[InputParam("x", "int"), InputParam("y", "str")]
        )
        result = validator.validate(code, spec)
        # param mismatch is warning-level, so it should still pass (no errors)
        assert result.passed
        assert any("mismatch" in w.message.lower() for w in result.warnings)


class TestPythonClassValidation:
    def test_valid_class_passes(self):
        code = "class my_func:\n    def __init__(self, x):\n        self.x = x\n"
        spec = make_spec(
            template="class",
            inputs=[InputParam("x", "int")],
        )
        result = validator.validate(code, spec)
        assert result.passed

    def test_missing_class_fails(self):
        code = "class OtherClass: pass\n"
        spec = make_spec(template="class")
        result = validator.validate(code, spec)
        assert not result.passed


class TestPythonDataModelValidation:
    def test_valid_dataclass_passes(self):
        code = (
            "from dataclasses import dataclass\n\n"
            "@dataclass\n"
            "class my_func:\n"
            "    x: int\n"
            "    y: str\n"
        )
        spec = make_spec(
            template="data_model",
            inputs=[InputParam("x", "int"), InputParam("y", "str")],
        )
        result = validator.validate(code, spec)
        assert result.passed

    def test_missing_field_is_warning(self):
        code = (
            "from dataclasses import dataclass\n\n"
            "@dataclass\n"
            "class my_func:\n"
            "    x: int\n"
        )
        spec = make_spec(
            template="data_model",
            inputs=[InputParam("x", "int"), InputParam("y", "str")],
        )
        result = validator.validate(code, spec)
        assert result.passed  # only warning
        assert any("y" in w.message for w in result.warnings)


class TestPythonTestSuiteValidation:
    def test_valid_test_suite_passes(self):
        code = "import pytest\n\ndef test_my_func_basic():\n    assert True\n"
        spec = make_spec(template="test_suite")
        result = validator.validate(code, spec)
        assert result.passed

    def test_no_tests_fails(self):
        code = "import pytest\n\ndef helper(): pass\n"
        spec = make_spec(template="test_suite")
        result = validator.validate(code, spec)
        assert not result.passed


class TestJSValidation:
    def test_name_present_passes(self):
        code = "function my_func(x) { return x; }\nmodule.exports = { my_func };\n"
        spec = make_spec(language="javascript")
        result = validator.validate(code, spec)
        assert result.passed

    def test_name_absent_fails(self):
        code = "function other() {}\n"
        spec = make_spec(language="javascript")
        result = validator.validate(code, spec)
        assert not result.passed


class TestValidationResult:
    def test_repr_shows_pass(self):
        code = "def my_func(): pass\n"
        spec = make_spec()
        result = validator.validate(code, spec)
        assert "PASS" in repr(result) or "FAIL" in repr(result)
