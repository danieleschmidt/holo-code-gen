"""Tests for HoloCodeGenerator — full pipeline."""
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holo_code_gen import HoloCodeGenerator, CodeSpec, InputParam, OutputParam, Example


gen = HoloCodeGenerator()


def simple_spec(**kwargs):
    defaults = dict(
        name="add_numbers",
        template="function",
        language="python",
        description="Add two numbers",
        inputs=[InputParam("a", "int"), InputParam("b", "int")],
        output=OutputParam("int", "sum"),
    )
    defaults.update(kwargs)
    return CodeSpec(**defaults)


class TestGenerate:
    def test_returns_generation_result(self):
        from holo_code_gen.generator import GenerationResult
        result = gen.generate(simple_spec())
        assert isinstance(result, GenerationResult)

    def test_result_has_code(self):
        result = gen.generate(simple_spec())
        assert "def add_numbers(" in result.code

    def test_result_has_validation(self):
        result = gen.generate(simple_spec())
        assert result.validation is not None

    def test_success_on_valid_spec(self):
        result = gen.generate(simple_spec())
        assert result.success

    def test_invalid_spec_raises(self):
        with pytest.raises(ValueError, match="Invalid spec"):
            gen.generate(CodeSpec(
                name="bad name!",
                template="function",
                language="python",
                description="x",
            ))


class TestGenerateFromDict:
    def test_dict_roundtrip(self):
        data = {
            "name": "greet",
            "template": "function",
            "language": "python",
            "description": "Return a greeting string",
            "inputs": [{"name": "name", "type": "str"}],
            "output": {"type": "str", "description": "greeting"},
        }
        result = gen.generate_from_dict(data)
        assert "def greet(" in result.code

    def test_json_roundtrip(self):
        import json
        data = {
            "name": "square",
            "template": "function",
            "language": "python",
            "description": "Square a number",
            "inputs": [{"name": "n", "type": "int"}],
            "output": {"type": "int"},
        }
        result = gen.generate_from_json(json.dumps(data))
        assert "def square(" in result.code


class TestGenerateBatch:
    def test_batch_returns_all(self):
        specs = [
            simple_spec(name="func_a"),
            simple_spec(name="func_b"),
            simple_spec(name="func_c"),
        ]
        results = gen.generate_batch(specs)
        assert len(results) == 3
        names = [r.spec.name for r in results]
        assert names == ["func_a", "func_b", "func_c"]


class TestAllTemplates:
    @pytest.mark.parametrize("template,language", [
        ("function", "python"),
        ("function", "javascript"),
        ("function", "typescript"),
        ("class", "python"),
        ("class", "typescript"),
        ("rest_endpoint", "python"),
        ("rest_endpoint", "javascript"),
        ("data_model", "python"),
        ("data_model", "typescript"),
        ("test_suite", "python"),
    ])
    def test_template_language_combo(self, template, language):
        spec = CodeSpec(
            name="test_unit",
            template=template,
            language=language,
            description="A test unit for all templates",
            inputs=[InputParam("value", "str", "input value")],
        )
        result = gen.generate(spec)
        assert result.code.strip() != ""
