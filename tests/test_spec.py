"""Tests for CodeSpec validation."""
import pytest
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from holo_code_gen.spec import CodeSpec, InputParam, OutputParam, Example


def make_spec(**kwargs):
    defaults = dict(
        name="my_func",
        template="function",
        language="python",
        description="A test function",
    )
    defaults.update(kwargs)
    return CodeSpec(**defaults)


def test_valid_spec_no_errors():
    spec = make_spec()
    assert spec.validate() == []


def test_invalid_template():
    spec = make_spec(template="wizard")
    errors = spec.validate()
    assert any("template" in e.lower() for e in errors)


def test_invalid_language():
    spec = make_spec(language="cobol")
    errors = spec.validate()
    assert any("language" in e.lower() for e in errors)


def test_empty_description():
    spec = make_spec(description="   ")
    errors = spec.validate()
    assert any("description" in e.lower() for e in errors)


def test_invalid_name():
    spec = make_spec(name="my-func!")
    errors = spec.validate()
    assert len(errors) > 0


def test_input_params_stored():
    spec = make_spec(
        inputs=[InputParam("x", "int"), InputParam("y", "str")]
    )
    assert len(spec.inputs) == 2
    assert spec.inputs[0].name == "x"


def test_to_dict_roundtrip():
    spec = make_spec(
        inputs=[InputParam("x", "int", "An integer")],
        output=OutputParam("str", "A string"),
        constraints=["x > 0"],
        examples=[Example({"x": 1}, "one", "example")],
    )
    d = spec.to_dict()
    assert d["name"] == "my_func"
    assert d["inputs"][0]["name"] == "x"
    assert d["output"]["type"] == "str"
    assert d["constraints"] == ["x > 0"]
    assert d["examples"][0]["description"] == "example"
