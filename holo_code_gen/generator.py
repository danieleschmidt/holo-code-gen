"""
HoloCodeGenerator: Orchestrates spec parsing → template selection → code generation → validation.

The main entry point. Given a CodeSpec (or a dict/JSON that describes one),
it drives the full pipeline and returns the generated code along with
validation results.
"""

from __future__ import annotations
import json
from typing import Dict, Any, Optional, List
from .spec import CodeSpec, InputParam, OutputParam, Example
from .templates import TemplateEngine
from .validator import SpecValidator, ValidationResult


class GenerationResult:
    """Full output from a single generation run."""

    def __init__(
        self,
        spec: CodeSpec,
        code: str,
        validation: ValidationResult,
    ):
        self.spec = spec
        self.code = code
        self.validation = validation

    @property
    def success(self) -> bool:
        return self.validation.passed

    def __repr__(self) -> str:
        lines = [
            f"=== GenerationResult: {self.spec.name} ===",
            f"Template : {self.spec.template}",
            f"Language : {self.spec.language}",
            f"Validation: {'PASS' if self.validation.passed else 'FAIL'}",
        ]
        if not self.validation.passed or self.validation.warnings:
            for err in self.validation.errors:
                lines.append(f"  {err}")
        lines.append("")
        lines.append("--- Generated Code ---")
        lines.append(self.code)
        lines.append("--- End ---")
        return "\n".join(lines)


class HoloCodeGenerator:
    """
    Orchestrates the full holo-code-gen pipeline.

    Usage:
        gen = HoloCodeGenerator()
        result = gen.generate(spec)
        print(result.code)
        print(result.validation)
    """

    def __init__(self):
        self.engine = TemplateEngine()
        self.validator = SpecValidator()

    def generate(self, spec: CodeSpec) -> GenerationResult:
        """
        Generate code from a CodeSpec.

        Steps:
          1. Validate the spec itself (field-level checks)
          2. Select and render the appropriate template
          3. Validate the generated code against the spec
          4. Return a GenerationResult

        Raises:
            ValueError: if the spec itself is invalid
        """
        # Step 1: Validate spec
        spec_errors = spec.validate()
        if spec_errors:
            raise ValueError(f"Invalid spec for '{spec.name}': {'; '.join(spec_errors)}")

        # Step 2: Render template
        code = self.engine.render(spec)

        # Step 3: Validate generated code
        validation = self.validator.validate(code, spec)

        return GenerationResult(spec=spec, code=code, validation=validation)

    def generate_from_dict(self, data: Dict[str, Any]) -> GenerationResult:
        """
        Generate code from a plain dictionary (e.g., parsed JSON/YAML).

        Expected keys mirror CodeSpec fields.
        """
        spec = self._dict_to_spec(data)
        return self.generate(spec)

    def generate_from_json(self, json_str: str) -> GenerationResult:
        """Generate code from a JSON string."""
        data = json.loads(json_str)
        return self.generate_from_dict(data)

    def generate_batch(self, specs: List[CodeSpec]) -> List[GenerationResult]:
        """Generate code for a list of specs, collecting all results."""
        return [self.generate(spec) for spec in specs]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _dict_to_spec(self, data: Dict[str, Any]) -> CodeSpec:
        inputs = [
            InputParam(
                name=p["name"],
                type=p.get("type", "Any"),
                description=p.get("description", ""),
                default=p.get("default"),
                required=p.get("required", True),
            )
            for p in data.get("inputs", [])
        ]
        out_data = data.get("output")
        output = OutputParam(
            type=out_data["type"],
            description=out_data.get("description", ""),
        ) if out_data else None
        examples = [
            Example(
                inputs=e.get("inputs", {}),
                expected_output=e.get("expected_output"),
                description=e.get("description", ""),
            )
            for e in data.get("examples", [])
        ]
        return CodeSpec(
            name=data["name"],
            template=data["template"],
            language=data["language"],
            description=data["description"],
            inputs=inputs,
            output=output,
            constraints=data.get("constraints", []),
            examples=examples,
            tags=data.get("tags", []),
            dependencies=data.get("dependencies", []),
        )
