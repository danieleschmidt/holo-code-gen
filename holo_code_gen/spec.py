"""
CodeSpec: Structured specification for a code unit.

Defines what a function, class, endpoint, model, or test suite should do
before any code is generated.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class InputParam:
    """A named input parameter with type and optional default."""
    name: str
    type: str
    description: str = ""
    default: Optional[Any] = None
    required: bool = True


@dataclass
class OutputParam:
    """Return value specification."""
    type: str
    description: str = ""


@dataclass
class Example:
    """A concrete input/output example for the code unit."""
    inputs: Dict[str, Any]
    expected_output: Any
    description: str = ""


@dataclass
class CodeSpec:
    """
    Complete specification for a code unit to be generated.

    Attributes:
        name:           Identifier for the code unit (e.g., 'calculate_tax')
        template:       Template type: 'function' | 'class' | 'rest_endpoint' |
                        'data_model' | 'test_suite'
        language:       Target language: 'python' | 'javascript' | 'typescript'
        description:    Human-readable description of what this unit does
        inputs:         List of input parameters
        output:         Return value specification
        constraints:    List of behavioral constraints (e.g., "must be O(n)")
        examples:       Concrete usage examples
        tags:           Optional metadata tags
        dependencies:   External modules this code requires
    """
    name: str
    template: str
    language: str
    description: str
    inputs: List[InputParam] = field(default_factory=list)
    output: Optional[OutputParam] = None
    constraints: List[str] = field(default_factory=list)
    examples: List[Example] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)

    VALID_TEMPLATES = {"function", "class", "rest_endpoint", "data_model", "test_suite"}
    VALID_LANGUAGES = {"python", "javascript", "typescript"}

    def validate(self) -> List[str]:
        """
        Validate the spec itself (not the generated code).
        Returns a list of error strings; empty list means valid.
        """
        errors = []
        if not self.name or not self.name.replace("_", "").isalnum():
            errors.append(f"Invalid name '{self.name}': must be alphanumeric/underscore")
        if self.template not in self.VALID_TEMPLATES:
            errors.append(f"Unknown template '{self.template}'. Valid: {self.VALID_TEMPLATES}")
        if self.language not in self.VALID_LANGUAGES:
            errors.append(f"Unknown language '{self.language}'. Valid: {self.VALID_LANGUAGES}")
        if not self.description.strip():
            errors.append("Description cannot be empty")
        for inp in self.inputs:
            if not inp.name or not inp.name.replace("_", "").isalnum():
                errors.append(f"Invalid input param name '{inp.name}'")
            if not inp.type:
                errors.append(f"Input param '{inp.name}' missing type")
        return errors

    def to_dict(self) -> Dict[str, Any]:
        """Serialize spec to a plain dict (for debugging/logging)."""
        return {
            "name": self.name,
            "template": self.template,
            "language": self.language,
            "description": self.description,
            "inputs": [
                {"name": p.name, "type": p.type, "description": p.description,
                 "default": p.default, "required": p.required}
                for p in self.inputs
            ],
            "output": {"type": self.output.type, "description": self.output.description}
                       if self.output else None,
            "constraints": self.constraints,
            "examples": [
                {"inputs": e.inputs, "expected_output": e.expected_output, "description": e.description}
                for e in self.examples
            ],
            "tags": self.tags,
            "dependencies": self.dependencies,
        }
