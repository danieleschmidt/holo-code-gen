"""
holo-code-gen: Holographic code generation from structured specifications.

Generates Python, JavaScript, and TypeScript code stubs from typed specs
using a template-based synthesis engine — no LLMs, no external dependencies.
"""

__version__ = "1.0.0"
__author__ = "Daniel Schmidt"

from .spec import CodeSpec, InputParam, OutputParam, Example
from .templates import TemplateEngine
from .validator import SpecValidator
from .generator import HoloCodeGenerator

__all__ = [
    "CodeSpec",
    "InputParam",
    "OutputParam",
    "Example",
    "TemplateEngine",
    "SpecValidator",
    "HoloCodeGenerator",
]
