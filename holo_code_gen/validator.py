"""
SpecValidator: Validates generated code matches its CodeSpec.

Uses the ast module to parse and inspect the generated source without
executing it. Checks:
  - Function/class definitions exist with the expected name
  - Function signatures match specified input params
  - Return type annotations (when present) match the spec output type
  - Required imports for declared dependencies are present
"""

from __future__ import annotations
import ast
from typing import List, Tuple
from .spec import CodeSpec


class ValidationError:
    """A single validation finding."""
    def __init__(self, level: str, message: str):
        self.level = level  # "error" | "warning"
        self.message = message

    def __repr__(self) -> str:
        return f"[{self.level.upper()}] {self.message}"


class ValidationResult:
    """Aggregate result of validating generated code against a spec."""
    def __init__(self, spec_name: str, errors: List[ValidationError]):
        self.spec_name = spec_name
        self.errors = errors

    @property
    def passed(self) -> bool:
        return not any(e.level == "error" for e in self.errors)

    @property
    def warnings(self) -> List[ValidationError]:
        return [e for e in self.errors if e.level == "warning"]

    @property
    def error_list(self) -> List[ValidationError]:
        return [e for e in self.errors if e.level == "error"]

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"ValidationResult({self.spec_name}): {status}"]
        for e in self.errors:
            lines.append(f"  {e}")
        return "\n".join(lines)


class SpecValidator:
    """
    Validates generated Python code against a CodeSpec using ast parsing.

    Note: Only supports Python validation (ast module is Python-specific).
    For JS/TS, performs basic structural checks via string analysis.
    """

    def validate(self, code: str, spec: CodeSpec) -> ValidationResult:
        """
        Validate generated code against a spec.
        Returns a ValidationResult with any errors or warnings.
        """
        if spec.language == "python":
            return self._validate_python(code, spec)
        else:
            return self._validate_text(code, spec)

    def _validate_python(self, code: str, spec: CodeSpec) -> ValidationResult:
        findings: List[ValidationError] = []

        # Parse AST
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return ValidationResult(
                spec.name,
                [ValidationError("error", f"SyntaxError: {e}")]
            )

        # Gather all top-level definitions
        functions = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef)
        }
        classes = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
        }
        imports = self._extract_imports(tree)

        template = spec.template

        if template in ("function", "rest_endpoint"):
            if spec.name not in functions:
                findings.append(ValidationError("error", f"Expected function '{spec.name}' not found"))
            else:
                fn = functions[spec.name]
                findings.extend(self._check_function_args(fn, spec))
                findings.extend(self._check_return_annotation(fn, spec))

        elif template == "class":
            if spec.name not in classes:
                findings.append(ValidationError("error", f"Expected class '{spec.name}' not found"))
            else:
                cls = classes[spec.name]
                findings.extend(self._check_class_init(cls, spec))

        elif template == "data_model":
            if spec.name not in classes:
                findings.append(ValidationError("error", f"Expected dataclass '{spec.name}' not found"))
            else:
                cls = classes[spec.name]
                findings.extend(self._check_dataclass_fields(cls, spec))

        elif template == "test_suite":
            test_fns = [n for n in functions if n.startswith("test_")]
            if not test_fns:
                findings.append(ValidationError("error", "No test functions (test_*) found"))
            else:
                # Should have at least as many tests as examples
                min_tests = max(1, len(spec.examples))
                if len(test_fns) < min_tests:
                    findings.append(ValidationError(
                        "warning",
                        f"Expected at least {min_tests} test(s), found {len(test_fns)}"
                    ))

        # Check dependencies are imported
        for dep in spec.dependencies:
            if dep not in imports:
                findings.append(ValidationError(
                    "warning", f"Dependency '{dep}' declared but not imported"
                ))

        return ValidationResult(spec.name, findings)

    def _extract_imports(self, tree: ast.AST) -> set:
        """Return set of all imported module/name strings."""
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported.add(node.module.split(".")[0])
        return imported

    def _check_function_args(self, fn: ast.FunctionDef, spec: CodeSpec) -> List[ValidationError]:
        findings = []
        fn_args = [arg.arg for arg in fn.args.args]
        spec_args = [p.name for p in spec.inputs]

        # Compare required args (skip 'self')
        actual = [a for a in fn_args if a != "self"]
        if actual != spec_args:
            findings.append(ValidationError(
                "warning",
                f"Parameter list mismatch. Expected {spec_args}, got {actual}"
            ))
        return findings

    def _check_return_annotation(self, fn: ast.FunctionDef, spec: CodeSpec) -> List[ValidationError]:
        findings = []
        if spec.output and fn.returns is not None:
            actual = ast.unparse(fn.returns) if hasattr(ast, "unparse") else None
            if actual and actual != spec.output.type:
                findings.append(ValidationError(
                    "warning",
                    f"Return type annotation '{actual}' differs from spec '{spec.output.type}'"
                ))
        return findings

    def _check_class_init(self, cls: ast.ClassDef, spec: CodeSpec) -> List[ValidationError]:
        findings = []
        init = None
        for node in cls.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "__init__":
                init = node
                break
        if init is None:
            findings.append(ValidationError("warning", f"Class '{spec.name}' has no __init__"))
        else:
            init_args = [arg.arg for arg in init.args.args if arg.arg != "self"]
            spec_args = [p.name for p in spec.inputs]
            if init_args != spec_args:
                findings.append(ValidationError(
                    "warning",
                    f"__init__ params {init_args} differ from spec {spec_args}"
                ))
        return findings

    def _check_dataclass_fields(self, cls: ast.ClassDef, spec: CodeSpec) -> List[ValidationError]:
        findings = []
        # Check for @dataclass decorator
        decorator_names = []
        for d in cls.decorator_list:
            if isinstance(d, ast.Name):
                decorator_names.append(d.id)
            elif isinstance(d, ast.Attribute):
                decorator_names.append(d.attr)
        if "dataclass" not in decorator_names:
            findings.append(ValidationError("warning", f"Class '{spec.name}' missing @dataclass decorator"))

        # Check field names exist as class-level annotations
        annotated_fields = set()
        for node in cls.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                annotated_fields.add(node.target.id)
        for p in spec.inputs:
            if p.name not in annotated_fields:
                findings.append(ValidationError(
                    "warning",
                    f"Field '{p.name}' from spec not found in dataclass body"
                ))
        return findings

    def _validate_text(self, code: str, spec: CodeSpec) -> ValidationResult:
        """Lightweight structural check for JS/TS (no AST available)."""
        findings = []
        if spec.name not in code:
            findings.append(ValidationError(
                "error", f"Name '{spec.name}' not found in generated code"
            ))
        if spec.template == "test_suite":
            if "test(" not in code and "it(" not in code:
                findings.append(ValidationError("error", "No test() or it() calls found"))
        if spec.template in ("class", "data_model"):
            if "class " not in code and "interface " not in code and "function create" not in code:
                findings.append(ValidationError(
                    "warning", "No class, interface, or factory function found"
                ))
        return ValidationResult(spec.name, findings)
