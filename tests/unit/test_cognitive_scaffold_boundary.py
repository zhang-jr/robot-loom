"""Boundary regression — tools/cognitive/ must not import memory/ or critic/.

ADR-018: CognitiveScaffold is session-scoped LLM meta-cognition.
It must never depend on robot_harness.memory (cross-session persistence)
or robot_harness.critic (embodied physical-world progress judgement).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_COGNITIVE_DIR = Path(__file__).parent.parent.parent / "robot_harness" / "tools" / "cognitive"

_FORBIDDEN_IMPORTS = [
    "robot_harness.memory",
    "robot_harness.critic",
    "robot_harness.tools.memory",
    "robot_harness.tools.critic",
]


def _collect_imports(source: str) -> list[str]:
    """Return all module names imported in the source file."""
    tree = ast.parse(source)
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                modules.append(node.module)
    return modules


@pytest.mark.parametrize("py_file", list(_COGNITIVE_DIR.glob("*.py")))
def test_no_forbidden_imports(py_file: Path) -> None:
    """Each file in tools/cognitive/ must not import from memory or critic."""
    source = py_file.read_text(encoding="utf-8")
    imports = _collect_imports(source)
    for imp in imports:
        for forbidden in _FORBIDDEN_IMPORTS:
            assert not imp.startswith(forbidden), (
                f"{py_file.name} illegally imports '{imp}' "
                f"(matches forbidden prefix '{forbidden}').  "
                "CognitiveScaffold must not depend on Memory or Critic. See ADR-018."
            )


def test_reflection_name_not_aliased_as_critic() -> None:
    """'reflection' must never be aliased or renamed to 'critic' in cognitive files."""
    for py_file in _COGNITIVE_DIR.glob("*.py"):
        source = py_file.read_text(encoding="utf-8")
        # Simple heuristic: no variable or class named 'critic' in cognitive dir
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name | ast.Attribute):
                name = node.id if isinstance(node, ast.Name) else node.attr
                assert name != "CriticStore", (
                    f"{py_file.name} references 'CriticStore' — must use 'ReflectionStore'. "
                    "See ADR-018 naming audit."
                )
