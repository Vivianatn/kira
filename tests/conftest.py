"""Fixtures partagées des tests Kira."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Permet `import kira` quand on lance pytest depuis la racine du projet.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from kira.security import EnforcementLayer  # noqa: E402


WORKSPACE_POLICY = """
version: 1
allowed_tools:
  - web
  - files
tools:
  files:
    root: workspace
    mode: read_only
    max_bytes: 100000
  web:
    max_results: 5
    provider: duckduckgo
require_human_approval: []
agent:
  max_steps: 4
"""


@pytest.fixture
def project(tmp_path: Path):
    """Crée un faux projet isolé (policy.yaml + workspace/) dans un tmp dir."""
    (tmp_path / "policy.yaml").write_text(WORKSPACE_POLICY, encoding="utf-8")
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "notes.txt").write_text("bonjour kira", encoding="utf-8")
    return tmp_path


@pytest.fixture
def security(project: Path) -> EnforcementLayer:
    return EnforcementLayer(
        policy_path=project / "policy.yaml",
        project_root=project,
    )
