"""Shared test fixtures."""
import pytest
from pathlib import Path
import sys

# Ensure src/ is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline.extraction import extract_contract
from src.pipeline.parsing import build_clause_tree
from src.pipeline.alignment import align_clauses


@pytest.fixture(scope="session")
def contract_paths():
    """Paths to test contract PDFs."""
    root = Path(__file__).resolve().parent.parent
    return {
        "v1": str(root / "docs" / "天猫服务协议2015(2).pdf"),
        "v2": str(root / "docs" / "天猫服务协议2026(2).pdf"),
    }


@pytest.fixture(scope="session")
def v1_doc(contract_paths):
    """Extracted 2015 contract document."""
    return extract_contract(contract_paths["v1"])


@pytest.fixture(scope="session")
def v2_doc(contract_paths):
    """Extracted 2026 contract document."""
    return extract_contract(contract_paths["v2"])


@pytest.fixture(scope="session")
def v1_tree(v1_doc):
    """Clause tree for 2015 contract."""
    return build_clause_tree(v1_doc.full_text)


@pytest.fixture(scope="session")
def v2_tree(v2_doc):
    """Clause tree for 2026 contract."""
    return build_clause_tree(v2_doc.full_text)


@pytest.fixture(scope="session")
def diff_map(v1_tree, v2_tree):
    """Alignment map between the two contracts."""
    return align_clauses(v1_tree, v2_tree)
