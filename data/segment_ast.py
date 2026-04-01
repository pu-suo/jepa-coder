"""
AST-based solution segmentation for JEPA-Coder SST training data.

Implements the grouped strategy from architecture spec Section 5.2:
  - Merge consecutive simple statements into one block
  - Keep compound statements as individual blocks
  - Unwrap single-top-level-node solutions
  - Append STOP block
  - Discard solutions with <2 or >15 blocks (excluding STOP)
"""

import ast
import json
import statistics
from pathlib import Path


# Statement type classification per spec Section 5.2
SIMPLE_TYPES = (
    ast.Assign,
    ast.AugAssign,
    ast.AnnAssign,
    ast.Expr,
    ast.Import,
    ast.ImportFrom,
    ast.Return,
    ast.Pass,
    ast.Delete,
    ast.Global,
    ast.Nonlocal,
    ast.Raise,
)

COMPOUND_TYPES = (
    ast.For,
    ast.AsyncFor,
    ast.If,
    ast.While,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.With,
    ast.AsyncWith,
    ast.Try,
    ast.TryStar,  # Python 3.11+ except*
)


def _has_body(node: ast.AST) -> bool:
    return hasattr(node, "body") and isinstance(node.body, list)


def _group_nodes(nodes: list[ast.stmt]) -> list[list[ast.stmt]]:
    """Group consecutive simple stmts; each compound stmt gets its own group."""
    groups: list[list[ast.stmt]] = []
    simple_buf: list[ast.stmt] = []

    for node in nodes:
        if isinstance(node, SIMPLE_TYPES):
            simple_buf.append(node)
        elif isinstance(node, COMPOUND_TYPES):
            if simple_buf:
                groups.append(simple_buf)
                simple_buf = []
            groups.append([node])
        else:
            # Unrecognised node type — treat as simple to avoid dropping it
            simple_buf.append(node)

    if simple_buf:
        groups.append(simple_buf)

    return groups


def segment_solution_grouped(code: str) -> list[dict] | None:
    """
    Segment a Python solution string into code blocks using the grouped AST strategy.

    Returns a list of block dicts [{'type': 'CODE'|'STOP', 'code': str}],
    or None if the solution is unparseable or falls outside the [2, 15] block range.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    body = tree.body
    if not body:
        return None

    # Unwrap single-top-level-node: if the module body has exactly one node
    # that has its own body (FunctionDef, If, For, …), segment inside it.
    if len(body) == 1 and _has_body(body[0]):
        body = body[0].body
        if not body:
            return None

    groups = _group_nodes(body)

    # Filter on block count (excluding the STOP block we'll append)
    n_blocks = len(groups)
    if n_blocks < 2 or n_blocks > 15:
        return None

    blocks: list[dict] = []
    for group in groups:
        # Build a temporary module so ast.unparse can reconstruct source
        tmp = ast.Module(body=group, type_ignores=[])
        try:
            source = ast.unparse(tmp)
        except Exception:
            return None
        blocks.append({"type": "CODE", "code": source})

    blocks.append({"type": "STOP", "code": "<STOP>"})
    return blocks


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    jsonl_path = Path(__file__).parent / "extracted_solutions.jsonl"

    counts: list[int] = []
    discarded = 0
    checked = 0

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            if checked >= 100:
                break
            entry = json.loads(line)
            code = entry["solution"]
            blocks = segment_solution_grouped(code)
            checked += 1
            if blocks is None:
                discarded += 1
            else:
                # Count excludes the STOP block
                counts.append(sum(1 for b in blocks if b["type"] == "CODE"))

    print(f"Checked : {checked}")
    print(f"Accepted: {len(counts)}  |  Discarded: {discarded}")
    if counts:
        print(f"Block counts (excl. STOP):")
        print(f"  mean   : {statistics.mean(counts):.2f}")
        print(f"  median : {statistics.median(counts):.1f}")
        print(f"  min    : {min(counts)}")
        print(f"  max    : {max(counts)}")
