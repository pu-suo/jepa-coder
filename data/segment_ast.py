"""
AST-based solution segmentation for JEPA-Coder SST training data.

Implements the recursive logical flattening strategy for Phase 2 SST:
  - Simple State Rule: group consecutive SIMPLE_TYPES into one block.
  - Control-Flow Boundary Rule: emit a valid pass-stub for each compound
    statement, then recurse into its body.
  - elif Chain Rolling: roll the full if/elif/else condition chain into a
    single signature block before recursing into each branch body.
  - Complexity Budget: if a compound node's primary body has ≤3 simple
    statements and zero nested compound statements, emit it monolithically
    (no recursion) to avoid trivial single-line blobs.
  - Append STOP block.
  - Discard solutions with <2 or >15 blocks (excluding STOP).
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


def _count_body_stats(stmts: list[ast.stmt]) -> tuple[int, int]:
    """Return (n_simple, n_compound) for a flat list of statements."""
    n_simple = n_compound = 0
    for s in stmts:
        if isinstance(s, SIMPLE_TYPES):
            n_simple += 1
        elif isinstance(s, COMPOUND_TYPES):
            n_compound += 1
    return n_simple, n_compound


def _is_trivial_body(node: ast.stmt) -> bool:
    """
    Complexity Budget: True if the compound node's primary body has
    ≤3 simple statements and zero nested compound statements.
    Trivial bodies are emitted as monolithic blocks without recursion.
    """
    body = getattr(node, "body", [])
    n_simple, n_compound = _count_body_stats(body)
    return n_simple <= 3 and n_compound == 0


def _build_if_stub(node: ast.If) -> ast.If:
    """
    Recursively build an if/elif/else stub that preserves the full condition
    chain.  Each branch body is replaced with [Pass()].

    Example output for if/elif/else:
        if cond1: pass
        elif cond2: pass
        else: pass
    """
    orelse = node.orelse
    if len(orelse) == 1 and isinstance(orelse[0], ast.If):
        stub_orelse: list[ast.stmt] = [_build_if_stub(orelse[0])]
    elif orelse:
        stub_orelse = [ast.Pass()]
    else:
        stub_orelse = []
    return ast.If(test=node.test, body=[ast.Pass()], orelse=stub_orelse)


def _make_stub(node: ast.stmt) -> ast.stmt | None:
    """
    Build a valid, parseable pass-stub for a compound statement.
    Returns None for unrecognised node types.
    """
    if isinstance(node, ast.For):
        return ast.For(
            target=node.target, iter=node.iter,
            body=[ast.Pass()], orelse=[],
        )
    if isinstance(node, ast.AsyncFor):
        return ast.AsyncFor(
            target=node.target, iter=node.iter,
            body=[ast.Pass()], orelse=[],
        )
    if isinstance(node, ast.While):
        return ast.While(test=node.test, body=[ast.Pass()], orelse=[])
    if isinstance(node, ast.If):
        return _build_if_stub(node)
    if isinstance(node, ast.With):
        return ast.With(items=node.items, body=[ast.Pass()])
    if isinstance(node, ast.AsyncWith):
        return ast.AsyncWith(items=node.items, body=[ast.Pass()])
    if isinstance(node, ast.FunctionDef):
        return ast.FunctionDef(
            name=node.name, args=node.args, body=[ast.Pass()],
            decorator_list=[], returns=node.returns,
        )
    if isinstance(node, ast.AsyncFunctionDef):
        return ast.AsyncFunctionDef(
            name=node.name, args=node.args, body=[ast.Pass()],
            decorator_list=[], returns=node.returns,
        )
    if isinstance(node, ast.ClassDef):
        return ast.ClassDef(
            name=node.name, bases=node.bases, keywords=node.keywords,
            body=[ast.Pass()], decorator_list=[],
        )
    if isinstance(node, ast.Try):
        handlers = [
            ast.ExceptHandler(type=h.type, name=None, body=[ast.Pass()])
            for h in node.handlers
        ]
        return ast.Try(
            body=[ast.Pass()], handlers=handlers, orelse=[], finalbody=[],
        )
    if hasattr(ast, "TryStar") and isinstance(node, ast.TryStar):
        handlers = [
            ast.ExceptHandler(type=h.type, name=None, body=[ast.Pass()])
            for h in node.handlers
        ]
        return ast.TryStar(
            body=[ast.Pass()], handlers=handlers, orelse=[], finalbody=[],
        )
    return None


def _unparse_stmts(stmts: list[ast.stmt]) -> str | None:
    """Unparse a list of statements to source. Returns None on failure."""
    mod = ast.fix_missing_locations(ast.Module(body=stmts, type_ignores=[]))
    try:
        return ast.unparse(mod)
    except Exception:
        return None


class _RecursiveSegmenter(ast.NodeVisitor):
    """
    Custom AST visitor that recursively segments a list of statements into
    CODE blocks following the four rules:
      1. Simple State Rule
      2. Control-Flow Boundary Rule
      3. elif Chain Rolling
      4. Complexity Budget
    """

    def __init__(self) -> None:
        self.blocks: list[dict] = []

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def visit_stmts(self, stmts: list[ast.stmt]) -> None:
        """Process a flat list of statements, grouping simples and splitting compounds."""
        simple_buf: list[ast.stmt] = []

        for stmt in stmts:
            if isinstance(stmt, SIMPLE_TYPES):
                simple_buf.append(stmt)
            elif isinstance(stmt, COMPOUND_TYPES):
                if simple_buf:
                    self._emit(simple_buf)
                    simple_buf = []
                self._visit_compound(stmt)
            else:
                # Unknown node type — treat as simple to avoid silently dropping it
                simple_buf.append(stmt)

        if simple_buf:
            self._emit(simple_buf)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _emit(self, stmts: list[ast.stmt]) -> None:
        source = _unparse_stmts(stmts)
        if source is not None:
            self.blocks.append({"type": "CODE", "code": source})

    def _visit_compound(self, node: ast.stmt) -> None:
        """Apply the Complexity Budget check, then emit stub and recurse."""
        if _is_trivial_body(node):
            # Body is too small to recurse into — emit the whole compound monolithically
            self._emit([node])
            return

        stub = _make_stub(node)
        if stub is None:
            # Unrecognised compound — emit monolithically as a fallback
            self._emit([node])
            return

        stub_source = _unparse_stmts([stub])
        if stub_source is not None:
            self.blocks.append({"type": "CODE", "code": stub_source})

        self._recurse_bodies(node)

    def _recurse_bodies(self, node: ast.stmt) -> None:
        """Recurse into every statement-carrying sub-body of a compound node."""
        if isinstance(node, (ast.For, ast.AsyncFor, ast.While)):
            self.visit_stmts(node.body)
            if node.orelse:
                self.visit_stmts(node.orelse)

        elif isinstance(node, ast.If):
            # Recurse into the if-body
            self.visit_stmts(node.body)
            # Walk the elif/else chain — each branch body gets its own recursion
            orelse = node.orelse
            while orelse:
                if len(orelse) == 1 and isinstance(orelse[0], ast.If):
                    inner = orelse[0]
                    self.visit_stmts(inner.body)
                    orelse = inner.orelse
                else:
                    self.visit_stmts(orelse)
                    break

        elif isinstance(node, (ast.With, ast.AsyncWith)):
            self.visit_stmts(node.body)

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            self.visit_stmts(node.body)

        elif isinstance(node, ast.Try):
            self.visit_stmts(node.body)
            for handler in node.handlers:
                self.visit_stmts(handler.body)
            if node.orelse:
                self.visit_stmts(node.orelse)
            if node.finalbody:
                self.visit_stmts(node.finalbody)

        elif hasattr(ast, "TryStar") and isinstance(node, ast.TryStar):
            self.visit_stmts(node.body)
            for handler in node.handlers:
                self.visit_stmts(handler.body)
            if node.orelse:
                self.visit_stmts(node.orelse)
            if node.finalbody:
                self.visit_stmts(node.finalbody)


def segment_solution_grouped(code: str) -> list[dict] | None:
    """
    Segment a Python solution string into code blocks using recursive logical
    flattening.

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

    segmenter = _RecursiveSegmenter()
    segmenter.visit_stmts(body)

    blocks = segmenter.blocks
    n_blocks = len(blocks)
    if n_blocks < 2 or n_blocks > 15:
        return None

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
