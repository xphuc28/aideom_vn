"""Count Python source lines for the AIDEOM-VN project.

The assignment requires at least 1,500 valid Python source lines across the
application, model modules, Streamlit pages, and tests. This script counts
physical lines in `.py` files under `src/`, `pages/`, `tests/`, and `app.py`.
It intentionally skips `__pycache__` and hidden directories.
"""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "pages",
    PROJECT_ROOT / "tests",
    PROJECT_ROOT / "app.py",
]


def iter_python_files() -> list[Path]:
    """Return sorted Python files included in the source-line count."""
    files: list[Path] = []
    for target in TARGETS:
        if target.is_file() and target.suffix == ".py":
            files.append(target)
        elif target.is_dir():
            files.extend(
                path
                for path in target.rglob("*.py")
                if "__pycache__" not in path.parts and not any(part.startswith(".") for part in path.parts)
            )
    return sorted(files)


def count_lines(path: Path) -> int:
    """Count physical lines in one Python file."""
    return len(path.read_text(encoding="utf-8").splitlines())


def build_line_report() -> tuple[list[dict[str, object]], int]:
    """Build per-file line counts and the project total."""
    rows = []
    total = 0
    for path in iter_python_files():
        line_count = count_lines(path)
        total += line_count
        rows.append(
            {
                "file": str(path.relative_to(PROJECT_ROOT)),
                "lines": line_count,
            }
        )
    return rows, total


def main() -> int:
    """Print the line-count report and return a shell status code."""
    rows, total = build_line_report()
    width = max(len(row["file"]) for row in rows) if rows else 10
    for row in rows:
        print(f"{row['file']:<{width}} {row['lines']:>5}")
    print("-" * (width + 7))
    print(f"{'TOTAL':<{width}} {total:>5}")
    return 0 if total >= 1500 else 1


if __name__ == "__main__":
    raise SystemExit(main())
