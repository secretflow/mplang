#!/usr/bin/env python3
"""
Simple find-replace for remaining simp.run/runAt calls.
This handles cases where the function is passed as an argument (not immediately invoked).
"""

import sys
from pathlib import Path


def process_file(filepath: Path) -> tuple[bool, int]:
    """Process a single file."""
    content = filepath.read_text()
    original = content
    count = 0

    # For simp.run that's not immediately invoked, just keep it for now
    # (these are usually passed as function arguments and need manual review)

    # Replace remaining simp.runAt(...) patterns - handle lambdas and other callables
    lines = content.split("\n")
    new_lines = []

    for line in lines:
        new_line = line
        # Skip comment-only lines
        if line.strip().startswith("#"):
            new_lines.append(line)
            continue

        # Replace simp.runAt with run_op_at - simple text replacement for remaining cases
        if "simp.runAt(" in new_line:
            new_line = new_line.replace("simp.runAt(", "run_op_at(")
            count += 1

        #  Replace remaining simp.run with run_jax
        if "simp.run(" in new_line:
            new_line = new_line.replace("simp.run(", "run_jax(")
            count += 1

        new_lines.append(new_line)

    content = "\n".join(new_lines)

    # Add imports if needed
    if content != original:
        needs_run_jax = "run_jax(" in content
        needs_run_op_at = "run_op_at(" in content

        has_import = "from mplang.simp.api import" in content

        if (needs_run_jax or needs_run_op_at) and not has_import:
            # Add import at appropriate location
            lines = content.split("\n")
            import_idx = 0

            # Find insertion point after existing imports
            for i, line in enumerate(lines):
                if line.startswith(("import ", "from ")) and "mplang" in line:
                    import_idx = i + 1
                elif (
                    import_idx > 0
                    and line
                    and not line.startswith(("import ", "from ", "#"))
                ):
                    break

            imports_to_add = []
            if needs_run_jax:
                imports_to_add.append("run_jax")
            if needs_run_op_at:
                imports_to_add.append("run_op_at")

            import_line = f"from mplang.simp.api import {', '.join(imports_to_add)}"
            lines.insert(import_idx, import_line)
            content = "\n".join(lines)
        elif (needs_run_jax or needs_run_op_at) and has_import:
            # Update existing import
            lines = content.split("\n")
            for i, line in enumerate(lines):
                if "from mplang.simp.api import" in line:
                    # Parse existing imports
                    parts = line.split("import", 1)
                    if len(parts) == 2:
                        existing = [x.strip() for x in parts[1].split(",")]
                        if needs_run_jax and "run_jax" not in existing:
                            existing.append("run_jax")
                        if needs_run_op_at and "run_op_at" not in existing:
                            existing.append("run_op_at")
                        lines[i] = f"{parts[0]}import {', '.join(sorted(existing))}"
                    break
            content = "\n".join(lines)

    changed = content != original
    if changed:
        filepath.write_text(content)

    return changed, count


def main():
    root = Path(__file__).parent.parent

    dirs = [root / "tutorials", root / "tests", root / "examples"]

    total_files = 0
    total_replacements = 0

    for dir_path in dirs:
        if not dir_path.exists():
            continue

        for py_file in dir_path.rglob("*.py"):
            changed, count = process_file(py_file)
            if changed:
                total_files += 1
                total_replacements += count
                print(f"✓ {py_file.relative_to(root)}: {count} replacements")

    print("\nMigration complete:")
    print(f"  {total_files} files changed")
    print(f"  {total_replacements} total replacements")

    return 0


if __name__ == "__main__":
    sys.exit(main())
