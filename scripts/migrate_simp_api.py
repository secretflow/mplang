#!/usr/bin/env python3
"""
Migrate from deprecated simp.run/runAt to new API.

- simp.run(fn) -> run_jax(fn, ...)
- simp.runAt(rank, fn) -> run_jax_at(rank, fn, ...) or run_op_at(rank, op, ...)
"""

import sys
from pathlib import Path


def migrate_file(filepath: Path) -> tuple[bool, int]:
    """Migrate a single file. Returns (changed, num_replacements)."""
    content = filepath.read_text()
    original = content
    count = 0

    # Pattern 1: simp.runAt(rank, callable)(...) -> run_op_at(rank, callable, ...)
    # Handle nested parens in callable (e.g., lambda, partial, etc.)
    def find_matching_paren(s, start):
        """Find matching closing paren starting from position start."""
        depth = 1
        i = start
        while i < len(s) and depth > 0:
            if s[i] == "(":
                depth += 1
            elif s[i] == ")":
                depth -= 1
            i += 1
        return i - 1 if depth == 0 else -1

    # Replace simp.runAt
    pos = 0
    new_content = []
    last_pos = 0

    while True:
        pos = content.find("simp.runAt(", pos)
        if pos == -1:
            break

        # Find the end of runAt(...)'s arguments
        rank_start = pos + len("simp.runAt(")
        comma_pos = content.find(",", rank_start)
        if comma_pos == -1:
            pos += 1
            continue

        rank = content[rank_start:comma_pos].strip()
        callable_start = comma_pos + 1

        # Find the matching paren for runAt
        runAt_end = find_matching_paren(content, rank_start - 1)
        if runAt_end == -1:
            pos += 1
            continue

        callable_part = content[callable_start:runAt_end].strip()

        # Check if there's an immediate invocation ()
        check_pos = runAt_end + 1
        while check_pos < len(content) and content[check_pos].isspace():
            check_pos += 1

        if check_pos < len(content) and content[check_pos] == "(":
            # Find the matching paren for the invocation
            invoke_end = find_matching_paren(content, check_pos)
            if invoke_end != -1:
                args = content[check_pos + 1 : invoke_end].strip()
                new_content.append(content[last_pos:pos])
                if args:
                    new_content.append(f"run_op_at({rank}, {callable_part}, {args})")
                else:
                    new_content.append(f"run_op_at({rank}, {callable_part})")
                count += 1
                last_pos = invoke_end + 1
                pos = invoke_end + 1
                continue

        pos += 1

    new_content.append(content[last_pos:])
    content = "".join(new_content)

    # Pattern 2: simp.run(fn)(...) -> run_jax(fn, ...)
    # Similar logic for simp.run
    pos = 0
    new_content = []
    last_pos = 0

    while True:
        pos = content.find("simp.run(", pos)
        if pos == -1:
            break

        # Find the matching paren for run
        fn_start = pos + len("simp.run(")
        run_end = find_matching_paren(content, fn_start - 1)
        if run_end == -1:
            pos += 1
            continue

        fn_part = content[fn_start:run_end].strip()

        # Check if there's an immediate invocation ()
        check_pos = run_end + 1
        while check_pos < len(content) and content[check_pos].isspace():
            check_pos += 1

        if check_pos < len(content) and content[check_pos] == "(":
            # Find the matching paren for the invocation
            invoke_end = find_matching_paren(content, check_pos)
            if invoke_end != -1:
                args = content[check_pos + 1 : invoke_end].strip()
                new_content.append(content[last_pos:pos])
                if args:
                    new_content.append(f"run_jax({fn_part}, {args})")
                else:
                    new_content.append(f"run_jax({fn_part})")
                count += 1
                last_pos = invoke_end + 1
                pos = invoke_end + 1
                continue

        pos += 1

    new_content.append(content[last_pos:])
    content = "".join(new_content)

    # Add import if we made changes and it's not already there
    if content != original:
        # Check if we need to add imports
        needs_run_jax = (
            "run_jax(" in content and "from mplang.simp.api import" not in content
        )
        needs_run_op_at = (
            "run_op_at(" in content and "from mplang.simp.api import" not in content
        )

        if needs_run_jax or needs_run_op_at:
            # Find the import block
            lines = content.split("\n")
            import_idx = -1

            # Find where to insert the import (after existing mplang imports or at the top)
            for i, line in enumerate(lines):
                if "from mplang" in line or "import mplang" in line:
                    import_idx = i + 1
                elif (
                    import_idx > 0
                    and line
                    and not line.startswith(("import ", "from "))
                ):
                    break

            if import_idx < 0:
                # No mplang imports found, add after standard library imports
                for i, line in enumerate(lines):
                    if line and not line.startswith((
                        "import ",
                        "from ",
                        "#",
                        '"""',
                        "'''",
                    )):
                        import_idx = i
                        break

            imports_to_add = []
            if needs_run_jax:
                imports_to_add.append("run_jax")
            if needs_run_op_at:
                imports_to_add.append("run_op_at")

            import_line = f"from mplang.simp.api import {', '.join(imports_to_add)}"

            if import_idx >= 0:
                lines.insert(import_idx, import_line)
            else:
                lines.insert(0, import_line)

            content = "\n".join(lines)

    changed = content != original
    if changed:
        filepath.write_text(content)

    return changed, count


def main():
    root = Path(__file__).parent.parent

    # Process tutorials, tests, and examples
    dirs = [root / "tutorials", root / "tests", root / "examples"]

    total_files = 0
    total_changes = 0
    total_replacements = 0

    for dir_path in dirs:
        if not dir_path.exists():
            continue

        for py_file in dir_path.rglob("*.py"):
            changed, count = migrate_file(py_file)
            if changed:
                total_files += 1
                total_changes += 1
                total_replacements += count
                print(f"✓ {py_file.relative_to(root)}: {count} replacements")

    print("\nMigration complete:")
    print(f"  {total_files} files changed")
    print(f"  {total_replacements} total replacements")

    return 0 if total_files > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
