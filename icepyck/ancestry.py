"""Rich-formatted ancestry display for Icechunk repositories.

Provides both a library function and CLI entry point for displaying
the commit ancestry of a branch.

CLI usage::

    uv run python -m icepyck.ancestry /path/to/repo
    uv run python -m icepyck.ancestry /path/to/repo --branch dev
    icepyck-log /path/to/repo
"""

from __future__ import annotations

import argparse
from pathlib import Path


def print_log(
    path: str | Path,
    branch: str = "main",
    *,
    max_entries: int = 0,
) -> None:
    """Print a rich-formatted commit log for a branch.

    Parameters
    ----------
    path : str or Path
        Repository path.
    branch : str
        Branch to show ancestry for.
    max_entries : int
        Maximum number of entries to show (0 = all).
    """
    from rich.console import Console
    from rich.text import Text
    from rich.tree import Tree

    import icepyck

    console = Console()
    repo = icepyck.open(path)
    log = repo.log(branch)

    if max_entries > 0:
        log = log[:max_entries]

    if not log:
        console.print(f"[dim]No commits on branch {branch!r}[/dim]")
        return

    console.print()
    title = Text.assemble(
        Text(" ", style="bold"),
        Text(branch, style="bold cyan"),
        Text(f"  ({len(log)} commits)", style="dim"),
    )
    tree = Tree(title)

    for i, entry in enumerate(log):
        snap_id = str(entry["id"])
        short_id = snap_id[:12]
        message = str(entry["message"]) or "(no message)"
        time_str = str(entry["time"])

        label = Text()
        label.append(short_id, style="bold yellow")
        label.append("  ", style="")
        label.append(message, style="bold white" if i == 0 else "white")
        label.append(f"  {time_str}", style="dim")

        if i == 0:
            label.append("  (HEAD)", style="bold green")

        tree.add(label)

    console.print(tree)
    console.print()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for ancestry display."""
    parser = argparse.ArgumentParser(
        prog="icepyck-log",
        description="Show commit ancestry for an Icechunk repository branch.",
    )
    parser.add_argument(
        "path",
        metavar="REPO_PATH",
        help="Path to Icechunk repository",
    )
    parser.add_argument(
        "--branch",
        "-b",
        default="main",
        help="Branch to show (default: main)",
    )
    parser.add_argument(
        "-n",
        "--max",
        type=int,
        default=0,
        help="Maximum number of entries (default: all)",
    )
    args = parser.parse_args(argv)

    try:
        print_log(args.path, branch=args.branch, max_entries=args.max)
    except (FileNotFoundError, KeyError) as e:
        from rich.console import Console

        Console().print(f"[red]Error:[/red] {e}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
