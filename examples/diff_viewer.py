# /// script
# requires-python = ">=3.12"
# dependencies = ["icepyck", "rich"]
# [tool.uv]
# sources = {icepyck = {path = ".."}}
# ///
"""Example: view the diff between two snapshots in the basic test repo."""

from icepyck.diff import diff_snapshots
from icepyck.diff_display import display_diff

# Compare the parent of main with main in the basic test repo
diff = diff_snapshots("test-repos/basic", "main~1", "main")
display_diff(diff)
