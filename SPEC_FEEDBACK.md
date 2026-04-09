# Icechunk Spec Feedback

Gaps, ambiguities, and undocumented behavior discovered while implementing
icepyck — a clean-room Python Icechunk V2 client built from the spec alone.

These are filed as feedback for the icechunk spec authors. Each item made
implementation harder and required trial-and-error or reverse engineering
from binary test data.

---

## V1 vs V2 confusion

1. **V2 repo file vs V1 ref files** — The spec describes a V1 layout with
   per-branch ref files (`refs/branch.$NAME/ref.json`). V2 uses a single
   `repo` FlatBuffer file containing all branches, tags, and snapshots.
   The spec doesn't clearly distinguish which sections apply to which version.

2. **Snapshot parent tracking: parent_id vs parent_offset** — The spec says
   snapshots have a `parent_id` field. But in V2, parent is tracked via the
   repo file's `snapshots[].parent_offset` index field. The spec says "This
   is left empty on V2 spec snapshots" but doesn't explain the V2 alternative.

3. **Tag tombstone semantics** — The spec mentions tag tombstone files
   (`ref.json.deleted`) for V1. In V2, deleted tags go in the repo file's
   `deleted_tags` string vector. This V2 behavior isn't in the spec.

## Undocumented fields and types

4. **FileType byte values are wrong** — Spec says TransactionLog=5,
   RepoInfo=4. Actual binary data uses TransactionLog=4, RepoInfo=6.

5. **Initial snapshot well-known ID** — The first snapshot always has ID
   `1CECHNKREP0F1RSTCMT0` (hex `0b1cc8d6787580f0e33a6534`). This is a
   convention visible in all repos but not mentioned in the spec.

6. **NodeSnapshot user_data contents** — The spec doesn't say that
   `user_data` contains the complete `zarr.json` as UTF-8 bytes.

7. **Update tracking (latest_updates)** — The repo file has a
   `latest_updates` field with union types (BranchCreatedUpdate,
   NewCommitUpdate, etc.). Not mentioned in the spec — discovered from the
   FlatBuffer schema only.

8. **RepoStatus field** — Required field in the repo file with availability
   enum (Online/ReadOnly/Offline). Not mentioned in spec.

## Ambiguous semantics

9. **Repo file parent_offset semantics** — The field is called "parent
   offset" but is actually an ABSOLUTE index into the snapshots array, not a
   relative offset. Value of -1 means no parent (root snapshot), but this
   default isn't documented.

10. **Crockford Base32 padding direction** — The spec doesn't say which end
    gets zero-padded. It's RIGHT-aligned (LSB), which is non-obvious.

11. **ManifestFileInfoV2 purpose/requirements** — The snapshot contains a
    `manifest_files_v2` list but the spec doesn't say whether it's required,
    what happens if it's missing, or how it's used during reads.

12. **Chunk ID generation strategy** — The spec doesn't say how to generate
    ObjectId12 for chunks/manifests/snapshots. Random? Content-hashed?
    Time-based?

## Transaction log gaps

13. **Transaction log: required or optional?** — Unclear whether writing a
    transaction log is mandatory for a valid commit. The spec says they're
    "an optimization" but also lists them in the write algorithm.

14. **Transaction log for initial snapshot** — The spec's init algorithm
    only says "creating a new empty snapshot file and then creating the
    reference for branch main." No mention of a transaction log. But the
    write algorithm says to always write one. Should init write a txn log?
    This matters for conflict resolution — if a txn log is missing for a
    snapshot, does rebase skip it or fail?

## FlatBuffer enforcement

15. **`(required)` field enforcement** — Python FlatBuffers silently returns
    None/0/empty for missing required fields. Rust FlatBuffers validates
    them. Python-written files that omit required fields seem to work locally
    but fail when the Rust implementation reads them. The spec doesn't warn
    about this cross-language behavior difference.

## Missing from spec entirely

16. **Repo file atomicity** — The spec doesn't discuss how to atomically
    update the V2 repo file (the single mutable file). For S3, conditional
    PUT with ETags is the obvious answer, but this isn't specified.

17. **Behavioral spec compliance** — The spec defines algorithms
    (conditional updates, conflict resolution) but doesn't provide
    conformance tests or a test suite. Structural conformance (file layout)
    is easy to verify; behavioral conformance (atomic commits, conflict
    detection) requires integration testing.
