"""Split a spend-log flush into statements the Prisma query engine can afford.

The query engine is a separate Rust process whose resident memory is a
high-water mark: it grows with the payload of the largest single statement it
is asked to execute and glibc never returns that memory to the OS, so a pod's
memory floor ratchets up to its worst-ever write and stays there for the life
of the worker. Under ``store_prompts_in_spend_logs`` a single spend-log row
carries the full prompt and response, so a fixed 1000-row ``create_many``
hands the engine tens of megabytes in one statement and permanently costs
hundreds of megabytes of RSS, which is what makes memory-based autoscaling
read the wrong number.

Bounding each statement by payload size instead caps that floor. Row-count
batching alone cannot: the same 1000 rows range from well under a megabyte
(spend counters only) to tens of megabytes (prompts stored), and only the
byte budget tracks what the engine actually allocates.
"""

import json
from collections.abc import Iterator, Mapping, Sequence

SpendLogRow = Mapping[str, object]


def _value_payload_bytes(value: object) -> int:
    """Bytes ``value`` contributes to the encoded write statement.

    Values are measured as they will be encoded rather than as they are held
    in memory, because the two differ by enough to defeat the budget. Counting
    characters under-measures a prompt in a non-Latin script by its
    bytes-per-character factor, and even an all-ASCII prompt grows when JSON
    escapes its quotes, backslashes and newlines (about 18% for a realistic
    stored prompt, and up to double for escape-dense content). Serializing
    settles both: ``json.dumps`` escapes non-ASCII to ``\\uXXXX`` and defaults
    to ASCII output, so the length it reports is a byte count that never
    under-states the wire size.

    Rows arrive after ``jsonify_object``, which converts dicts to strings but
    leaves lists alone, and ``messages`` / ``response`` are typed to allow a
    list, so both shapes are measured the same way here. Scalars are
    negligible next to the blobs and are not counted.
    """
    if isinstance(value, (str, list, tuple, dict)):
        try:
            return len(json.dumps(value, default=str))
        except (TypeError, ValueError):
            return 0
    return 0


def _row_payload_bytes(row: SpendLogRow) -> int:
    """Approximate the bytes this row contributes to the write statement."""
    return sum(_value_payload_bytes(value) for value in row.values())


def spend_log_write_batches(
    rows: Sequence[SpendLogRow],
    max_bytes: int,
) -> Iterator[Sequence[SpendLogRow]]:
    """Yield consecutive slices of ``rows`` whose payload fits ``max_bytes``.

    Slices preserve input order and together cover every row exactly once. A
    row larger than ``max_bytes`` on its own is yielded alone rather than
    dropped: the budget is a memory guardrail, not an admission filter, and
    losing spend data to protect RSS would be the worse failure.
    """
    sizes = tuple(_row_payload_bytes(row) for row in rows)
    start = 0
    while start < len(rows):
        end = start + 1
        used = sizes[start]
        while end < len(rows) and used + sizes[end] <= max_bytes:
            used += sizes[end]
            end += 1
        yield rows[start:end]
        start = end
