"""Writing a file so that a reader never sees it half-written."""

import os
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def replacing(path: str | Path) -> Iterator[Path]:
    """
    A temporary path next to ``path``, moved onto ``path`` once the block succeeds.

    Write the file to the path this yields. ``os.replace`` is atomic on one
    filesystem, so another process reading ``path`` sees the old file or the new one,
    never part of one, and a write that fails or is killed leaves the old file as it
    was. The temporary file is removed if the block raises. A ``path`` that is a
    symlink keeps it: the file it points to is the one replaced.

    Examples
    --------
    >>> import tempfile
    >>> from pathlib import Path
    >>> target = Path(tempfile.mkdtemp()) / "events.csv"
    >>> with replacing(target) as tmp:
    ...     _ = tmp.write_text("start,end")
    >>> target.read_text()
    'start,end'
    """
    # through a symlink to its target, so the link stays a link
    path = Path(path).resolve()
    # hidden, and not ending in the target's suffix, so a glob for the real files skips it
    tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
