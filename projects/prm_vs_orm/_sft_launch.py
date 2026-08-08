"""Thin ``accelerate launch`` target that runs open-instruct SFT from a LOCAL base model.

``open_instruct/finetune.py`` unconditionally prefetches the base model with
``huggingface_hub.snapshot_download(model_path)`` (finetune.py:527). ``snapshot_download``
validates its first argument as a Hub repo id and rejects a local filesystem path
(``/tmp/olmo3_base`` fails ``validate_repo_id``), so pointing ``--model_name_or_path`` at a
converted local checkpoint would crash before training starts.

This launcher imports the finetune module, replaces the *module-local* ``snapshot_download``
binding with one that returns the path unchanged when it is an existing directory (and defers
to the real one otherwise), then parses the finetune CLI and calls ``main`` directly. Importing
the module does not parse args or train (both live under its ``if __name__ == "__main__"``),
so this is a clean, additive shim — no edit to ``finetune.py``. Calling ``main`` directly also
skips the ``__main__`` block's ``check_oe_eval_internal()``, which we do not need here.

Invoked by ``run_sft.py`` as::

    accelerate launch --mixed_precision bf16 --num_processes N \
        projects/prm_vs_orm/_sft_launch.py <finetune args...>
"""

from __future__ import annotations

import os
import sys
import traceback

import open_instruct.finetune as ft
from open_instruct.finetune import ArgumentParserPlus, FlatArguments, TokenizerConfig

# torch.distributed.elastic swallows the failing rank's Python traceback (error_file is
# N/A here) and prints only a per-rank exit-code summary, so the real exception never
# reaches the ~50-line log tail `edullm logs` returns. We re-emit it line-by-line with a
# unique prefix the outer wrapper (run_sft.py) greps out and prints LAST, guaranteeing the
# root cause lands in that tail instead of the distributed-teardown boilerplate.
_ERR_MARKER = "SFTERR| "

_real_snapshot_download = ft.snapshot_download


def _snapshot_download_or_local(repo_id, *args, **kwargs):
    """Return a local directory as-is; otherwise fall through to the real Hub download."""
    if isinstance(repo_id, str) and os.path.isdir(repo_id):
        return repo_id
    return _real_snapshot_download(repo_id, *args, **kwargs)


ft.snapshot_download = _snapshot_download_or_local


def main() -> None:
    parser = ArgumentParserPlus((FlatArguments, TokenizerConfig))
    args, tc = parser.parse_args_into_dataclasses()
    try:
        ft.main(args, tc)
    except BaseException:
        # re-emit the full traceback with a greppable per-line marker, then re-raise so the
        # process still exits non-zero. Only the failing rank(s) reach here; the rest are
        # SIGTERM'd by elastic and print nothing.
        tb = traceback.format_exc()
        sys.stderr.write("".join(f"{_ERR_MARKER}{ln}\n" for ln in tb.splitlines()))
        sys.stderr.flush()
        raise


if __name__ == "__main__":
    main()
