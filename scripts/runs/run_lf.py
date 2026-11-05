#!/usr/bin/env python
"""Launch LlamaFactory training with torchturbo from the torchturbo repo.

torchturbo-specific arguments use the ``--tt_*`` prefix and are stripped
from argv before forwarding to LlamaFactory.  For example::

    torchrun ... run_lf.py --tt_sp_size 2 --tt_ep_size 8 --stage sft ...

The ``--tt_*`` flags replace the ``TORCH_TURBO_*`` environment variables.
Priority (lowest → highest): defaults → LF args → ``--tt_*`` CLI args → env vars.

Usage:
    torchrun --nproc_per_node=8 scripts/runs/run_lf.py \\
        --tt_sp_size 2 --tt_ep_size 8 \\
        --stage sft \\
        --model_name_or_path /path/to/model \\
        --dataset your_dataset \\
        --template qwen3 \\
        --output_dir ./output/my_run
"""

from __future__ import annotations

import os
import sys


# Ensure torchturbo is importable when running without pip install -e
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_REPO_SRC = os.path.join(_REPO_ROOT, "src")
if _REPO_SRC not in sys.path:
    sys.path.insert(0, _REPO_SRC)


def _patch_npu_cross_entropy_ignore_index() -> None:
    import torch
    import torch.nn.functional as functional

    original = functional.cross_entropy

    def cross_entropy(
        input: torch.Tensor,
        target: torch.Tensor,
        weight: torch.Tensor | None = None,
        size_average: bool | None = None,
        ignore_index: int = -100,
        reduce: bool | None = None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> torch.Tensor:
        if input.device.type != "npu" or ignore_index >= 0:
            return original(
                input, target, weight, size_average, ignore_index, reduce, reduction, label_smoothing
            )

        class_count = input.shape[1]
        logits = input.movedim(1, -1).reshape(-1, class_count)
        labels = target.reshape(-1)
        valid = labels != ignore_index
        if not torch.any(~valid):
            return original(
                input, target, weight, size_average, ignore_index, reduce, reduction, label_smoothing
            )
        logits = logits[valid]
        labels = labels[valid]
        if labels.numel() == 0:
            return input.sum() * 0
        return original(logits, labels, weight, size_average, -100, reduce, reduction, label_smoothing)

    functional.cross_entropy = cross_entropy

    def cross_entropy_module_forward(module, input, target):
        return cross_entropy(
            input,
            target,
            module.weight,
            None,
            module.ignore_index,
            None,
            module.reduction,
            module.label_smoothing,
        )

    torch.nn.CrossEntropyLoss.forward = cross_entropy_module_forward


def auto_detect_dataset_dir() -> str | None:
    """Auto-detect LlamaFactory's data/ directory from the package location."""
    try:
        import llamafactory
    except ImportError:
        return None
    pkg_dir = os.path.dirname(llamafactory.__file__)
    for _ in range(4):
        pkg_dir = os.path.dirname(pkg_dir)
        candidate = os.path.join(pkg_dir, "data")
        if os.path.isfile(os.path.join(candidate, "dataset_info.json")):
            return candidate
    return None


def main():
    _patch_npu_cross_entropy_ignore_index()
    import torchturbo
    from torchturbo.engine.args import parse_torchturbo_args, set_torchturbo_args

    # Explicitly activate the LlamaFactory adapter. ``import torchturbo`` only
    # auto-activates when a single framework is detectable; in a dev image with
    # both ms-swift and LlamaFactory installed that is ambiguous (availability
    # can't tell which one this process will run — torchturbo is imported
    # before the framework loads), so activate LF here unambiguously.
    #
    # Must run BEFORE ``from llamafactory.train.tuner import run_exp`` below so
    # the adapter's RunExpPatch has monkey-patched ``tuner.run_exp`` by the
    # time ``run_exp`` is bound into this module's namespace.
    torchturbo.activate("llamafactory_v0")

    # Parse --tt_* flags and strip them from argv
    argv = sys.argv[1:]
    tt_args, remaining_argv = parse_torchturbo_args(argv)
    set_torchturbo_args(tt_args)

    # Auto-inject dataset_dir if not explicitly provided
    if "--dataset_dir" not in remaining_argv:
        dataset_dir = auto_detect_dataset_dir()
        if dataset_dir is not None:
            remaining_argv.extend(["--dataset_dir", dataset_dir])

    from llamafactory.train.tuner import run_exp

    run_exp(args=remaining_argv)


if __name__ == "__main__":
    main()
