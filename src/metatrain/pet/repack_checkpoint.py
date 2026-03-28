"""
Repack a PET checkpoint with a different head architecture.

Loads a pre-trained PET checkpoint, builds a new model whose backbone
hypers match the checkpoint but head hypers come from the user, copies
the backbone weights over, and saves the result as a new checkpoint
that is ready for standard ``mtt train`` fine-tuning.

Usage
-----

Read head configuration from a training YAML (same format as ``mtt train``):

.. code-block:: bash

    python -m metatrain.pet.repack_checkpoint \
        --input  pretrained.ckpt \
        --output repacked.ckpt \
        --config train.yaml

Or specify individual overrides via CLI flags:

.. code-block:: bash

    python -m metatrain.pet.repack_checkpoint \
        --input  pretrained.ckpt \
        --output repacked.ckpt \
        --node-head-num-layers 3 \
        --node-head-activation GELU

CLI flags take precedence over values from ``--config``.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Set

import torch
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

HEAD_PREFIXES = ("node_heads.", "edge_heads.", "node_last_layers.", "edge_last_layers.")

HEAD_HYPER_KEYS = {
    "node_head_num_layers",
    "node_head_activation",
    "node_head_dropout",
    "edge_head_num_layers",
    "edge_head_activation",
    "edge_head_dropout",
}


def _is_head_key(key: str) -> bool:
    return any(key.startswith(p) for p in HEAD_PREFIXES)


def _extract_head_hypers_from_yaml(path: str) -> Dict[str, Any]:
    """Extract head-related hypers from a metatrain YAML config.

    Looks for keys under ``architecture.model`` that match known
    head hyper names.
    """
    with open(path) as f:
        config = yaml.safe_load(f)

    model_section = config.get("architecture", {}).get("model", {})
    return {k: v for k, v in model_section.items() if k in HEAD_HYPER_KEYS}


def repack(
    input_path: str,
    output_path: str,
    head_overrides: Dict[str, object],
) -> None:
    from metatrain.pet.model import PET

    logger.info("Loading checkpoint from %s", input_path)
    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)

    if hasattr(PET, "upgrade_checkpoint"):
        checkpoint = PET.upgrade_checkpoint(checkpoint)

    model_data = checkpoint["model_data"]
    old_hypers = dict(model_data["model_hypers"])

    new_hypers = {**old_hypers, **head_overrides}
    model_data["model_hypers"] = new_hypers

    logger.info("Creating new model with updated head hypers")
    model = PET(
        hypers=new_hypers,
        dataset_info=model_data["dataset_info"],
    )

    old_state = checkpoint["best_model_state_dict"]
    old_state.pop("finetune_config", None)

    backbone_state = {k: v for k, v in old_state.items() if not _is_head_key(k)}
    new_state = model.state_dict()

    loaded: Set[str] = set()
    skipped_shape: Set[str] = set()
    for key, value in backbone_state.items():
        if key in new_state:
            if new_state[key].shape == value.shape:
                new_state[key] = value
                loaded.add(key)
            else:
                skipped_shape.add(key)

    model.load_state_dict(new_state)

    head_keys = [k for k in new_state if _is_head_key(k)]

    logger.info(
        "Backbone: loaded %d/%d parameters",
        len(loaded),
        len(backbone_state),
    )
    if skipped_shape:
        logger.warning("Skipped (shape mismatch): %s", sorted(skipped_shape))
    logger.info("Head parameters (randomly initialized): %d", len(head_keys))

    new_checkpoint = model.get_checkpoint()
    new_checkpoint["model_ckpt_version"] = checkpoint["model_ckpt_version"]
    new_checkpoint["metadata"] = checkpoint.get("metadata", new_checkpoint["metadata"])

    logger.info("Saving repacked checkpoint to %s", output_path)
    torch.save(new_checkpoint, output_path)
    logger.info("Done")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repack a PET checkpoint with different head architecture",
    )
    parser.add_argument("--input", required=True, help="Path to source .ckpt")
    parser.add_argument("--output", required=True, help="Path for repacked .ckpt")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to a metatrain YAML config; head hypers are read from "
        "architecture.model section",
    )

    parser.add_argument("--node-head-num-layers", type=int, default=None)
    parser.add_argument("--node-head-activation", type=str, default=None)
    parser.add_argument("--node-head-dropout", type=float, default=None)
    parser.add_argument("--edge-head-num-layers", type=int, default=None)
    parser.add_argument("--edge-head-activation", type=str, default=None)
    parser.add_argument("--edge-head-dropout", type=float, default=None)

    args = parser.parse_args()

    if not Path(args.input).exists():
        logger.error("Input checkpoint not found: %s", args.input)
        sys.exit(1)

    overrides: Dict[str, object] = {}

    if args.config is not None:
        if not Path(args.config).exists():
            logger.error("Config file not found: %s", args.config)
            sys.exit(1)
        overrides.update(_extract_head_hypers_from_yaml(args.config))
        logger.info("Read from %s: %s", args.config, overrides)

    cli_mapping = {
        "node_head_num_layers": args.node_head_num_layers,
        "node_head_activation": args.node_head_activation,
        "node_head_dropout": args.node_head_dropout,
        "edge_head_num_layers": args.edge_head_num_layers,
        "edge_head_activation": args.edge_head_activation,
        "edge_head_dropout": args.edge_head_dropout,
    }
    for key, value in cli_mapping.items():
        if value is not None:
            overrides[key] = value

    if not overrides:
        logger.error(
            "No head overrides found. Use --config with a YAML file "
            "and/or --node-head-* / --edge-head-* flags"
        )
        sys.exit(1)

    logger.info("Final head overrides: %s", overrides)
    repack(args.input, args.output, overrides)


if __name__ == "__main__":
    main()
