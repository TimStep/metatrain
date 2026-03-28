import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

from .formatter import CustomHelpFormatter


def _add_repack_parser(subparser: argparse._SubParsersAction) -> None:
    if repack_model.__doc__ is not None:
        description = repack_model.__doc__.split(r":param")[0]
    else:
        description = None

    parser = subparser.add_parser(
        "repack",
        description=description,
        formatter_class=CustomHelpFormatter,
    )
    parser.set_defaults(callable="repack_model")

    parser.add_argument(
        "input",
        type=str,
        help="Path to the source PET checkpoint (.ckpt)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Path for the repacked checkpoint",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
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


def _prepare_repack_args(args: argparse.Namespace) -> None:
    if not Path(args.input).exists():
        logging.error("Input checkpoint not found: %s", args.input)
        sys.exit(1)

    if args.config is not None and not Path(args.config).exists():
        logging.error("Config file not found: %s", args.config)
        sys.exit(1)


def repack_model(
    input: str,
    output: str,
    config: str = None,
    node_head_num_layers: int = None,
    node_head_activation: str = None,
    node_head_dropout: float = None,
    edge_head_num_layers: int = None,
    edge_head_activation: str = None,
    edge_head_dropout: float = None,
    **kwargs: Any,
) -> None:
    """Repack a PET checkpoint with a different head architecture.

    Loads a pre-trained PET checkpoint, keeps the backbone weights, and
    creates a new checkpoint with randomly initialized heads matching
    the requested architecture. The result can be used directly with
    ``mtt train`` for fine-tuning.

    :param input: Path to the source checkpoint.
    :param output: Path for the repacked checkpoint.
    :param config: Optional path to a metatrain YAML config file.
    :param node_head_num_layers: Number of layers in node head MLP.
    :param node_head_activation: Activation for node head MLP.
    :param node_head_dropout: Dropout for node head MLP.
    :param edge_head_num_layers: Number of layers in edge head MLP.
    :param edge_head_activation: Activation for edge head MLP.
    :param edge_head_dropout: Dropout for edge head MLP.
    """
    from ..pet.repack_checkpoint import (
        _extract_head_hypers_from_yaml,
        repack,
    )

    overrides: Dict[str, object] = {}

    if config is not None:
        overrides.update(_extract_head_hypers_from_yaml(config))
        logging.info("Read from %s: %s", config, overrides)

    cli_mapping = {
        "node_head_num_layers": node_head_num_layers,
        "node_head_activation": node_head_activation,
        "node_head_dropout": node_head_dropout,
        "edge_head_num_layers": edge_head_num_layers,
        "edge_head_activation": edge_head_activation,
        "edge_head_dropout": edge_head_dropout,
    }
    for key, value in cli_mapping.items():
        if value is not None:
            overrides[key] = value

    if not overrides:
        logging.error(
            "No head overrides found. Use --config with a YAML file "
            "and/or --node-head-* / --edge-head-* flags"
        )
        sys.exit(1)

    logging.info("Final head overrides: %s", overrides)
    repack(input, output, overrides)
