<h1 align="center">
    <img src="https://raw.githubusercontent.com/metatensor/metatrain/refs/heads/main/docs/src/logo/metatrain-horizontal-dark.svg" alt="Metatensor logo" width="600"/>
</h1>

<h4 align="center">

[![tests status](https://img.shields.io/github/checks-status/metatensor/metatrain/main)](https://github.com/metatensor/metatrain/actions?query=branch%3Amain)
[![documentation](https://img.shields.io/badge/📚_documentation-latest-sucess)](https://metatensor.github.io/metatrain)
[![coverage](https://codecov.io/gh/metatensor/metatrain/branch/main/graph/badge.svg)](https://codecov.io/gh/metatensor/metatrain)
</h4>

<!-- marker-introduction -->

`metatrain` is a command line interface (CLI) to **train** and **evaluate** atomistic
models of various architectures. It features a common `yaml` option inputs to configure
training and evaluation. Trained models are exported as standalone files that can be
used directly in various molecular dynamics (MD) engines (e.g. `ASE`, `LAMMPS`, `i-PI`, 
`TorchSim`, `ESPResSo`,...) using the [metatomic](https://docs.metatensor.org/metatomic)
interface.

The idea behind `metatrain` is to have a general training hub that provides a
homogeneous environment and user interface, transforming every ML architecture into an
end-to-end model that can be connected to MD engines. Any custom architecture compatible
with [TorchScript](https://pytorch.org/docs/stable/jit.html) can be integrated into
`metatrain`, gaining automatic access to a training and evaluation interface, as well as
compatibility with various MD engines.

<!-- marker-architectures -->

# List of Implemented Architectures

Currently `metatrain` supports the following architectures for building an atomistic
model:

| Name                                     | Description                                                                                                                          |
|------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| [PET][arch-pet]                          | Point Edge Transformer (PET), interatomic machine learning potential                                                                 |
| [SOAP-BPNN][arch-soap_bpnn]              | A Behler-Parrinello neural network with SOAP features                                                                                |
| [MACE][arch-mace]                        | A higher order equivariant message passing neural network.                                                                           |
| [PhACE][arch-phace]                      | SO(3)-equivariant message-passing model with physical radial functions and fast tensor products.                                     |
| [GAP][arch-gap]                          | Sparse Gaussian Approximation Potential (GAP) using Smooth Overlap of Atomic Positions (SOAP).                                       |
| [FlashMD][arch-flashmd]                  | An architecture for the direct prediction of molecular dynamics                                                                      |

<!-- marker-arch-links -->

<!-- links for the different architectures. To be replaced if we are building the docs locally or
on a PR, since the docs use this README file directly.-->
[arch-flashmd]: https://docs.metatensor.org/metatrain/latest/architectures/generated/flashmd.html
[arch-gap]: https://docs.metatensor.org/metatrain/latest/architectures/generated/gap.html
[arch-mace]: https://docs.metatensor.org/metatrain/latest/architectures/generated/mace.html
[arch-pet]: https://docs.metatensor.org/metatrain/latest/architectures/generated/pet.html
[arch-phace]: https://docs.metatensor.org/metatrain/latest/architectures/generated/phace.html
[arch-soap_bpnn]: https://docs.metatensor.org/metatrain/latest/architectures/generated/soap_bpnn.html

<!-- marker-pet-head-config -->

# Configurable PET Prediction Heads

This fork adds independently configurable prediction head MLPs for the
node and edge branches of the PET architecture. The heads sit between the
transformer backbone and the final linear projection for each target.

**Node head** parameters:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `node_head_num_layers` | `int` | `2` | Number of hidden layers in the node head MLP. |
| `node_head_activation` | `str` | `"SiLU"` | Activation function. Supported: `SiLU`, `GELU`, `ReLU`, `Tanh`. |
| `node_head_dropout` | `float` | `0.0` | Dropout probability after each activation. |

**Edge head** parameters:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `edge_head_num_layers` | `int` | `2` | Number of hidden layers in the edge head MLP. |
| `edge_head_activation` | `str` | `"SiLU"` | Activation function. Supported: `SiLU`, `GELU`, `ReLU`, `Tanh`. |
| `edge_head_dropout` | `float` | `0.0` | Dropout probability after each activation. |

Default values reproduce the original PET heads exactly, so existing
checkpoints load without any changes.

### When to change these settings

- **`*_num_layers`** -- increase (e.g. 3--4) if the backbone is pre-trained
  and frozen and you need a more expressive readout; decrease to 1 for a
  near-linear probe. Node heads may benefit from more depth since node features
  already contain aggregated neighborhood information.
- **`*_activation`** -- `SiLU` (default) works well in most cases. `GELU` is
  a common alternative in transformer-based models.
- **`*_dropout`** -- start with 0.1--0.2 when fine-tuning on small datasets
  to reduce overfitting; keep at 0.0 for large-scale training from scratch.

### Example YAML configuration

```yaml
architecture:
  name: pet
  model:
    d_pet: 128
    d_head: 128
    # deeper node head with regularization
    node_head_num_layers: 3
    node_head_activation: GELU
    node_head_dropout: 0.1
    # simpler edge head
    edge_head_num_layers: 2
    edge_head_activation: SiLU
    edge_head_dropout: 0.0
```

### Architecture diagram

```
                   PET backbone (CartesianTransformer layers)
                                    |
                     +--------------+--------------+
                     |                             |
               node features                 edge features
               [N, d_node]                   [N, M, d_pet]
                     |                             |
              +-----------+                 +-----------+
              | node head |                 | edge head |    <-- independently configurable
              +-----------+                 +-----------+
                     |                             |
              node_last_layer              edge_last_layer   <-- Linear(d_head, output_dim)
                     |                             |
                     |                    * cutoff, sum over neighbors
                     |                             |
                     +------------- + -------------+
                                    |
                            sum over atoms
                                    |
                              prediction
```

### Repacking a pre-trained checkpoint with new heads

If you have a pre-trained PET checkpoint and want to swap the head
architecture (e.g. deeper MLP, different activation), use `mtt repack`.
It keeps the backbone weights intact and randomly initializes the new heads.

**From a YAML config** (reads `architecture.model` section):

```bash
mtt repack pretrained.ckpt -o repacked.ckpt -c train.yaml
```

**From CLI flags:**

```bash
mtt repack pretrained.ckpt -o repacked.ckpt \
    --node-head-num-layers 3 \
    --node-head-activation GELU \
    --node-head-dropout 0.1
```

**Both** (YAML as base, CLI flags override):

```bash
mtt repack pretrained.ckpt -o repacked.ckpt -c train.yaml --edge-head-dropout 0.2
```

Then fine-tune using the standard metatrain workflow:

```yaml
# train.yaml
architecture:
  name: pet
  model:
    node_head_num_layers: 3
    node_head_activation: GELU
    node_head_dropout: 0.1
  training:
    finetune:
      read_from: repacked.ckpt
      method: full   # or "heads" to only train the heads
```

```bash
mtt train train.yaml
```

> The module can also be invoked directly as
> `python -m metatrain.pet.repack_checkpoint --input ... --output ... --config ...`

<!-- marker-documentation -->

# Documentation

For details, tutorials, and examples, please visit our
[documentation](https://metatensor.github.io/metatrain/latest/).

<!-- marker-installation -->

# Installation

Install `metatrain` with pip:

```bash
pip install metatrain
```

Install specific models by specifying the model name. For example, to install the SOAP-BPNN model:

```bash
pip install metatrain[soap-bpnn]
```

We also offer a conda installation:

```bash
conda install -c conda-forge metatrain
```

> ⚠️ The conda installation does not install model-specific dependencies and will only
> work for architectures without optional dependencies such as PET.

After installation, you can use mtt from the command line to train your models!

<!-- marker-quickstart -->

# Quickstart

To train a model, use the following command:

```bash
mtt train options.yaml
```

Where options.yaml is a configuration file specifying training options. For example, the
following configuration trains a *SOAP-BPNN* model on the QM9 dataset:

```yaml
# architecture used to train the model
architecture:
  name: soap_bpnn
training:
  num_epochs: 5  # a very short training run

# Mandatory section defining the parameters for system and target data of the training set
training_set:
  systems: "qm9_reduced_100.xyz"  # file where the positions are stored
  targets:
    energy:
      key: "U0"      # name of the target value
      unit: "eV"     # unit of the target value

test_set: 0.1        # 10% of the training_set are randomly split for test
validation_set: 0.1  # 10% of the training_set are randomly split for validation
```

<!-- marker-shell -->

# Shell Completion

`metatrain` comes with completion definitions for its commands for bash and zsh. You
must manually configure your shell to enable completion support.

To make the completions available, source the definitions in your shell’s startup file
(e.g., `~/.bash_profile`, `~/.zshrc`, or `~/.profile`):

```bash
source $(mtt --shell-completion)
```

<!-- marker-issues -->

# Having problems or ideas?

Having a problem with metatrain? Please let us know by submitting an issue.

Submit new features or bug fixes through a pull request.

<!-- marker-contributing -->

# Contributors

Thanks goes to all people who make metatrain possible:

[![Contributors](https://contrib.rocks/image?repo=metatensor/metatrain)](https://github.com/metatensor/metatrain/graphs/contributors)

# Citing metatrain

If you found ``metatrain`` useful, you can cite its pre-print
(<https://doi.org/10.48550/arXiv.2508.15704>) as

```
@misc{metatrain,
title = {Metatensor and Metatomic: Foundational Libraries for Interoperable Atomistic
Machine Learning},
shorttitle = {Metatensor and Metatomic},
author = {Bigi, Filippo and Abbott, Joseph W. and Loche, Philip and Mazitov, Arslan
and Tisi, Davide and Langer, Marcel F. and Goscinski, Alexander and Pegolo, Paolo
and Chong, Sanggyu and Goswami, Rohit and Chorna, Sofiia and Kellner, Matthias and
Ceriotti, Michele and Fraux, Guillaume},
year = {2025},
month = aug,
publisher = {arXiv},
doi = {10.48550/arXiv.2508.15704},
}
```
