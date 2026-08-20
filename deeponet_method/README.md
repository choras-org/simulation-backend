# DeepONet

CHORAS interface for DeepONet-based acoustic impulse-response prediction. Wraps
the [`deeponet_acoustics`](https://github.com/dtu-act/deeponet-acoustic-wave-prop)
training/inference and the sibling DG simulator for high-fidelity training data.

## Install

```bash
cd deeponet_method
uv sync --extra dev
```

## Run headless

`HEADLESS=true` leaves the user-facing JSON read-only and writes results to a
side file under the module directory. Without it, the input JSON is overwritten
in place (CHORAS flow).

```bash
HEADLESS=true \
JSON_PATH=tests/test_input_deeponet.json \
  uv run python -m deeponet_interface
```

Geometry paths inside the JSON are resolved relative to the JSON's directory.
Intermediate artifacts (DG NPZ, HDF5 train/val splits, model checkpoints) land
under `deeponet_interface/tmp/deeponet/`.

## Inspect training with TensorBoard

```bash
uv run tensorboard --logdir deeponet_interface/tmp/deeponet/results
```

Then open <http://localhost:6006>. Scalars: `Loss/train/loss`, `Loss/val/loss`,
`Loss/learning_rate`. Safe to launch before or during a training run; events are
picked up live.

## Tests

```bash
uv run pytest
```
