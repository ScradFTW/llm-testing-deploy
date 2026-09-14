# image-classifier — computer vision, train-on-GPU / serve-on-CPU

Live at: https://bradjobe.dev/image-classifier/

The fourth demo, and the first that isn't text: a small CNN trained from
scratch on CIFAR-10, exported to ONNX, and served with ONNX Runtime — no
PyTorch anywhere in production at all. Added specifically because the other three
demos are 100% text/NLP, and a computer-vision role should have a
computer-vision artifact, not just an LLM chat widget.

## Model

- **Data**: CIFAR-10 (60k 32×32 RGB images, 10 classes), via
  `torchvision.datasets.CIFAR10`.
- **Architecture**: a small custom CNN — 3 conv+batchnorm+pool blocks, 2
  FC layers, ~1M params (`train/train.py`). Not a pretrained backbone;
  trained from scratch.
- **Training**: 15 epochs with random-crop/flip augmentation and cosine LR
  decay, on a local GPU (RTX 3060) — about 90 seconds total.
- **Result**: **80.6% test accuracy** vs. a 10% majority-class baseline.
  Full per-class precision/recall/F1 and confusion matrix in
  `train/eval_report.txt`. Cats/dogs and the various vehicle-adjacent
  classes are the main confusions, as expected for a small CNN on 32×32
  images.

## Train on GPU, serve on CPU

The training script exports the trained model to ONNX
(`torch.onnx.export`) and immediately verifies it: it runs both the
original PyTorch model and the exported ONNX model on the same batch and
asserts the predicted classes match exactly before considering the export
good. Only then does the `.onnx` file, plus the normalization constants
and class list, get copied to the server.

`serve/app.py` loads that ONNX file with `onnxruntime` (CPU execution
provider) — no PyTorch, no CUDA, nothing GPU-related in the serving
container. Memory footprint at rest is about 50MB, inference is 1-5ms per
image. This is the real pattern for "train expensive, serve cheap": the
GPU-hours are spent once, offline, on hardware that has them; the
always-on production service runs on hardware that doesn't need them.

## Serving

Same pattern as every other service here: Flask + waitress in a container
on Cloud Run (own least-privilege IAM service account, no direct public
URL), reached through the shared Global Load Balancer with a Cloud Armor
rate limit in front of it (see `bradjobe-dev-infra`'s `cloud_run.tf` /
`cloud_armor.tf`). `/predict` takes a base64-encoded image (or a `data:` URL directly
from a `<input type=file>`/drag-and-drop, which is what the frontend
sends), resizes to 32×32, and returns the ranked class probabilities.
`/stats` and `/metrics` follow the same telemetry pattern as
`genre-classifier` and feed the same `/status` dashboard.

## Honest limitations

- CIFAR-10 is a toy, well-worn benchmark, not a real product dataset — the
  point here is demonstrating the train→export→serve pipeline and honest
  evaluation methodology, not proving vision research chops.
- This is a separate model behind a separate tool from the text pipeline —
  not a true vision-language model reasoning jointly over both. The
  `agent-orchestrator` doesn't call this tool (yet); extending it to accept
  an image alongside text is the natural next step to make the "reasoning
  agent" demo genuinely multi-modal rather than two demos that happen to
  share a dashboard.
- No data augmentation beyond crop/flip, no test-time augmentation, no
  ensembling — all available levers for a few more points of accuracy that
  weren't pulled because 80.6% vs. 10% already makes the point.
