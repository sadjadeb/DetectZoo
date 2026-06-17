# DetectZoo

![DetectZoo](https://anonymous.4open.science/api/repo/DetectZoo-1BEC/file/DetectZoo_banner.png?v=6072c3e2)

DetectZoo is a research-oriented Python toolkit that provides **implementations of AI-generated content detectors across multiple modalities**, including **text, images, and audio**.

The goal of DetectZoo is to make detection methods **easy to use, reproducible, and extensible**, enabling researchers and practitioners to benchmark and deploy AI-generated content detectors with minimal effort.

DetectZoo aggregates detection approaches into a **single, unified API**, allowing users to load and apply detectors with just a few lines of code.

---

## Installation

For the sake of anonymity, we put the package on TestPyPI and you can install it with the following command:

*Note: This is a temporary solution and we will release the package on PyPI after the paper is accepted.*

```bash
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ detectzoo-anon
```

or install from source:

```bash
git clone https://anonymous.4open.science/r/DetectZoo-1BEC/
cd detectzoo
pip install -e .
```

Optional extra for contributors (`pytest`, `pytest-cov`, `ruff`):

```bash
pip install -e ".[dev]"
```

The base install already includes dependencies for text, image, and audio detectors.

---

## Quick Start

### Detect AI-generated text

```python
from detectzoo import load_detector

detector = load_detector("fast_detectgpt")

text = "Large language models are transforming many fields."
result = detector.predict(text)

print(result)
# DetectionResult(score=1.2345, label='ai', confidence=0.8012)
print(result.score, result.label)
```

### Detect AI-generated images

```python
from detectzoo import load_detector

detector = load_detector("aeroblade")

result = detector.predict("image.png")
print(result.label)  # "ai" or "human"
```

### Detect synthetic audio

```python
from detectzoo import load_detector

detector = load_detector("rawnet2")

result = detector.predict("speech.wav")
print(result.score, result.label)
```

### List all available detectors

```python
from detectzoo import list_detectors

print(list_detectors())            # all detectors
print(list_detectors("text"))      # text-only
print(list_detectors("image"))     # image-only
print(list_detectors("audio"))     # audio-only
```

---

## Supported Detectors

DetectZoo ships detectors for **text**, **images**, and **audio**. Each uses the same interface: `detector.predict(input) → DetectionResult`.

See [METHODS_AND_MODELS.md](METHODS_AND_MODELS.md) for detailed tables of supported detectors, including registry names, implementation classes, and method summaries. To programmatically list available detector names in code, use `list_detectors()` or specify a type: `list_detectors("text" | "image" | "audio")`.

---

## Core Components

### DetectionResult

Every `predict()` call returns a `DetectionResult` dataclass:

```python
@dataclass
class DetectionResult:
    score: float       # Higher = more likely AI-generated
    label: str         # "ai" or "human"
    confidence: float  # Confidence in the label (0–1)
    metadata: dict     # Detector-specific extra info
```

The `metadata` dictionary varies by detector and may include values like `avg_log_likelihood`, `mean_curvature`, `ppl_observer`, `hf_lf_ratio`, etc.

---

## Benchmarking

DetectZoo includes a built-in evaluation pipeline for comparing detectors on labelled datasets.

### Built-in datasets

DetectZoo ships with loaders for popular detection benchmarks. Data is downloaded and cached automatically on first use — no manual setup needed.

See [METHODS_AND_MODELS.md — Built-in datasets](METHODS_AND_MODELS.md#built-in-datasets) for a complete table of built-in datasets, with class names, descriptions, sources, and `load_dataset` registry keys.


```python
from detectzoo.datasets import CHEATDataset

# Auto-downloads from GitHub on first call, cached in .detectzoo_data/cheat/
dataset = CHEATDataset()
dataset = CHEATDataset(categories=["generation"])   # only first-pass ChatGPT abstracts

# Or point to a local copy
dataset = CHEATDataset(path="data/cheat/")

for item in dataset:
    print(item.label, item.data[:80])
```


All datasets cache downloaded files under a `.detectzoo_data/` directory (configurable via `cache_dir`) so subsequent loads are instant.

### Using the evaluator

```python
from detectzoo import load_detector
from detectzoo.datasets import BaseDataset, HC3Dataset
from detectzoo.benchmarks import BenchmarkEvaluator

# Built-in benchmark dataset
dataset = HC3Dataset(subsets=["finance"])

# Or load a dataset from two directories
dataset = BaseDataset.from_directory("data/real/", "data/fake/")

# Or from a CSV (text modality)
dataset = BaseDataset.from_csv("data/texts.csv", text_column="text", label_column="label")

# Evaluate detectors
evaluator = BenchmarkEvaluator(dataset)
evaluator.run_and_print([
    load_detector("log_likelihood"),
    load_detector("entropy"),
    load_detector("fast_detectgpt"),
])
```

This prints a comparison table with accuracy, precision, recall, F1, and AUROC.

### Metrics

The `compute_metrics` utility computes standard binary-classification metrics:

```python
from detectzoo.utils import compute_metrics

metrics = compute_metrics(
    labels=[0, 0, 1, 1],
    scores=[0.1, 0.3, 0.8, 0.9],
    threshold=0.5,
)
# {'accuracy': 1.0, 'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'tpr': 1.0, 'fpr': 0.0, 'roc_auc': 1.0, 'pr_auc': 1.0, 'avg_precision': 1.0}
```


## Design Philosophy

DetectZoo is built around three principles.

### 1. Reproducibility

Many detection methods are difficult to reproduce due to missing implementation details. DetectZoo provides **clean and standardized implementations of published detectors** with references to the original papers.

### 2. Accessibility

Users should not need to reimplement detectors. DetectZoo provides **simple imports and unified interfaces**. Loading any detector is a single function call.

### 3. Extensibility

Adding a new detector takes a single file. Subclass `BaseDetector`, implement `predict`, and register with a decorator:

```python
from detectzoo.detectors import BaseDetector
from detectzoo.core.registry import register_detector

@register_detector("my_detector")
class MyDetector(BaseDetector):
    modality = "text"  # or "image" or "audio"

    def __init__(self, threshold=0.5, device="cpu", **kwargs):
        super().__init__(threshold=threshold, device=device, **kwargs)

    def predict(self, input_data):
        # Your detection logic here
        score = 0.0
        return self._make_result(score)
```

The detector is then immediately available via `load_detector("my_detector")`. See `examples/custom_detector.py` for a complete runnable example.

---

## Examples

The `examples/` directory contains runnable scripts grouped by modality. Most replication scripts download public benchmark data, run detectors with `BenchmarkEvaluator`, and write metrics under `experiments/`.

### Getting started

| Script | Description |
|--------|-------------|
| [custom_detector.py](examples/custom_detector.py) | Create, register, and use a toy custom text detector (`word_length`). |

### Replication scripts

| Script | Description |
|--------|-------------|
| [text/ood_replicate.py](examples/text/ood_replicate.py) | Replicate OOD paper baselines on the labeled RAID test split (default 1000 samples). |
| [text/gecscore_replicate.py](examples/text/gecscore_replicate.py) | Replicate GECScore baselines on released `normal_data` JSON files (per source × generator model). |
| [text/imbd_replicate.py](examples/text/imbd_replicate.py) | Replicate ImBD baselines on released rewrite/paraphrase JSON (human `original` vs AI `rewritten`). |
| [text/text_fluoroscopy_replicate.py](examples/text/text_fluoroscopy_replicate.py) | Replicate Text-Fluoroscopy baselines on processed JSON files from the authors' repo. |
| [image/image_replicate.py](examples/image/image_replicate.py) | Run image detectors on built-in datasets (`self_synthesis`, `aigcdetect`, `cnn_detection`, `genimage`, `univfd_diffusion`) and save benchmark JSON. |
| [audio/audio_replicate.py](examples/audio/audio_replicate.py) | Run audio detectors on built-in benchmarks (`asvspoof2019`, `for`, `in_the_wild`, `deepfake_eval_2024`) with balanced sampling. |

Run from the project root:

```bash
# Quick start — no downloads
python examples/custom_detector.py

# Text replication (OOD on RAID)
python examples/text/ood_replicate.py --device cuda --max-samples 100

# Image replication
python examples/image/image_replicate.py \
    --dataset self_synthesis \
    --partitions AttGAN BEGAN \
    --detectors cnnspot patchcraft univfd

# Audio replication
python examples/audio/audio_replicate.py --dataset in_the_wild --detectors rawnet2 aasist
```

---

## Contributing

We welcome community contributions. You can contribute by:

* Adding new detectors (see the extensibility section above)
* Improving existing implementations
* Adding benchmark datasets
* Improving documentation
* Reporting issues and suggesting features
