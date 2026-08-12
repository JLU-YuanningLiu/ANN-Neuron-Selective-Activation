# BSNA-Net Python Reference Implementation

This repository implements the Python dynamic neuron-gating pathway described in the manuscript.

The implementation follows the three-stage mechanism:

1. Dense representation learning
2. Data-subset partition and neuron-subset initialization
3. Input-dependent sparse training with dynamic neuron and connection gating

The core implementation is in `model/bsna.py`. It contains data-subset routing, Neuron Mapping Gate, learnable neuron thresholds, connection-level threshold masks, the surrogate gradient, and the task, sparsity, mapping-consistency, functional-diversification, and activation-budget objectives.

## Model files

Computer vision:

- `model/resnet50.py`
- `model/densenet121.py`
- `model/efficientnet_b0.py`
- `model/convnext_t.py`
- `model/vit_b16.py`
- `model/swin_t.py`

Natural language processing:

- `model/fasttext.py`
- `model/textcnn.py`
- `model/bilstm.py`
- `model/bert_base.py`
- `model/roberta_base.py`
- `model/deberta_base.py`

## Data-subset settings

- ImageNet-1K: K=100
- BDD100K: K=20
- Yahoo Answers Topics: K=20
- DBpedia14: K=14

The mapping initialization stage samples 15% of the training set, clusters semantic representations with K-Means, accumulates neuron activation frequency and gradient-based task contribution, computes neuron importance, constructs candidate neuron subsets, and initializes the dynamic gates from the resulting subset-neuron correspondence.

## Training stages

Dense stage:

```bash
python train.py --config configs/imagenet.yaml --model resnet50 --data /path/to/imagenet --stage dense --epochs 100 --output checkpoints/resnet50_dense.pt
```

Mapping initialization:

```bash
python initialize_mapping.py --config configs/imagenet.yaml --model resnet50 --data /path/to/imagenet --checkpoint checkpoints/resnet50_dense.pt --output checkpoints/resnet50_mapping.pt
```

Sparse stage:

```bash
python train.py --config configs/imagenet.yaml --model resnet50 --data /path/to/imagenet --stage sparse --epochs 200 --checkpoint checkpoints/resnet50_mapping.pt --output checkpoints/resnet50_bsna.pt
```

Evaluation:

```bash
python evaluate.py --config configs/imagenet.yaml --model resnet50 --data /path/to/imagenet --checkpoint checkpoints/resnet50_bsna.pt
```

The manuscript specifies the total optimization settings for the computational-cost experiments but does not provide a unique epoch allocation between the dense and sparse stages. For this reason, `train.py` requires the number of epochs for each stage to be supplied explicitly rather than silently imposing an undocumented split.

## Dataset layout

Vision datasets use an ImageFolder-compatible layout with `train` and optionally `val` directories.

The NLP loader expects a CSV file with `text` and `label` columns. Lightweight NLP models use a 50,000-word vocabulary and a maximum sequence length of 256. Transformer models use their standard tokenizers with a maximum sequence length of 256.
