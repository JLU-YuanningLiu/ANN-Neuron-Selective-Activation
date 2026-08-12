````
# ADS-CNN: Brain-Inspired Neurodegenerative Damage Modeling in Artificial Neural Networks

This repository provides the implementation of **ADS-CNN**, a
computational framework for studying how biologically inspired
structural damage affects artificial-neuron activation and network
function.

ADS-CNN introduces **Neurodegenerative Drop (NDD)**, a
pathology-constrained irreversible damage mechanism inspired by
Alzheimer's disease (AD) progression. By combining tau-inspired path
degeneration and A尾-inspired block degeneration, NDD models selective
artificial-neuron loss and investigates how damage organization
influences the relationship between activation and behavior.

------------------------------------------------------------------------

## Overview

Deep neural networks provide simplified models for studying information
processing, but conventional pruning or dropout methods usually ignore
the structured nature of biological degeneration.

ADS-CNN evaluates three types of perturbations:

-   **Reversible perturbation**
    -   DropPath
    -   DropBlock
    -   Feature-map dropout
-   **Irreversible lesion**
    -   Random permanent artificial-neuron deletion
-   **Neurodegenerative Drop (NDD)**
    -   Path-level degeneration inspired by tau pathology
    -   Block-level degeneration inspired by A尾 pathology

The framework tests whether pathology-inspired damage organization
produces selective activation changes beyond simple structural deletion.

------------------------------------------------------------------------

## Experimental Setting

-   **Dataset:** CIFAR-100
-   **Model:** ResNet-18
-   **Task:** 100-class image classification
-   **Activation measurement:** forward activation threshold = 0.02

Artificial-neuron activation is quantified from forward responses, and
functional changes are evaluated through classification performance and
activation dynamics.

------------------------------------------------------------------------

## Neurodegenerative Drop (NDD)

NDD applies irreversible damage through survival-mask-controlled channel
suppression.

Damaged channels:

-   are permanently deactivated;
-   have weights fixed to zero;
-   receive no gradient updates.

Unlike random pruning, NDD introduces structured degeneration patterns:

-   **Path-level damage:** selective single-channel degeneration.
-   **Block-level damage:** grouped channel degeneration.

These mechanisms provide a computational analogy to selective
vulnerability during AD progression.

------------------------------------------------------------------------

## Main Results

At the same final structural damage level:

  Condition                      Top-1 Accuracy   Activation Rate
  ---------------------------- ---------------- -----------------
  Random irreversible lesion        21.6 ± 1.3%       11.5 ± 0.8%
  NDD                                6.8 ± 0.7%        4.9 ± 0.5%

NDD produced a stronger decline in both task performance and
artificial-neuron activation compared with damage-matched random
lesions, suggesting that **damage organization influences
activation-performance coupling**.

------------------------------------------------------------------------

## AD Stage Simulation

ADS-CNN further models Braak-stage-inspired AD progression using
region-to-layer functional mapping:

  Biological Region            Network Layer
  ---------------------------- ---------------
  ECII-inspired degeneration   Middle layers
  CA1-inspired degeneration    Late layers

From healthy state to late-stage AD proxy:

  Stage            Top-1 Accuracy   Activation Rate
  -------------- ---------------- -----------------
  Baseline                  75.8%             80.2%
  Braak I-II                64.1%             73.9%
  Braak III-IV              55.0%             49.4%
  Braak V-VI                38.1%             26.4%

------------------------------------------------------------------------

## Installation

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## Usage

Example:

``` bash
python main.py \
    --experiment ndd \
    --model resnet18 \
    --dataset cifar100
```

------------------------------------------------------------------------

## Limitations

ADS-CNN is a computational analogy rather than a biological
reconstruction.

The brain-region-to-network-layer mapping and Braak-stage-inspired
damage schedules are manually defined based on neuropathological
considerations. The current validation is performed on the
ResNet-18/CIFAR-100 setting, and broader validation across architectures
and datasets remains necessary.

````