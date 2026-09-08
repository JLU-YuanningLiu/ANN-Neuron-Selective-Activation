## Usage

### 1. Environment Setup

Install the required Python dependencies using `pip`:

```bash
pip install torch torchvision numpy pandas opencv-python matplotlib pyarrow
```

We recommend using a dedicated Python environment (e.g., Conda or `venv`) to avoid dependency conflicts.

---

### 2. Dataset Configuration

Before running the scripts, configure the paths to the training and test datasets.

The experiments use the Alzheimer_MRI dataset available on Hugging Face.

Dataset: Falah/Alzheimer_MRI

Open the following files:

* `AD_reconstruction_experiment.py`
* `AD_classification_experiment.py`

Locate the dataset path variables:

```python
DATA_PATH = '/path/to/your/train-00000-of-00001.parquet'
TEST_DATA_PATH = '/path/to/your/test-00000-of-00001.parquet'
```

Replace the placeholder paths with the actual locations of your Parquet files. For example:

```python
DATA_PATH = '/path/to/dataset/train-00000-of-00001.parquet'
TEST_DATA_PATH = '/path/to/dataset/test-00000-of-00001.parquet'
```

Make sure that both files are accessible from your local environment before running the scripts.

---

### 3. Pre-trained Model Weights

If you want to use the pre-trained CNN weights in `AD_classification_experiment`, update the corresponding model checkpoint paths:

```python
model_class0.load_state_dict(
    torch.load('/path/to/your/model8_0.pth')
)

model_class1.load_state_dict(
    torch.load('/path/to/your/model8_1.pth')
)
```

Replace the placeholder paths with the locations of your `.pth` files.

For example:

```python
model_class0.load_state_dict(
    torch.load('/path/to/checkpoints/model8_0.pth')
)

model_class1.load_state_dict(
    torch.load('/path/to/checkpoints/model8_1.pth')
)
```

If pre-trained weights are not available, please ensure that the corresponding script is configured to train the models from scratch before execution.

---

### 4. Running the Scripts

#### 4.1 Autoencoder Simulation

Run the following command to train the autoencoder models:

```bash
python AD_reconstruction_experiment.py
```

This script performs the autoencoder simulation and generates:

* Image reconstruction visualizations
* Layer-wise activation plots
* Other related analysis results

---

#### 4.2 CNN Classification and Analysis

Run the following command to perform CNN classification and subsequent analysis:

```bash
python AD_classification_experiment.py
```

This script performs:

* CNN classification
* Layer-wise penalty map analysis
* Weight distribution analysis
* Comparison of sparse and dense models
* Weight distribution histograms

---



