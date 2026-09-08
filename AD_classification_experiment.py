import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from dataset import KaggleADDdataset
from torch.utils.data import random_split, DataLoader, Subset
from network import EightLayerCNN, TwelveLayerCNN

N_CLASSES = 2  # Binary classification (AD vs. Normal)


def split_dataset_by_label(dataset):
    """
    Splits the given PyTorch Dataset into two subsets based on binary labels (0 and 1).

    Args:
        dataset: PyTorch Dataset returning (x, y) pairs.
    Returns:
        class0_dataset, class1_dataset
    """
    indices_class0 = []
    indices_class1 = []

    for i in range(len(dataset)):
        _, label = dataset[i]
        if label == 0:
            indices_class0.append(i)
        elif label == 1:
            indices_class1.append(i)

    class0_dataset = Subset(dataset, indices_class0)
    class1_dataset = Subset(dataset, indices_class1)

    return class0_dataset, class1_dataset


class BaselineCNN(nn.Module):
    def __init__(self):
        super(BaselineCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.batchnorm1 = nn.BatchNorm2d(num_features=32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()

        # Note: Input must be 128x128 for the flattened dimension to match 64 * 32 * 32
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.out = nn.Linear(128, N_CLASSES)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.batchnorm1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x


def train_classifier_with_val(model, train_loader, val_loader, device, l1_lambda, target_label_value, description):
    """
    Standard training loop with validation and global L1 regularization.
    """
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train_loss_history = []
    val_loss_history = []
    train_class_loss_history = []

    print(f"\n--- Starting Training: {description} (L1: {l1_lambda}) ---")
    epochs = 20

    for epoch in range(epochs):
        # 1. Training Phase
        model.train()
        total_train_loss = 0
        train_batches = 0
        train_class_loss = 0

        for images, _ in train_loader:
            images = images.to(device)
            # Force target label since we are training on a single class distribution
            labels = torch.full((images.size(0),), target_label_value, dtype=torch.long).to(device)

            if images.max() > 1.0:
                images = images / 255.0

            outputs = model(images)
            class_loss = criterion(outputs, labels)

            # Global L1 Regularization
            l1_loss = 0
            if l1_lambda > 0:
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        l1_loss += torch.sum(torch.abs(param))

            loss = class_loss + (l1_lambda * l1_loss)
            train_class_loss += class_loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_batches += 1

        avg_class_loss = total_train_loss / train_batches
        avg_train_loss = total_train_loss / train_batches
        train_loss_history.append(avg_train_loss)
        train_class_loss_history.append(avg_class_loss)

        # 2. Validation Phase
        model.eval()
        total_val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                labels = torch.full((images.size(0),), target_label_value, dtype=torch.long).to(device)

                if images.max() > 1.0:
                    images = images / 255.0

                outputs = model(images)

                # Validation loss focuses on actual classification capability without regularization terms
                val_loss = criterion(outputs, labels)
                total_val_loss += val_loss.item()
                val_batches += 1

        avg_val_loss = total_val_loss / val_batches
        val_loss_history.append(avg_val_loss)

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    return model, train_loss_history, train_class_loss_history, val_loss_history


def train_classifier_with_val_L1(model, train_loader, val_loader, device, base_lambda, penalty_map, target_label_value,
                                 description):
    """
    Training loop implementing layer-wise custom L1 regularization to simulate pathological sparsity.
    """
    model = model.to(device)
    if target_label_value == 1:
        optimizer = optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.00002)

    criterion = nn.CrossEntropyLoss()

    train_loss_history = []
    val_loss_history = []
    train_class_loss_history = []

    print(f"\n--- Starting Training: {description} ---")
    print(f"Strategy Configuration: Base Lambda={base_lambda}")

    # Store convolutional layers for layer-specific L1 penalties
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append((name, module))

    epochs = 100

    for epoch in range(epochs):
        # 1. Training Phase
        model.train()
        total_train_loss = 0
        train_batches = 0
        train_class_loss = 0

        for images, _ in train_loader:
            images = images.to(device)
            labels = torch.full((images.size(0),), target_label_value, dtype=torch.long).to(device)

            if images.max() > 1.0:
                images = images / 255.0

            outputs = model(images)
            task_loss = criterion(outputs, labels)

            # Core concept: Compute layer-wise L1 Loss based on predefined penalty map
            l1_loss = 0
            if base_lambda > 0:
                for i, (name, layer) in enumerate(conv_layers):
                    factor = penalty_map.get(i, 1.0)
                    if factor > 0:
                        l1_loss += factor * torch.sum(torch.abs(layer.weight))

            loss = task_loss + (base_lambda * l1_loss)

            train_class_loss += task_loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            train_batches += 1

        avg_class_loss = total_train_loss / train_batches
        avg_train_loss = total_train_loss / train_batches
        train_loss_history.append(avg_train_loss)
        train_class_loss_history.append(avg_class_loss)

        # 2. Validation Phase
        model.eval()
        total_val_loss = 0
        val_batches = 0

        with torch.no_grad():
            for images, _ in val_loader:
                images = images.to(device)
                labels = torch.full((images.size(0),), target_label_value, dtype=torch.long).to(device)

                if images.max() > 1.0:
                    images = images / 255.0

                outputs = model(images)
                val_loss = criterion(outputs, labels)

                # Reapply validation L1 measurement for consistent tracking
                l1_loss = 0
                if base_lambda > 0:
                    for i, (name, layer) in enumerate(conv_layers):
                        factor = penalty_map.get(i, 1.0)
                        if factor > 0:
                            l1_loss += factor * torch.sum(torch.abs(layer.weight))

                val_loss = val_loss + (base_lambda * l1_loss)
                total_val_loss += val_loss.item()
                val_batches += 1

        avg_val_loss = total_val_loss / val_batches
        val_loss_history.append(avg_val_loss)

        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    return model, train_loss_history, train_class_loss_history, val_loss_history


def calculate_activation_ratio(model, threshold=0.01):
    """
    Calculates the global percentage of active weights exceeding a given threshold.
    """
    total = 0
    active = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            # Only evaluate weights for convolutional and linear layers, ignoring the final classifier
            if 'weight' in name and param.dim() > 1 and 'classifier' not in name:
                total += param.numel()
                active += (torch.abs(param) > threshold).sum().item()
    return active / total if total > 0 else 0


def plot_comparison(model_0, model_1, threshold=0.01):
    """
    Plots a global activation ratio comparison between two models.
    """
    ratio_0 = calculate_activation_ratio(model_0, threshold)
    ratio_1 = calculate_activation_ratio(model_1, threshold)

    labels = ['Class 0 (AD)', 'Class 1 (Normal)']
    values = [ratio_0, ratio_1]
    colors = ['red', 'blue']

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, values, color=colors, alpha=0.7)

    plt.title(f'Weight Activation Ratio (Threshold > {threshold})')
    plt.ylabel('Activation Ratio')
    plt.ylim(0, max(values) * 1.2)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{height:.2%}', ha='center', va='bottom', fontsize=12)

    plt.show()


def visualize_layer_wise_activation(model_sparse, model_dense, threshold=0.05):
    """
    Visualizes layer-wise active weight ratios, comparing sparse (AD) vs dense (Normal) models.
    """
    layer_names = []
    ratios_sparse = []
    ratios_dense = []

    print(f"\n>>> Calculating layer-wise activation ratios (Threshold > {threshold})...")

    for (name_s, param_s), (name_d, param_d) in zip(model_sparse.named_parameters(), model_dense.named_parameters()):
        # Filter for convolutional and fully connected layers using dim > 1
        if 'weight' in name_s and param_s.dim() > 1:
            short_name = name_s.replace('.weight', '').replace('module.', '')
            layer_names.append(short_name)

            w_s = param_s.detach().cpu().numpy()
            w_d = param_d.detach().cpu().numpy()

            total = w_s.size
            active_s = np.sum(np.abs(w_s) > threshold)
            active_d = np.sum(np.abs(w_d) > threshold)

            r_s = active_s / total
            r_d = active_d / total

            ratios_sparse.append(r_s)
            ratios_dense.append(r_d)

            print(f"Layer: {short_name:10} | Sparse: {r_s:.2%} | Dense: {r_d:.2%}")

    x = np.arange(len(layer_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    rects1 = ax.bar(x - width / 2, ratios_dense, width, label='Class 1 (Normal)', color='#4ECDC4', alpha=0.9)
    rects2 = ax.bar(x + width / 2, ratios_sparse, width, label='Class 0 (AD)', color='#FF6B6B', alpha=0.9)

    ax.set_ylabel(f'Activation Ratio (|w| > {threshold})', fontsize=12)
    ax.set_title(f'Layer-wise Weight Activation Comparison\n(Threshold = {threshold})', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(layer_names, rotation=45, ha='right', fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.grid(axis='y', linestyle='--', alpha=0.3)

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1%}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)

    plt.tight_layout()
    plt.show()


def plot_loss_dynamics(train_losses, val_losses, title="Loss Dynamics"):
    """
    Plots training and validation loss curves across epochs.
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, 'b-o', label='Training Loss ', linewidth=2)
    plt.plot(epochs, val_losses, 'r-s', label='Validation Loss ', linewidth=2)

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.show()


def plot_weight_distributions(model_sparse, model_dense, THRESHOLD, layer_names=None):
    """
    Plots side-by-side weight distribution histograms for each corresponding layer.
    """
    weights_sparse = []
    weights_dense = []
    names = []

    # Extract weights for the sparse model
    for name, param in model_sparse.named_parameters():
        if 'weight' in name and param.dim() > 1:
            weights_sparse.append(param.data.cpu().numpy().flatten())
            names.append(name)

    # Extract weights for the dense model
    for name, param in model_dense.named_parameters():
        if 'weight' in name and param.dim() > 1:
            weights_dense.append(param.data.cpu().numpy().flatten())

    num_layers = len(weights_sparse)
    if num_layers == 0:
        print("Error: No weights extracted. Please verify parameter names.")
        return

    fig, axes = plt.subplots(nrows=num_layers, ncols=1, figsize=(10, 4 * num_layers))
    if num_layers == 1:
        axes = [axes]

    print(f"\n>>> Generating weight distribution plots for {num_layers} layers...")

    for i in range(num_layers):
        ax = axes[i]
        w_s = weights_sparse[i]
        w_d = weights_dense[i]

        range_lim = [-0.5, 0.5]

        ax.hist(w_d, bins=100, range=range_lim, alpha=0.5, color='blue', label='Normal (Class 1)', density=True)
        ax.hist(w_s, bins=100, range=range_lim, alpha=0.6, color='red', label='AD (Class 0)', density=True)

        ax.set_title(f"Layer: {names[i]} Weight Distribution")
        ax.set_xlabel("Weight Value")
        ax.set_ylabel("Density")
        ax.legend()
        ax.grid(True, alpha=0.3)

        active_s = np.sum(np.abs(w_s) > THRESHOLD) / len(w_s)
        active_d = np.sum(np.abs(w_d) > THRESHOLD) / len(w_d)
        text_str = f"Active Ratio (>{THRESHOLD}):\nAD: {active_s:.2%}\nNormal: {active_d:.2%}"
        ax.text(0.95, 0.5, text_str, transform=ax.transAxes, fontsize=10,
                verticalalignment='center', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_path = '/media/disk/zhouzy8217/pycharmProject/medical_image_segmentation/dataset/Alzheimer MRI Disease Classification Dataset/Data/train-00000-of-00001-c08a401c53fe5312.parquet'
    test_path = "/media/disk/zhouzy8217/pycharmProject/medical_image_segmentation/dataset/Alzheimer MRI Disease Classification Dataset/Data/test-00000-of-00001-44110b9df98c5585.parquet"

    train_dataset = KaggleADDdataset(data_path=train_path)
    test_dataset = KaggleADDdataset(data_path=test_path)

    val_len = int(len(train_dataset) * 0.2)
    train_len = len(train_dataset) - val_len
    train_dataset, valid_dataset = random_split(train_dataset, [train_len, val_len])

    train_dataset_class0, train_dataset_class1 = split_dataset_by_label(dataset=train_dataset)
    valid_dataset_class0, valid_dataset_class1 = split_dataset_by_label(dataset=valid_dataset)
    test_dataset_class0, test_dataset_class1 = split_dataset_by_label(dataset=test_dataset)

    train_loader_class0 = DataLoader(train_dataset_class0, batch_size=128, num_workers=4, shuffle=True)
    val_loader_class0 = DataLoader(valid_dataset_class0, batch_size=128, num_workers=4, shuffle=False)
    test_loader_class0 = DataLoader(valid_dataset_class0, batch_size=128, num_workers=4, shuffle=False)

    train_loader_class1 = DataLoader(train_dataset_class1, batch_size=128, num_workers=4, shuffle=True)
    val_loader_class1 = DataLoader(valid_dataset_class1, batch_size=128, num_workers=4, shuffle=False)
    test_loader_class1 = DataLoader(valid_dataset_class1, batch_size=128, num_workers=4, shuffle=False)

    model_class0 = EightLayerCNN()
    model_class1 = EightLayerCNN()

    model_name_0 = 'model12_0'
    model_name_1 = 'model8_1v0'

    # Strategy configuration for Model 0 (AD Pathological simulation)
    # Applying L1 regularization heavily on association layers while protecting early/deep layers
    L1_LAMBDA = 0.00001
    ad_penalty_map = {
        0: 0.001,  # Layer 1 (Shallow visual features): Protected
        1: 0.01,  # Layer 2: Light penalty
        2: 0.5,  # Layer 3: Transition penalty
        3: 1.0,  # Layer 4 (Intermediate association area): Heavy penalty!
        4: 1.0,  # Layer 5 (Intermediate association area): Heavy penalty!
        5: 0.5,  # Layer 6: Moderate penalty
        6: 0.1,  # Layer 7: Light penalty
        7: 0.01  # Layer 8 (Deep semantic area): Protected
    }

    # Reference structure for 12-Layer architecture
    # ad_penalty_map_12layer = {
    #     0: 0.0001, 1: 0.001,       # Shallow layers: Intact primary visual cortex
    #     2: 0.01,   3: 0.1,         # Transition layers: Initial structural damage
    #     4: 0.8,    5: 1.0,         # Intermediate layers: White matter tract disconnection
    #     6: 1.0,    7: 0.8,         # Intermediate layers: White matter tract disconnection
    #     8: 0.5,    9: 0.3,         # Deep layers: Hippocampal atrophy simulation
    #     10: 0.1,   11: 0.05        # Bottleneck: Preserve core concepts to retain >50% accuracy
    # }

    normal_penalty_map = {}

    # Loading pre-trained state dicts for visualization
    model_class0.load_state_dict(torch.load(
        '/media/disk/zhouzy8217/pycharmProject/medical_image_segmentation/kaggle_ad/save_single_model/model8_0.pth'))
    model_class1.load_state_dict(torch.load(
        '/media/disk/zhouzy8217/pycharmProject/medical_image_segmentation/kaggle_ad/save_single_model/model8_1.pth'))

    print("\n>>> Final Results Comparison <<<")
    THRESHOLD = 0.01

    plot_weight_distributions(model_class0, model_class1, THRESHOLD)