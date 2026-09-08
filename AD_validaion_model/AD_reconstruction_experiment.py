import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, random_split
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt


def split_dataset_by_label(dataset):
    """
    Splits the dataset into two subsets based on binary labels (0 and 1).
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


def plot_weight_distributions(model_sparse, model_dense, threshold, layer_names=None):
    """
    Plots layer-wise histograms comparing weight distributions between the sparse and dense models.
    Filters valid convolutional/linear layers by checking if dimension > 1.
    """
    weights_sparse = []
    weights_dense = []
    names = []

    for name, param in model_sparse.named_parameters():
        if 'weight' in name and param.dim() > 1:
            weights_sparse.append(param.data.cpu().numpy().flatten())
            names.append(name)

    for name, param in model_dense.named_parameters():
        if 'weight' in name and param.dim() > 1:
            weights_dense.append(param.data.cpu().numpy().flatten())

    num_layers = len(weights_sparse)
    if num_layers == 0:
        print("Error: No valid weights extracted. Please check parameter names.")
        return

    fig, axes = plt.subplots(nrows=num_layers, ncols=1, figsize=(10, 4 * num_layers))
    if num_layers == 1:
        axes = [axes]

    print(f"\n>>> Generating weight distribution comparison plots for {num_layers} layers...")

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

        active_s = np.sum(np.abs(w_s) > threshold) / len(w_s)
        active_d = np.sum(np.abs(w_d) > threshold) / len(w_d)
        text_str = f"Active Ratio (>{threshold}):\nAD: {active_s:.2%}\nNormal: {active_d:.2%}"
        ax.text(0.95, 0.5, text_str, transform=ax.transAxes, fontsize=10,
                verticalalignment='center', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    plt.show()


def plot_layer_activation_summary(model_sparse, model_dense, threshold=0.05):
    """
    Plots a bar chart comparing the active weight ratios for each layer between the two models.
    """
    layers = []
    ratios_sparse = []
    ratios_dense = []

    for (n_s, p_s), (n_d, p_d) in zip(model_sparse.named_parameters(), model_dense.named_parameters()):
        if 'weight' in n_s and p_s.dim() > 1:
            layer_name = n_s.replace('.weight', '')
            layers.append(layer_name)

            w_s = p_s.data.cpu().numpy().flatten()
            r_s = np.sum(np.abs(w_s) > threshold) / len(w_s)
            ratios_sparse.append(r_s)

            w_d = p_d.data.cpu().numpy().flatten()
            r_d = np.sum(np.abs(w_d) > threshold) / len(w_d)
            ratios_dense.append(r_d)

    if not layers:
        print("Error: No layer information extracted for the summary plot.")
        return

    x = np.arange(len(layers))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width / 2, ratios_dense, width, label='Normal', color='skyblue')
    rects2 = ax.bar(x + width / 2, ratios_sparse, width, label='AD', color='salmon')

    ax.set_ylabel(f'Active Weight Ratio ( > {threshold})')
    ax.set_title('Layer-wise Activation Ratio Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(layers, rotation=45, ha='right')
    ax.legend()

    ax.bar_label(rects1, fmt='%.2f', padding=3)
    ax.bar_label(rects2, fmt='%.2f', padding=3)

    plt.tight_layout()
    plt.show()


def dict_to_image(image_dict):
    """
    Converts a dictionary containing byte data into a grayscale OpenCV image format.
    """
    if isinstance(image_dict, dict) and 'bytes' in image_dict:
        byte_string = image_dict['bytes']
        nparr = np.frombuffer(byte_string, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        return img
    else:
        raise TypeError(f"Expected dictionary with 'bytes' key, got {type(image_dict)}")


class KaggleADDdataset(Dataset):
    """
    Custom PyTorch Dataset for loading parquet data containing Alzheimer's MRI scans.
    """

    def __init__(self, data_path, transform=None):
        data_df = pd.read_parquet(data_path, engine='pyarrow')
        data_df['img_arr'] = data_df['image'].apply(dict_to_image)
        data_df.drop("image", axis=1, inplace=True)
        self.data_df = data_df
        self.transform = transform

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, item):
        image_array = self.data_df.iloc[item]['img_arr']
        label = self.data_df.iloc[item]['label']

        if label == 2:
            label = 1
        else:
            label = 0

        image = torch.tensor(image_array, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label, dtype=torch.long)
        return image, label


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 0.01
THRESHOLD = 0.05
TARGET_SIZE = (128, 128)

DATA_PATH = '/media/disk/zhouzy8217/pycharmProject/medical_image_segmentation/dataset/Alzheimer MRI Disease Classification Dataset/Data/train-00000-of-00001-c08a401c53fe5312.parquet'
TEST_DATA_PATH = '/media/disk/zhouzy8217/pycharmProject/medical_image_segmentation/dataset/Alzheimer MRI Disease Classification Dataset/Data/test-00000-of-00001-44110b9df98c5585.parquet'

L1_LAMBDA_SPARSE = 0.0001
L1_LAMBDA_DENSE = 0.0


class SimpleCNN(nn.Module):
    """
    A 6-layer CNN Autoencoder architecture.
    """

    def __init__(self):
        super(SimpleCNN, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def calculate_activation_ratio(model, threshold=THRESHOLD):
    """
    Computes the proportion of active weights (absolute value > threshold) across the entire model.
    """
    total_weights = 0
    active_weights = 0
    with torch.no_grad():
        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() > 1:
                w = param.data
                total_weights += w.numel()
                active_weights += (torch.abs(w) > threshold).sum().item()
    return active_weights / total_weights if total_weights > 0 else 0


def train_model(model, loader, val_loader, l1_lambda, description, model_name):
    """
    Trains the model utilizing a specialized layer-wise L1 regularization mechanism to simulate pathology.
    """
    model = model.to(DEVICE)
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
    criterion = nn.MSELoss()

    print(f"\n--- Starting Training: {description} (Base L1: {l1_lambda}) ---")
    train_losses = []
    val_losses = []

    for epoch in range(EPOCHS):
        model.train()
        train_total_loss = 0
        train_count = 0

        for images, _ in loader:
            images = images.to(DEVICE)

            images = images / 255.0
            if images.shape[2:] != TARGET_SIZE:
                images = F.interpolate(images, size=TARGET_SIZE, mode='bilinear', align_corners=False)

            output = model(images)
            recon_loss = criterion(output, images)

            l1_loss = 0
            if l1_lambda > 0:
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        if 'encoder.0' in name:
                            factor = 0.001
                        elif 'encoder.2' in name:
                            factor = 0.01
                        elif 'encoder.4' in name:
                            factor = 0.5
                        else:
                            factor = 1.0

                        l1_loss += factor * torch.sum(torch.abs(param))

            loss = recon_loss + (l1_lambda * l1_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_total_loss += loss.item()
            train_count += 1

        train_losses.append(train_total_loss / train_count)

        model.eval()
        val_total_loss = 0
        val_count = 0

        for images, _ in val_loader:
            with torch.no_grad():
                images = images.to(DEVICE)

                images = images / 255.0
                if images.shape[2:] != TARGET_SIZE:
                    images = F.interpolate(images, size=TARGET_SIZE, mode='bilinear', align_corners=False)

                output = model(images)

                l1_loss = 0
                if l1_lambda > 0:
                    for name, param in model.named_parameters():
                        if 'weight' in name:
                            if 'encoder.0' in name:
                                factor = 0.001
                            elif 'encoder.2' in name:
                                factor = 0.01
                            elif 'encoder.4' in name:
                                factor = 0.5
                            else:
                                factor = 1.0

                            l1_loss += factor * torch.sum(torch.abs(param))

                recon_loss = criterion(output, images)
                loss = recon_loss + (l1_lambda * l1_loss)

            val_total_loss += loss.item()
            val_count += 1

        val_losses.append(val_total_loss / val_count)
        print(
            f"Epoch {epoch + 1}/{EPOCHS}, Train Avg Loss: {train_total_loss / train_count:.6f} Val Avg Loss: {val_total_loss / val_count:.6f}")

        if (epoch + 1) % 20 == 0:
            torch.save(model.state_dict(),
                       f'/media/disk/zhouzy8217/pycharmProject/medical_image_segmentation/kaggle_ad/autoencoder_save_model/{epoch + 1}_{model_name}.pth')

    return model, train_losses, val_losses


def visualize_reconstruction(model, loader, device, target_size, title="Reconstruction Comparison"):
    """
    Visualizes original versus reconstructed images for the first two samples in a batch.
    """
    model.eval()

    images, _ = next(iter(loader))
    images = images.to(device)

    images = images / 255.0
    if images.shape[2:] != target_size:
        images = F.interpolate(images, size=target_size, mode='bilinear', align_corners=False)

    with torch.no_grad():
        outputs = model(images)

    imgs_np = images.cpu().numpy()
    outs_np = outputs.cpu().numpy()

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))
    plt.suptitle(title, fontsize=16)

    axes[0, 0].imshow(imgs_np[0, 0], cmap='gray')
    axes[0, 0].set_title("Sample 1: Input")
    axes[0, 0].axis('off')

    axes[0, 1].imshow(outs_np[0, 0], cmap='gray')
    axes[0, 1].set_title("Sample 1: Reconstructed")
    axes[0, 1].axis('off')

    axes[1, 0].imshow(imgs_np[1, 0], cmap='gray')
    axes[1, 0].set_title("Sample 2: Input")
    axes[1, 0].axis('off')

    axes[1, 1].imshow(outs_np[1, 0], cmap='gray')
    axes[1, 1].set_title("Sample 2: Reconstructed")
    axes[1, 1].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    plt.show()


def plot_loss_curves(train_losses, val_losses, title="Training & Validation Loss"):
    """
    Plots the training and validation loss curves over epochs.
    """
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 6))
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.grid(True, linestyle='--', alpha=0.6)

    plt.plot(epochs, train_losses, 'b-o', label='Training Loss', linewidth=2, markersize=4)
    plt.plot(epochs, val_losses, 'r-s', label='Validation Loss', linewidth=2, markersize=4)

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss (MSE)', fontsize=12)
    plt.legend(fontsize=12, loc='upper right')

    plt.tight_layout()
    plt.show()


def plot_global_activation_comparison(ratio_0, ratio_1, threshold_info="> 0.05"):
    """
    Generates a bar chart comparing the global weight activation ratios between the two models.
    """
    labels = ['Class 0\n(AD)', 'Class 1\n(Normal)']
    values = [ratio_0, ratio_1]
    colors = ['#FF6B6B', '#4ECDC4']

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, values, color=colors, width=0.5, alpha=0.9)

    plt.title(f'Global Weight Activation Ratio Comparison\n(Threshold {threshold_info})', fontsize=14,
              fontweight='bold')
    plt.ylabel('Activation Ratio (Active / Total)', fontsize=12)
    plt.ylim(0, max(values) * 1.2)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                 f'{height:.2%}',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print(f"Loading data from: {DATA_PATH}")
    train_dataset = KaggleADDdataset(DATA_PATH)
    test_dataset = KaggleADDdataset(DATA_PATH)

    binary_labels = train_dataset.data_df['label'].apply(lambda x: 1 if x == 2 else 0)
    print(binary_labels.value_counts())

    val_len = int(len(train_dataset) * 0.2)
    train_len = len(train_dataset) - val_len
    full_dataset, valid_dataset = random_split(train_dataset, [train_len, val_len])

    dataset_0, dataset_1 = split_dataset_by_label(full_dataset)
    val_dataset_0, val_dataset_1 = split_dataset_by_label(valid_dataset)

    loader_0 = DataLoader(dataset_0, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    loader_1 = DataLoader(dataset_1, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    val_loader_0 = DataLoader(val_dataset_0, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)
    val_loader_1 = DataLoader(val_dataset_1, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)

    model_0 = SimpleCNN()
    model_0, train_losses_0, val_losses_0 = train_model(model_0, loader_0, val_loader_0, l1_lambda=L1_LAMBDA_SPARSE,
                                                        description="Class 0 (Sparse Strategy)", model_name='model8_0')

    model_1 = SimpleCNN()
    model_1, train_losses_1, val_losses_1 = train_model(model_1, loader_1, val_loader_1, l1_lambda=L1_LAMBDA_DENSE,
                                                        description="Class 1 (Normal Strategy)", model_name='model8_1')

    print("\n" + "=" * 40)
    print(f"Final Results Comparison (Threshold > {THRESHOLD})")
    print("=" * 40)

    ratio_0 = calculate_activation_ratio(model_0)
    ratio_1 = calculate_activation_ratio(model_1)

    print(f"Model A (Class 0): Activation Ratio = {ratio_0:.2%}")
    print(f"Model B (Class 1): Activation Ratio = {ratio_1:.2%}")

    if ratio_0 < ratio_1:
        print(">> Success: Class 0 model demonstrates significant sparsity.")
    else:
        print(">> Note: Differences are minimal. Consider increasing the L1_LAMBDA_SPARSE coefficient.")

    print("\n>>> Initiating visual analysis...")

    plot_global_activation_comparison(ratio_0, ratio_1)

    plot_loss_curves(train_losses_0, val_losses_0, title='Disease Training & Validation Loss')
    plot_loss_curves(train_losses_1, val_losses_1, title='Normal Training & Validation Loss')

    plot_layer_activation_summary(model_0, model_1, threshold=0.0)

    print("\n>>> Initiating visual analysis...")

    plot_weight_distributions(model_0, model_1)

    plot_layer_activation_summary(model_0, model_1, threshold=0.05)

    print("\n>>> Generating reconstruction comparisons...")

    visualize_reconstruction(
        model_0,
        loader_0,
        DEVICE,
        TARGET_SIZE,
        title="Model A (AD) Reconstruction"
    )

    visualize_reconstruction(
        model_1,
        loader_1,
        DEVICE,
        TARGET_SIZE,
        title="Model B (Normal) Reconstruction"
    )