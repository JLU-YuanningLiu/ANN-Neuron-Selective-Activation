import torch.nn as nn
import torchvision.models as tv_models

# 尝试导入新版 torchvision 的权重枚举（0.13+）
try:
    from torchvision.models import ResNet18_Weights, ResNet34_Weights, VGG16_Weights
except ImportError:
    # 旧版 torchvision 没有这些枚举时，这些变量为 None
    ResNet18_Weights = None
    ResNet34_Weights = None
    VGG16_Weights = None


def build_resnet18(num_classes: int = 100, pretrained: bool = False, cifar_stem: bool = False) -> nn.Module:
    """
    Build a ResNet-18 model for classification.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to use ImageNet-pretrained weights.

    Returns:
        nn.Module: ResNet-18 model.
    """
    # 新版 torchvision 推荐使用 weights 参数
    if ResNet18_Weights is not None:
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv_models.resnet18(weights=weights)
    else:
        # 兼容旧版：没有 ResNet18_Weights 时仍然使用 pretrained 参数
        model = tv_models.resnet18(pretrained=pretrained)

    if cifar_stem:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model



def build_resnet34(num_classes: int = 100, pretrained: bool = False, cifar_stem: bool = False) -> nn.Module:
    """
    Build a ResNet-34 model for classification.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to use ImageNet-pretrained weights.

    Returns:
        nn.Module: ResNet-34 model.
    """
    # 新版 torchvision 推荐使用 weights 参数
    if ResNet34_Weights is not None:
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv_models.resnet34(weights=weights)
    else:
        # 兼容旧版：没有 ResNet34_Weights 时仍然使用 pretrained 参数
        model = tv_models.resnet34(pretrained=pretrained)

    if cifar_stem:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_vgg16(num_classes: int = 100, pretrained: bool = False) -> nn.Module:
    """
    Build a VGG-16 model for classification.

    Args:
        num_classes: Number of output classes.
        pretrained: Whether to use ImageNet-pretrained weights.

    Returns:
        nn.Module: VGG-16 model.
    """
    if VGG16_Weights is not None:
        weights = VGG16_Weights.IMAGENET1K_V1 if pretrained else None
        model = tv_models.vgg16(weights=weights)
    else:
        # 兼容旧版
        model = tv_models.vgg16(pretrained=pretrained)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


MODEL_FACTORY = {
    "resnet18": build_resnet18,
    "resnet34": build_resnet34,
    "vgg16": build_vgg16,
}


def build_model(model_name: str, num_classes: int = 100, pretrained: bool = False, **kwargs) -> nn.Module:
    """
    Build a model by name.

    Args:
        model_name: Name of the model, e.g. 'resnet34', 'vgg16'.
        num_classes: Number of output classes.
        pretrained: Whether to use pretrained weights.

    Returns:
        nn.Module: The constructed model.
    """
    model_name = model_name.lower()
    if model_name not in MODEL_FACTORY:
        raise ValueError(f"Unknown model name: {model_name}. Supported models: {list(MODEL_FACTORY.keys())}")
    builder = MODEL_FACTORY[model_name]
    return builder(num_classes=num_classes, pretrained=pretrained, **kwargs)

