"""CNN model definitions for deepfake / AI-generated image detection.

This project deliberately focuses on a single CNN family (EfficientNet, with
ResNet-50 as an alternative backbone) instead of a multi-model ensemble: one
architecture that is fast to run on CPU, straightforward to explain, and
honestly evaluated end-to-end (see evaluate.py and results/).

The backbone's classifier head is replaced with dropout + a single-logit
linear layer, which matches the checkpoints written by train.py.
"""
import torch.nn as nn
import torchvision.models as models

_EFFICIENTNET_BUILDERS = {
    'efficientnet_b0': (models.efficientnet_b0, models.EfficientNet_B0_Weights),
    'efficientnet_v2_s': (models.efficientnet_v2_s, models.EfficientNet_V2_S_Weights),
    'efficientnet_v2_m': (models.efficientnet_v2_m, models.EfficientNet_V2_M_Weights),
}

SUPPORTED_MODELS = tuple(_EFFICIENTNET_BUILDERS) + ('resnet50',)


class CNNDeepfakeDetector(nn.Module):
    """Binary (real vs fake) classifier on a torchvision CNN backbone."""

    def __init__(self, model_name='efficientnet_b0', pretrained=True):
        super().__init__()
        if model_name in _EFFICIENTNET_BUILDERS:
            builder, weights_enum = _EFFICIENTNET_BUILDERS[model_name]
            weights = weights_enum.DEFAULT if pretrained else None
            self.backbone = builder(weights=weights)
            num_ftrs = self.backbone.classifier[1].in_features
            self.backbone.classifier[1] = nn.Sequential(
                nn.Dropout(p=0.4, inplace=True),
                nn.Linear(num_ftrs, 1),
            )
        elif model_name == 'resnet50':
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
            num_ftrs = self.backbone.fc.in_features
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=0.4, inplace=True),
                nn.Linear(num_ftrs, 1),
            )
        else:
            raise ValueError(
                f"Unsupported model: {model_name}. Supported: {', '.join(SUPPORTED_MODELS)}"
            )

    def forward(self, x):
        return self.backbone(x).squeeze(1)


def get_model(model_name='efficientnet_b0', pretrained=True):
    """Build a CNNDeepfakeDetector for one of the SUPPORTED_MODELS."""
    return CNNDeepfakeDetector(model_name=model_name, pretrained=pretrained)


if __name__ == '__main__':
    import torch

    x = torch.randn(2, 3, 224, 224)
    for name in SUPPORTED_MODELS:
        model = get_model(name, pretrained=False)
        print(f"{name}: output shape {model(x).shape}")
