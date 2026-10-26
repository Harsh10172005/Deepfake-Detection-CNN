"""Single-image inference for the CNN deepfake detector.

Used both by the web app (app.py imports predict_image) and as a CLI:

    python inference.py path/to/image.jpg \
        --weights checkpoints/best_cnn_efficientnet_b0.pth
"""
import argparse

import numpy as np
import torch
from PIL import Image

from dataset import get_transforms
from models import get_model


def predict_image(image_path, model, transform, device):
    """Return the model's fake-probability (0..1) for one image file."""
    image = Image.open(image_path).convert('RGB')
    image = np.array(image)
    input_tensor = transform(image=image)['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output).item()

    return prob


def load_model(model_name, weights_path, device):
    """Build the CNN and (when given) load a train.py checkpoint into it.

    When a checkpoint is provided the backbone is built WITHOUT ImageNet
    weights (pretrained=False): the checkpoint overwrites every weight
    anyway, so downloading them would be wasted bandwidth.
    """
    model = get_model(model_name, pretrained=weights_path is None)
    if weights_path:
        checkpoint = torch.load(weights_path, map_location=device)
        state = (
            checkpoint['model_state_dict']
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint
            else checkpoint
        )
        model.load_state_dict(state)
    return model.to(device).eval()


def main(args):
    device = torch.device(
        'cuda' if torch.cuda.is_available()
        else ('mps' if torch.backends.mps.is_available() else 'cpu')
    )
    model = load_model(args.model_name, args.weights, device)
    _, val_transform = get_transforms(img_size=args.img_size)

    prob = predict_image(args.input_path, model, val_transform, device)
    print(f"Input Image: {args.input_path}")

    label = "Fake (AI-generated / manipulated)" if prob > 0.5 else "Real"
    conf = prob if prob > 0.5 else 1.0 - prob
    print(f"Prediction: {label} (Confidence: {conf:.4f}, Fake Probability: {prob:.4f})")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deepfake Detector Inference")
    parser.add_argument('input_path', type=str, help="Path to input image")
    parser.add_argument('--weights', type=str,
                        default='checkpoints/best_cnn_efficientnet_b0.pth',
                        help="Path to trained model weights .pth file")
    parser.add_argument('--model_name', type=str, default='efficientnet_b0',
                        help="CNN backbone (e.g., efficientnet_b0, efficientnet_v2_m, resnet50)")
    parser.add_argument('--img_size', type=int, default=224, help="Image size for model input")

    main(parser.parse_args())
