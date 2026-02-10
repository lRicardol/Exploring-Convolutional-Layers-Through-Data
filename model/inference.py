"""
Inference script for Fashion-MNIST CNN model.

This script loads a trained PyTorch model (model.pth),
performs inference on new Fashion-MNIST samples,
and returns predicted class labels.

Designed to be compatible with local execution
and Amazon SageMaker inference workflows.
"""

import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader


# Model Definition

class FashionMNISTCNN(nn.Module):
    """
    Convolutional Neural Network designed for Fashion-MNIST.
    Architecture must match the one used during training.
    """

    def __init__(self):
        super(FashionMNISTCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# Load Model

def load_model(model_path: str):
    """
    Loads the trained model from disk.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FashionMNISTCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    return model, device


# Data Loader (Inference)

def get_test_loader(batch_size=32):
    """
    Loads Fashion-MNIST test dataset for inference.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    test_dataset = FashionMNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return test_loader


# Inference Function

def run_inference(model, device, data_loader, num_samples=10):
    """
    Runs inference on a batch of test images and prints predictions.
    """

    class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]

    predictions = []

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            for i in range(min(num_samples, len(preds))):
                predictions.append({
                    "predicted_label": class_names[preds[i].item()],
                    "true_label": class_names[labels[i].item()]
                })

            break  # Only one batch for demonstration

    return predictions


# SageMaker-compatible Functions

def model_fn(model_dir):
    """
    Loads the model for SageMaker inference.
    """
    model_path = f"{model_dir}/model.pth"
    model, device = load_model(model_path)
    return model


def input_fn(request_body, content_type):
    """
    Parses input data for SageMaker endpoint.
    """
    if content_type == "application/json":
        data = json.loads(request_body)
        tensor = torch.tensor(data, dtype=torch.float32)
        return tensor
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    """
    Performs inference using the loaded model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_data = input_data.to(device)

    with torch.no_grad():
        outputs = model(input_data)
        _, predictions = torch.max(outputs, 1)

    return predictions.cpu().numpy()


def output_fn(prediction, content_type):
    """
    Formats the prediction output.
    """
    if content_type == "application/json":
        return json.dumps(prediction.tolist())
    else:
        raise ValueError(f"Unsupported content type: {content_type}")


# Local Execution

if __name__ == "__main__":
    model, device = load_model("model.pth")
    test_loader = get_test_loader()
    results = run_inference(model, device, test_loader)

    for r in results:
        print(f"Predicted: {r['predicted_label']} | True: {r['true_label']}")
