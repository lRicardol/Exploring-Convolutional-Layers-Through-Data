import torch
import torch.nn as nn
import numpy as np

class FashionMNISTCNN(nn.Module):
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
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.fc_layers(x)
        return x



def load_model(model_path="model.pth"):
    model = FashionMNISTCNN()
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    return model



def preprocess_input(pixel_array):
    """
    pixel_array: array-like of shape (784,)
    Returns tensor of shape (1, 1, 28, 28)
    """

    x = np.array(pixel_array, dtype=np.float32)

    # Normalize [0,255] -> [0,1]
    x = x / 255.0

    # Reshape to CNN format
    x = x.reshape(1, 1, 28, 28)

    return torch.tensor(x)



def predict(pixel_array, model_path="model.pth"):
    model = load_model(model_path)
    x = preprocess_input(pixel_array)

    with torch.no_grad():
        outputs = model(x)
        prediction = torch.argmax(outputs, dim=1).item()

    return prediction



CLASS_NAMES = {
    0: "T-shirt / Top",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle boot"
}


if __name__ == "__main__":

    # Example: dummy input (replace with a row from CSV)
    dummy_pixels = np.zeros(784)

    pred_class = predict(dummy_pixels)
    print("Predicted class ID:", pred_class)
    print("Predicted label:", CLASS_NAMES[pred_class])
