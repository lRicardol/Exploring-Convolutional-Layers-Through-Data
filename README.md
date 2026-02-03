# Exploring Convolutional Layers Through Data and Experiments  
## Fashion-MNIST Case Study

### Digital Transformation and Enterprise Solutions (TDSE)

---

## 1. Problem Description

Neural networks are often treated as black-box models, but in enterprise and large-scale systems, **architectural decisions must be understood, justified, and governed**.

This laboratory explores **convolutional layers** as a concrete architectural choice that introduces **inductive bias** into learning systems. Instead of following a predefined recipe, the goal is to **analyze, design, and experimentally validate** convolutional architectures using a real-world image dataset.

The project compares:
- A **baseline non-convolutional neural network**
- A **custom-designed Convolutional Neural Network (CNN)**

The analysis focuses on how convolutional design choices affect:
- Learning efficiency
- Parameter usage
- Generalization
- Interpretability

This work aligns with **Digital Transformation and Enterprise Solutions (TDSE)** principles by treating neural networks as **modular, explainable, and deployable architectural components**.

---

## 2. Dataset Description

**Dataset:** Fashion-MNIST  
**Source:**  
- TensorFlow / Keras Datasets  
- Original format: IDX binary files  
- CSV representations for exploratory analysis

Fashion-MNIST is a standardized benchmark dataset consisting of grayscale images of clothing items, designed as a more challenging drop-in replacement for MNIST digits.

### Dataset Characteristics
- **Training samples:** 60,000
- **Test samples:** 10,000
- **Image size:** 28 × 28 pixels
- **Channels:** 1 (grayscale)
- **Number of classes:** 10

### Classes
| Label | Description |
|-----|------------|
| 0 | T-shirt / top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

### Why Fashion-MNIST is Suitable for CNNs
- Image-based, spatially structured data
- Local patterns (edges, textures, shapes)
- Translation-invariant features
- Moderate complexity without excessive computational cost

These properties make Fashion-MNIST an ideal dataset for studying **convolutional inductive bias**.

---

## 3. Repository Structure

fashion-mnist-convolutional-experiments/

│

├── README.md

│

├── data/

│ ├── raw/

│ │ ├── train-images-idx3-ubyte

│ │ ├── train-labels-idx1-ubyte

│ │ ├── t10k-images-idx3-ubyte

│ │ └── t10k-labels-idx1-ubyte

│ │

│ ├── processed/

│ │ ├── fashion-mnist_train.csv

│ │ └── fashion-mnist_test.csv

│

├── notebooks/

│ └── fashion_mnist_cnn_analysis.ipynb

│

├── models/

│ └── cnn_fashion_mnist/

│

├── sagemaker/

│ ├── train.py

│ ├── inference.py

│ └── requirements.txt

│

├── environment/

│ └── requirements.txt


---

## 4. Dataset Exploration (EDA)

A concise exploratory analysis is performed to understand the structure of the dataset rather than exhaustively analyze statistics.

The EDA includes:
- Dataset size and train/test split
- Image dimensions and channel structure
- Class distribution analysis
- Visualization of sample images per class
- Normalization of pixel values to the range [0, 1]

**Goal:**  
Ensure the dataset is correctly structured and suitable for convolutional architectures.

---

## 5. Baseline Model (Non-Convolutional)

A baseline neural network is implemented **without convolutional layers** to establish a reference point.

### Architecture
- Input: Flattened 28×28 image (784 features)
- Dense hidden layers
- ReLU activations
- Softmax output layer

### Evaluation
- Training and validation accuracy
- Loss curves
- Number of trainable parameters

### Observed Limitations
- Large number of parameters
- No spatial awareness
- Reduced generalization compared to CNNs
- Sensitivity to small spatial variations

This model serves as a control for evaluating the benefits of convolutional layers.

---

## 6. Convolutional Neural Network Architecture

A custom CNN is designed **from scratch**, with explicit architectural reasoning.

### Design Choices
- Convolutional layers with small kernels
- Increasing number of filters with depth
- ReLU activations
- Pooling layers for spatial downsampling
- Fully connected classifier head

### Justification
- Local receptive fields capture spatial patterns
- Weight sharing reduces parameter count
- Pooling introduces translation invariance
- Shallow depth balances performance and interpretability

The architecture is intentionally simple but expressive.

---

## 7. Controlled Convolutional Experiment

A controlled experiment is conducted to isolate the effect of a single convolutional design parameter.

### Experiment Focus (Example)
- **Kernel size:** 3×3 vs 5×5

### Experimental Setup
- All other hyperparameters kept constant
- Same optimizer, learning rate, and training epochs
- Same dataset splits

### Results
- Quantitative comparison (accuracy, loss)
- Parameter count differences
- Training stability observations

### Trade-offs
- Larger kernels capture broader context
- Smaller kernels improve efficiency and composability
- Performance vs computational cost considerations

---

## 8. Interpretation and Architectural Reasoning

### Why CNNs Outperform the Baseline
Convolutional layers exploit spatial locality and translation invariance, which fully connected layers ignore.

### Inductive Bias Introduced by Convolution
- Local connectivity
- Shared weights
- Hierarchical feature extraction

This bias aligns well with image data and leads to better generalization.

### When Convolution Is Not Appropriate
- Tabular data without spatial structure
- Highly irregular graph-based data
- Problems where spatial locality has no semantic meaning

Architectural choices must always reflect data structure.

---

## 9. Deployment with Amazon SageMaker

The final CNN model is trained and deployed using **Amazon SageMaker** to simulate a production-grade workflow.

### Deployment Steps
1. Upload notebook and dataset to SageMaker Studio
2. Train the CNN model in the cloud environment
3. Save the trained model artifacts
4. Deploy a real-time inference endpoint
5. Invoke the endpoint with sample inputs

### Example Inference
> **Input:** Fashion-MNIST image (28×28)  
> **Output:** Predicted class = "Sneaker"  
> **Confidence:** 0.91

Screenshots and endpoint evidence are included in the `images/` directory.

---

## 10. Conclusions

- Convolutional layers provide strong inductive bias for image-based tasks
- CNNs outperform dense baselines in both efficiency and accuracy
- Architectural reasoning is more important than hyperparameter tuning
- Controlled experiments reveal meaningful trade-offs
- Cloud deployment enables scalable and reproducible ML workflows

This laboratory demonstrates how **AI-driven architectures support Digital Transformation** by combining theory, experimentation, and enterprise deployment.

---

## 11. Course Context

This project was developed as part of:

**Digital Transformation and Enterprise Solutions (TDSE)**  
*Neural Networks and Deep Learning Module*

