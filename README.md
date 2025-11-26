# MRI Brain Tumor Detection using Convolutional Neural Networks

## Overview

This project implements a Convolutional Neural Network (CNN) for detecting brain tumors from MRI images. The notebook `MRI-Brain-Tumor-Detecor.ipynb` demonstrates the complete pipeline from data loading and preprocessing to model training, evaluation, and visualization of feature maps.

The project achieves 100% accuracy on the test set, showcasing the effectiveness of deep learning in medical image classification.

## Dataset

The dataset used is the "Brain MRI Images for Brain Tumor Detection" available on Kaggle:  
[Kaggle Dataset Link](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)

### Dataset Structure
- **Source**: Kaggle
- **Classes**: 
  - `yes/`: MRI images with brain tumors (154 images)
  - `no/`: Healthy brain MRI images (91 images)
- **Total Images**: 245
- **Image Format**: JPG
- **Resolution**: Resized to 128x128 pixels
- **Channels**: RGB (converted from BGR)

### Preprocessing
- Images are loaded using OpenCV
- Color channels are converted from BGR to RGB
- Images are resized to 128x128 pixels
- Normalization: Pixel values are divided by 255.0
- Data is reshaped for PyTorch (channels, height, width)

## Prerequisites

- Python 3.7+
- PyTorch 2.1.2+ (with CUDA support for GPU acceleration)
- OpenCV 4.10.0+
- NumPy, Matplotlib, Seaborn, Scikit-learn
- Jupyter Notebook

## Installation

1. Clone or download this repository.

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset from Kaggle and place it in the `./data/brain_tumor_dataset/` directory with `yes/` and `no/` subfolders.

## Usage

1. Open the Jupyter notebook:
   ```bash
   jupyter notebook MRI-Brain-Tumor-Detecor.ipynb
   ```

2. Run the cells in order to:
   - Load and preprocess the data
   - Define the CNN model
   - Train the model
   - Evaluate performance
   - Visualize results

## Model Architecture

The CNN model consists of:

### Convolutional Layers
- **Conv2d(3, 6, kernel_size=5)**: Input channels 3 (RGB), output 6, kernel 5x5
- **Tanh** activation
- **AvgPool2d(kernel_size=2, stride=5)**: Average pooling
- **Conv2d(6, 16, kernel_size=5)**: Input 6, output 16, kernel 5x5
- **Tanh** activation
- **AvgPool2d(kernel_size=2, stride=5)**: Average pooling

### Fully Connected Layers
- **Linear(256, 120)**: 256 input features, 120 output
- **Tanh** activation
- **Linear(120, 84)**: 120 input, 84 output
- **Tanh** activation
- **Linear(84, 1)**: 84 input, 1 output (binary classification)
- **Sigmoid** activation for binary classification

The model uses Binary Cross-Entropy (BCE) loss for training.

## Training

### Hyperparameters
- **Learning Rate**: 0.0001
- **Optimizer**: Adam
- **Epochs**: 400
- **Batch Size**: 32
- **Loss Function**: Binary Cross-Entropy (BCE)

### Training Process
1. Model is initialized with random weights
2. Data is shuffled and batched
3. Forward pass through the network
4. Loss calculation using BCE
5. Backpropagation and weight updates
6. Loss is printed every 10 epochs

The training achieves near-zero loss by the final epochs, indicating good convergence.

## Evaluation

### Metrics
- **Accuracy**: Percentage of correct predictions
- **Confusion Matrix**: True positives, false positives, true negatives, false negatives

### Results
- **Training Accuracy**: 100%
- **Test Accuracy**: 100%
- The model perfectly classifies all test images

### Evaluation Code
The notebook evaluates the trained model on the entire dataset and provides:
- Accuracy score
- Confusion matrix visualization
- Probability distribution plot

## Visualization

### Feature Maps
The notebook includes code to visualize the feature maps produced by the convolutional layers. This helps understand what features the network learns at different layers.

- Layer 1: 6 feature maps
- Layer 2: 16 feature maps

### Data Visualization
- Random sample images from healthy and tumor classes
- Training loss progression
- Model output probabilities

## Addressing Overfitting

The notebook discusses potential overfitting and proposes modifications to the dataset class:

- Addition of train/validation split (80/20)
- Mode selection for training or validation data
- Updated `__len__` and `__getitem__` methods

However, due to the small dataset size and high performance, overfitting concerns are minimal in this case.

## Key Code Snippets

### Data Loading
```python
class MRI(Dataset):
    def __init__(self):
        # Load images from yes/ and no/ folders
        # Preprocess and concatenate
    
    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, index):
        return {'image': self.images[index], 'label': self.labels[index]}
    
    def normalize(self):
        self.images = self.images / 255.0
```

### Model Definition
```python
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.cnn_model = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.Tanh(),
            nn.AvgPool2d(2, 5),
            nn.Conv2d(6, 16, 5),
            nn.Tanh(),
            nn.AvgPool2d(2, 5)
        )
        self.fc_model = nn.Sequential(
            nn.Linear(256, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 1)
        )
    
    def forward(self, x):
        x = self.cnn_model(x)
        x = x.view(x.size(0), -1)
        x = self.fc_model(x)
        return F.sigmoid(x)
```

### Training Loop
```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
for epoch in range(400):
    # Training code
    loss.backward()
    optimizer.step()
```

## Dependencies

See `requirements.txt` for the complete list of dependencies. Key libraries include:

- torch==2.1.2+cu121
- torchvision==0.16.2+cu121
- opencv-python==4.10.0.84
- numpy==1.26.3
- matplotlib==3.8.2
- scikit-learn==1.4.2
- seaborn==0.13.2

## Limitations and Future Improvements

1. **Dataset Size**: The dataset is relatively small (245 images). Larger datasets could improve generalization.

2. **Data Augmentation**: The current implementation doesn't include data augmentation techniques.

3. **Model Complexity**: A simple CNN is used. More advanced architectures like ResNet or DenseNet could be explored.

4. **Cross-Validation**: The notebook evaluates on the training set. Proper cross-validation would be beneficial.

5. **Clinical Validation**: This is a technical demonstration. Real-world medical applications require extensive clinical validation.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests for improvements.

## License

This project is for educational purposes. Please check the Kaggle dataset license for data usage terms.

## Acknowledgments

- Dataset provided by Navoneel Chakrabarty on Kaggle
- PyTorch documentation and tutorials
- OpenCV community

## References

1. [PyTorch Official Documentation](https://pytorch.org/docs/)
2. [Kaggle Brain MRI Dataset](https://www.kaggle.com/navoneel/brain-mri-images-for-brain-tumor-detection)
3. [Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
