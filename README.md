# Fashion-MNIST Classification System using PyTorch 🧥👗👟

A deep learning model built using PyTorch to classify images from the Fashion-MNIST dataset. This project is part of the **Deep Learning with Python** course on Coursera and showcases the classification of clothing items such as shirts, trousers, dresses, and shoes. The model utilizes Convolutional Neural Networks (CNNs) to perform image classification on a dataset of 60,000 training images and 10,000 test images.

## Features 🌟

- **Fashion-MNIST Dataset**: 📸 Classifies 10 types of clothing items from grayscale images.
- **PyTorch Implementation**: 🔥 Built using PyTorch, a powerful deep learning framework.
- **Convolutional Neural Network (CNN)**: 🧠 Uses a CNN model to extract features and classify images.
- **Accuracy Evaluation**: 📊 Evaluates model performance using accuracy metrics on the test dataset.
- **Training Visualization**: 📈 Visualizes training progress and loss over epochs.

## Technologies Used 🛠️

- **PyTorch**: 🔥 For building and training the deep learning model.
- **NumPy**: 🧮 For numerical computations and array manipulation.
- **Matplotlib**: 📊 For plotting loss and accuracy graphs.
- **Fashion-MNIST Dataset**: 🧥 A dataset containing 60,000 training images and 10,000 test images of clothing items.
- **Python 3.x**: 🐍 Programming language for building the classification system.

## How It Works 🚀

1. **Dataset Loading**: 🧥 The Fashion-MNIST dataset is loaded using PyTorch’s `torchvision` module.
2. **Data Preprocessing**: 🧹 The images are normalized and transformed into tensors suitable for the CNN model.
3. **Model Architecture**: 🧠 A Convolutional Neural Network (CNN) is designed with multiple layers of convolution, pooling, and fully connected layers.
4. **Training the Model**: 🚀 The model is trained using the training dataset, and loss is calculated at each epoch to monitor progress.
5. **Model Evaluation**: 📊 After training, the model is evaluated on the test dataset for accuracy.
6. **Visualization**: 📈 The training process and performance metrics (like accuracy and loss) are visualized for better insight into the model’s performance.

## Setup & Installation ⚙️

### Prerequisites

- **Python 3.x**
- **PyTorch**: Install PyTorch by following the instructions at [PyTorch Installation](https://pytorch.org/get-started/locally/).
- **Matplotlib**: For plotting graphs.

### Steps to Install

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/Fashion-MNIST-Classification-System-using-PyTorch.git
   cd Fashion-MNIST-Classification-System-using-PyTorch
   ```

2. **Install Dependencies:**
   Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter Notebook or Python Script:**
   You can run the project either by using a Jupyter notebook or by running the main Python script:
   ```bash
   jupyter notebook
   ```
   OR
   ```bash
   python main.py
   ```

### Running the Project 🏃

1. **Start Training**:
   - The model will begin training on the Fashion-MNIST dataset.
   - You will see the training progress with each epoch showing the loss and accuracy metrics.
   
2. **Evaluation**:
   - Once training is complete, the model will evaluate its accuracy on the test dataset and display the results.

3. **Model Visualization**:
   - The training loss and accuracy will be plotted after training for a better understanding of model performance.

## Example Output 🎬

### Example Input:
An image of a clothing item (e.g., a pair of sneakers).

### Example Output:
```
Predicted Label: Sneaker
Accuracy: 93.2%
```

### Sample Plots:
- **Training Loss**: Shows the change in loss across training epochs.
- **Accuracy Plot**: Visualizes how accuracy increases with each epoch during training.


## License 📝

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements 🙏

- 🎓 Coursera’s **Deep Learning with Python** course for inspiring and guiding this project.
- 🧠 [PyTorch](https://pytorch.org/) for providing a powerful deep learning framework.
- 📚 [Fashion-MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist) for a well-structured image classification dataset.

