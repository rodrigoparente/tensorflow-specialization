# Training with the cats vs. dogs dataset

- Kaggle is an online platform where machine learning challenges are posted, often offering prizes.
- The "Cats vs. Dogs" challenge is a famous example that requires building a classifier to distinguish between cats and dogs.
- The techniques already learned in the previous weeks can be applied directly to this problem, offering a head start.

## Working through the notebook

Here is a step-by-step guide for training a DNN to solve the problem in question. You can find the link to the notebook [here](./Labs/Ungraded/W1/C2_W1_Lab_1_cats_vs_dogs.ipynb).

1. **Data Preparation**:
    - Import packages and organize the filtered dataset into training and validation folders for cats and dogs (1000 training, 500 validation images per class).
    - Visualize random images using `matplotlib` to check diversity in the dataset.

2. **Neural Network Setup**:
   - Build the network with a rescaling layer to ensure uniform input scaling.
   - The model’s summary shows image size reduction through convolution and pooling layers.

3. **Model Compilation**:
   - Compile the model with a loss function and optimizer, and load the datasets.
   - Implement caching, prefetching, and shuffling for efficient data processing.

4. **Training**:
   - Training takes about a minute. The model achieves reasonable accuracy, with validation results leveling out around 75%.

5. **Predictions**:
   - Test the model with five uploaded images.
   - The model performs well, even with partially obscured animals or complex backgrounds, but misclassifies one image due to possible confusion with dark colors.


## Visualizing the effect of the convolutions

6. **Exploring the CNN Layers**:
   - The model layers API allows access to and visualization of outputs from each convolutional and pooling layer.
   - A visualization model processes a random image, displaying feature maps in a grid for better visibility.
   - This highlights specific image features (e.g., a dog's ears or a cat's tail), showing what the CNN emphasizes during classification.
   - This method provides insight into how the model interprets and distinguishes image features.

## Looking at accuracy and loss

7. **Tracking History**:
   - The `model.fit` function is assigned to a `history` object, which tracks training and validation metrics (accuracy and loss).
   - This data is used to plot training accuracy, which improves steadily, while validation accuracy plateaus around 0.7-0.75 after two epochs.
   - The training loss decreases, but validation loss increases, indicating overfitting.
   - The model reaches about 75% validation accuracy, making further training unnecessary.
   - Using the full dataset may improve results, with more optimization options to be explored in the next lesson.