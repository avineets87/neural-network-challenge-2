# Employee Attrition Prediction with a Multi-Output Neural Network

## Overview

This project builds a multi-output neural network model to predict employee attrition and department based on various features. The model utilizes TensorFlow/Keras and scikit-learn for preprocessing, training, and evaluation.

## Dataset

The dataset used for this project is the "attrition.csv" file, which contains information about employee demographics, job satisfaction, work-life balance, and other relevant factors. The dataset can be found in the Google Colab notebook (check the code).

## Preprocessing

The following preprocessing steps were applied to the data:

1. **Data Selection**: Relevant features were selected for the input (X) and output (y) data.
2. **Data Encoding**: Categorical features like 'OverTime' were encoded using Label Encoding, while 'Department' and 'Attrition' were one-hot encoded.
3. **Data Scaling**: Numerical features were scaled using StandardScaler to ensure consistent ranges.
4. **Data Splitting**: The data was split into training and testing sets using `train_test_split` for model evaluation.

## Model Architecture

A multi-output neural network was designed with the following structure:

- **Input Layer**: Accepts the preprocessed features.
- **Shared Layers**: Two hidden layers with ReLU activation, shared between both output branches.
- **Department Branch**: A hidden layer with ReLU activation followed by an output layer with softmax activation for department prediction.
- **Attrition Branch**: A hidden layer with ReLU activation followed by an output layer with softmax activation for attrition prediction.

## Training and Evaluation

The model was compiled using the Adam optimizer and categorical cross-entropy loss for both output branches. Accuracy was used as the evaluation metric. The model was trained for 10 epochs with a batch size of 32. The training process included a validation split for monitoring performance.

## Results

The model achieved the following accuracy scores on the testing data:

- Department Accuracy: [Insert the actual accuracy value from the notebook]
- Attrition Accuracy: [Insert the actual accuracy value from the notebook]

## Summary and Potential Improvements

The multi-output model demonstrates the ability to predict both department and attrition with reasonable accuracy. However, there are several areas for potential improvement:

1. **Metrics**: Consider using more robust metrics like precision, recall, F1-score, or AUC-ROC, especially if class imbalance exists.
2. **Feature Engineering**: Explore additional features and preprocessing techniques to enhance model performance.
3. **Model Architecture**: Experiment with different architectures, layer sizes, and activation functions.
4. **Hyperparameter Tuning**: Optimize hyperparameters like learning rate, batch size, and optimizer settings.
5. **Handling Class Imbalance**: Address potential class imbalance issues using techniques like class weighting or oversampling.


## Usage

To use this model, follow these steps:

1. Open the Google Colab notebook.
2. Ensure you have the necessary libraries installed (TensorFlow, scikit-learn, pandas, etc.).
3. Run the code cells to preprocess the data, build, train, and evaluate the model.
4. Modify or extend the code as needed for your specific application.
