# Insurance Premium Prediction Model

This project predicts medical insurance charges using a deep learning model built with TensorFlow and Keras. The model leverages personal and demographic data, such as age, sex, BMI, smoking status, region, and number of children, to estimate insurance premiums.

---

## Project Overview

The goal is to accurately predict insurance premiums based on a set of user attributes. The dataset used contains 3,630 rows for training and 492 rows for testing, encompassing features like age, sex, BMI, smoker status, region, and number of children.

---

## Features

The model uses the following features:
- **Age:** Numerical input representing the individual's age.
- **Sex:** Categorical input (male/female), one-hot encoded.
- **BMI:** Body Mass Index (numeric).
- **Children:** Number of dependents (numeric).
- **Smoker Status:** Categorical input (yes/no), one-hot encoded.
- **Region:** Categorical input for geographic location (southwest, southeast, northwest, northeast), one-hot encoded.

---

## Workflow

### 1. **Data Preprocessing**
   - **Categorical Encoding:** Convert sex, smoker status, and region into one-hot encoded numerical formats.
   - **Feature Scaling:** Prepare features for the neural network.
   - **Dataset Splitting:** Separate data into training (`Train_Data.csv`) and testing (`Test_Data.csv`).

### 2. **Model Architecture**
   - Input layer with 11 features.
   - Three hidden layers:
     - 64 neurons in the first layer.
     - 32 neurons in the second and third layers.
     - ReLU activation for all hidden layers.
   - Output layer with a single neuron for continuous predictions.
   - Mean Absolute Error (MAE) used as the loss function.
   - Adam optimizer employed for gradient descent.

### 3. **Training**
   - Training is performed with early stopping based on validation loss, using a patience of 20 epochs.
   - Model trains for a maximum of 300 epochs with validation on 10% of the training data.

### 4. **Evaluation**
   - Loss values are displayed for both training and validation datasets.
   - Predictions are visualized against actual charges for comparison.

### 5. **Prediction**
   - The trained model predicts charges for unseen test data.
   - Outputs include:
     - Input details (age, sex, BMI, smoker status, region).
     - Predicted insurance charges.

---

## How to Use

1. **Install Requirements:**
   ```bash
   pip install pandas numpy tensorflow matplotlib
   ```

2. **Prepare Data:**
   - Place `Train_Data.csv` and `Test_Data.csv` in the project directory.
   - Ensure data formatting matches the sample dataset provided.

3. **Run the Code:**
   Execute the script to preprocess data, train the model, and generate predictions.

4. **Visualize Results:**
   - Training and validation loss curves.
   - Comparisons between predicted and actual charges.

---

## Example Results

**Sample Prediction Output:**
```
AGE    SEX    BMI   SMOKER  REGION        CHARGES
23     male   22    yes     southwest     predicted --> $2134
45     female 30    no      northeast     predicted --> $4567
```

---

## Results & Performance

- **Training Loss:** The model achieves a low MAE, indicating good predictive capability.
- **Visualization:** Loss curves show how the model generalizes to unseen data.
- **Comparison:** Predicted charges closely align with actual charges in the dataset.

![image](https://github.com/user-attachments/assets/d590cb23-c6bd-4c07-8c20-f1adf867e30e)
![image](https://github.com/user-attachments/assets/bd5a8949-b34a-41c2-9555-7fb317a3a131)


---

## Acknowledgments

- Dataset sourced from [insert source link].
- Thanks to contributors of TensorFlow, Keras, and Python for enabling this project.
