# Insurance Premium Prediction Model

This project uses a deep learning model to predict medical insurance charges based on various personal and demographic features such as age, sex, BMI, smoking status, region, and number of children. The model is built using TensorFlow and Keras, with data preprocessed and formatted for training a neural network.

## Project Overview

The goal of this project is to predict medical insurance charges based on user features like age, sex, body mass index (BMI), smoking status, region, and the number of children. The dataset is used to train a neural network, which can then make predictions on unseen test data.

## Features Used

The model uses the following features to predict insurance charges:
- `age`: The age of the individual.
- `sex`: The gender of the individual (male or female).
- `bmi`: The Body Mass Index of the individual.
- `smoker`: Whether the individual is a smoker (`yes` or `no`).
- `region`: The region where the individual resides (southwest, northeast, southeast, northwest).
- `children`: The number of children or dependents.

## Data

The dataset used in this project comes from a medical insurance dataset containing information on various factors that influence the cost of insurance premiums. The dataset is split into a training set and a test set:
- **Training Data**: Used to train the neural network model.
- **Test Data**: Used to test the accuracy and performance of the model on unseen data.

## Requirements

The following libraries are required to run this project:
- pandas
- numpy
- tensorflow
- matplotlib

You can install the required libraries using pip:

```bash
pip install pandas numpy tensorflow matplotlib
```

## How to Run the Code

1. **Download the Dataset:**
   You will need the `Train_Data.csv` and `Test_Data.csv` files. Make sure the dataset is placed in the appropriate folder.

2. **Data Preprocessing:**
   The code begins by loading and cleaning the data. Categorical variables like sex, smoker status, and region are transformed using one-hot encoding into numerical format suitable for neural network training.

3. **Model Architecture:**
   A deep neural network is built using Keras with the following architecture:
   - Input layer: 11 input features (age, sex, BMI, smoker status, children, and region).
   - 3 hidden layers with ReLU activation and 64, 32, and 32 neurons respectively.
   - Output layer: A single neuron for the continuous prediction of insurance charges.
   
4. **Training:**
   The model is trained using Mean Absolute Error (MAE) as the loss function and the Adam optimizer. Early stopping is used to avoid overfitting, and the model is trained for a maximum of 300 epochs.

5. **Evaluation and Prediction:**
   The model is evaluated on the training dataset, and predictions are made on the test set. These predictions are displayed alongside the actual values for comparison.

## Example Output

Once the model is trained and tested, it will output predictions like this:

```
AGE    SEX    BMI   SMOKER  REGION        CHARGES
23     male   22    yes     southwest     predicted --> $ 2134
45     female 30    no      northeast     predicted --> $ 4567
...
```

The output will show predicted charges along with the corresponding inputs like age, sex, BMI, smoker status, and region.

## Results

- **Training Loss:** After training, the loss on the training set is displayed.
- **Predictions:** The model makes predictions on the test data, and you can compare predicted insurance charges against the actual values.

## Conclusion

This deep learning model effectively predicts medical insurance charges based on personal and demographic features. The model can be further improved by fine-tuning hyperparameters or adding more features.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The dataset used in this project was sourced from [insert data source here].
- Thanks to the contributors of the Keras and TensorFlow libraries.
