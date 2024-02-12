# Gold Price Prediction using Random Forest Regressor

This repository contains Python code that uses a Random Forest Regressor to predict gold prices based on historical data. The dataset used for training and testing is loaded from a CSV file named "gld_price_data.csv." The code utilizes popular data science and machine learning libraries, including NumPy, Pandas, Matplotlib, Seaborn, and Scikit-learn.

## Getting Started

1. Clone the repository:
   ```bash
   git clone [https://github.com/your_username/your_repository.git](https://github.com/OmarAshor/Gold-Price-Prediction.git)
   ```

2. Install the required libraries:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```

3. Download the dataset "gld_price_data.csv" and place it in the same directory as the code.

4. Run the code:
   ```bash
   python gold_price_prediction.py
   ```

## Code Overview

### Data Loading and Exploration

The code starts by importing necessary libraries and loading the gold price dataset into a Pandas DataFrame.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics

GoldData = pd.read_csv("gld_price_data.csv")
```

It then performs basic data exploration, displaying the first and last 5 rows, checking the data shape, providing basic information, checking for missing values, and showing statistical measures of the data.

### Data Visualization

The code creates a heatmap to visualize the correlation matrix of the features, especially focusing on the correlation with the target variable 'GLD.' It also displays a distribution plot for the 'GLD' variable.

```python
correlation = GoldData.corr()

plt.figure(figsize=(8,8))
sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size': 8}, cmap='Greens')
print(correlation['GLD'])

sns.displot(GoldData['GLD'] , color='red')
```

### Data Preprocessing

The features and target variable are separated into 'x' and 'y.' The dataset is then split into training and testing sets using the `train_test_split` method.

```python
x = GoldData.drop(['Date', 'GLD'], axis=1)
y = GoldData['GLD']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)
```

### Model Training and Prediction

A Random Forest Regressor is used to train the model on the training set, and predictions are made on the test set.

```python
regressor = RandomForestRegressor(n_estimators=100)
regressor.fit(x_train, y_train)
test_data_prediction = regressor.predict(x_test)
```

### Model Evaluation and Visualization

The code evaluates the model's performance using the R-squared error and visualizes the actual vs. predicted values using a line plot.

```python
error_score = metrics.r2_score(y_test, test_data_prediction)
print("R Squared Error:", error_score * 100)

y_test = list(y_test)

plt.plot(y_test, color='black', label='Actual Value')
plt.plot(test_data_prediction, color='blue', label='Predicted Value')
plt.title('Actual Value VS Predicted Value')
plt.xlabel('Number of Values')
plt.ylabel('Gold Price')
plt.legend()
plt.show()
```


## Acknowledgments

- The code was inspired by a passion for data science and machine learning.
- Thanks to the open-source community for providing valuable tools and resources.
