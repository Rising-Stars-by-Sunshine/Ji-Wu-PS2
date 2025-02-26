# House Price Prediction and Regression Discontinuity Analysis

This project demonstrates how to predict house prices using structured data (such as square footage, number of bedrooms, etc.) and perform **Regression Discontinuity Analysis** on the data. The dataset is from the **King County House Sales Dataset**, and the analysis focuses on exploring the impact of certain features (like house price) on house valuation.

## Requirements

Make sure you have the following Python packages installed to run the code:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- transformers (for NLP tasks)
- torch

You can install them via pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost transformers torch
```

## Dataset

The dataset used in this project is the **King County House Sales Dataset**, which contains various columns about house sales in King County. The dataset includes the following columns:

- **id**: Unique identifier for each house
- **date**: Date of the sale
- **price**: Price of the house
- **bedrooms**: Number of bedrooms
- **bathrooms**: Number of bathrooms
- **sqft_living**: Square footage of the living area
- **sqft_lot**: Lot size in square feet
- **floors**: Number of floors in the house
- **waterfront**: Whether the house is located on the waterfront
- **view**: Quality of the view
- **condition**: Condition of the house
- **grade**: Grade of the house
- **sqft_above**: Square footage of the house excluding the basement
- **sqft_basement**: Square footage of the basement
- **yr_built**: Year the house was built
- **yr_renovated**: Year the house was renovated (if applicable)
- **zipcode**: Zip code of the house
- **lat**: Latitude of the house
- **long**: Longitude of the house
- **sqft_living15**: Living area square footage of the nearest 15 neighbors
- **sqft_lot15**: Lot size of the nearest 15 neighbors

## Steps

### 1. **Data Preprocessing**

The first step is to preprocess the data:
- **Handle missing values**: Missing values are imputed with the median value.
- **Feature engineering**: A new feature, `price_per_sqft`, is calculated by dividing the price of the house by the living square footage.
- **Normalization and encoding**: Numerical features are normalized, and categorical variables (e.g., `zipcode`) are one-hot encoded.

### 2. **Predictive Model**

We build predictive models to forecast house prices:
- **Random Forest Regressor**: A decision tree-based model that handles non-linear relationships.
- **XGBoost Regressor**: A gradient-boosting model known for high performance.

### 3. **Regression Discontinuity Analysis (RDA)**

A **regression discontinuity design** is used to analyze how house prices respond to a specific threshold:
- **Threshold**: The threshold is assumed to be based on house price. For example, houses priced above $500,000 are considered in a high-income group, while others are considered in a low-income group.
- **Causal Inference**: We create a treatment variable based on the threshold and assess how house prices behave around that threshold.

### 4. **Evaluation**

The performance of the models is evaluated using:
- **RMSE (Root Mean Square Error)**
- **MAE (Mean Absolute Error)**
- **R² (Coefficient of Determination)**

### 5. **Results and Plotting**

The regression discontinuity results are plotted with a scatter plot of house prices against a threshold, with a red line showing the predicted values.

## How to Run

1. Download the `kc_house_data.csv` file and place it in the `data/` directory.
2. Open the Jupyter Notebook (`.ipynb` file) in the `code/` directory.
3. Run the notebook cells sequentially to execute the code.

### Example Output:
The model will output the RMSE, MAE, and R² for both **Random Forest** and **XGBoost** models. It will also generate a plot showing the regression discontinuity analysis.

## Customization

You can customize the following:
- **Threshold value** in Regression Discontinuity (currently set to `$500,000`).
- **Data columns**: If you have additional columns or wish to replace `income` with a different feature, adjust the notebook accordingly.
