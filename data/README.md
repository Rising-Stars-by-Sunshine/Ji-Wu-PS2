# **King County House Price Dataset**

## **📌 Dataset Overview**
This dataset contains real estate transaction records from **King County, USA**, including **21,614** housing records with details on house prices, features, and geographic locations. The dataset is widely used for **house price prediction, real estate market analysis, and machine learning applications**.

## **📍 Data Source**
- **Original Source**: King County, Washington, Public Real Estate Transactions
- **Collection Period**: Historical transaction records
- **Data Format**: CSV (Comma-Separated Values)
- **Primary Use**: Price prediction modeling and spatial real estate analysis

## **📊 Data Dictionary (Feature Descriptions)**
This dataset includes the following columns:

| **Column Name**   | **Description** |
|-------------------|----------------|
| `id`             | Unique identifier for each house |
| `date`           | Sale date (YYYYMMDD format) |
| `price`          | Sale price of the house (in USD) |
| `bedrooms`       | Number of bedrooms in the house |
| `bathrooms`      | Number of bathrooms (including partial baths) |
| `sqft_living`    | Square footage of the interior living space |
| `sqft_lot`       | Square footage of the lot |
| `floors`         | Number of floors in the house |
| `waterfront`     | Whether the house has a waterfront view (0 = No, 1 = Yes) |
| `view`           | Number of times the house has been viewed/rated |
| `condition`      | Overall condition rating (1 = Poor, 5 = Excellent) |
| `grade`          | House quality grade (based on construction & design) |
| `sqft_above`     | Square footage of house above ground (excluding basement) |
| `sqft_basement`  | Square footage of the basement |
| `yr_built`       | Year the house was built |
| `yr_renovated`   | Year of last renovation (0 if never renovated) |
| `zipcode`        | ZIP code of the house location |
| `lat`            | Latitude coordinate of the property |
| `long`           | Longitude coordinate of the property |
| `sqft_living15`  | Average living space in the nearest 15 houses |
| `sqft_lot15`     | Average lot space in the nearest 15 houses |

## **📈 Applications & Use Cases**
This dataset is commonly used for:
- **Machine Learning & AI**: Predicting house prices using regression models
- **Real Estate Market Analysis**: Understanding pricing trends and location-based variations
- **Geospatial Analysis**: Visualizing house price distributions and neighborhood trends
- **Feature Engineering**: Developing new predictors from existing data to enhance model accuracy

## **🔍 Important Notes**
- The dataset is **clean**, but exploratory data analysis (EDA) is recommended to handle potential outliers.
- House prices are recorded in **USD**.
- Geographic coordinates (`lat`, `long`) allow for spatial analysis and mapping.

Here is the link to [the original web page of the dataset](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction).
