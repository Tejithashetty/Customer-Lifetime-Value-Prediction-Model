# Project 1:
# Customer Lifetime Value Prediction

This project focuses on predicting **Customer Lifetime Value (LTV)** using historical purchase and demographic data. The predicted LTV is used to assist in **targeted marketing** and **customer segmentation**.

---

## Project Files

| Filename                | Description |
|------------------------|-------------|
| `clvp.ipynb`            | Main Python notebook containing all preprocessing, model training, evaluation, and visualization steps. |
| `customer_segmentation.csv` | Input dataset with customer features used for training and prediction. |
| `ltv_predictions.csv`   | Final output CSV containing predicted LTV values for each customer. |
| `trained_ltv_model.pkl` | Trained XGBoost regression model saved using `pickle`. |
| `visuals.pdf`           | Report with visualizations and charts used in analysis and insights. |

---

## Tech Stack

- Python
- Scikit-learn
- XGBoost
- Pandas, NumPy
- Matplotlib, Seaborn
- Jupyter Notebook

---

## Workflow

1. **Data Preprocessing**  
   - Missing value handling  
   - Encoding categorical variables  
   - Feature engineering: Frequency, Recency, AOV (Average Order Value)

2. **Model Training**  
   - Train `XGBRegressor` on customer data  
   - Evaluate using `MAE`, `RMSE`, and `R² Score`

3. **Prediction**  
   - Predict LTV for each customer  
   - Export predictions to `ltv_predictions.csv`

4. **Customer Segmentation**  
   - Segment customers into groups based on LTV range (Low, Medium, High)  
   - Visualized in `visuals.pdf`

# Project 2
# Movie Success Prediction & Sentiment Analysis

This project predicts movie success using IMDB data and analyzes viewer sentiments from movie overviews.  
We use Python, NLTK (VADER), and machine learning models to gain insights into what makes a movie successful!

---
## Project Files

| Filename                | Description |
|------------------------|------------------------|
| `moviesuccess.ipynb`|  Main Python notebook containing all preprocessing, model training, evaluation, and visualization steps. |
| `imdb_top_1000.csv` |  Input dataset used for training and prediction. |
| `Predictive Model Summary + Sentiment Visuals`   |Report with visualizations and charts used in analysis and insights. |
 


## Project Overview

- **Dataset:** `imdb_top_1000.csv`
- **Objective:**  
  - Predict the success of movies based on key features.
  - Analyze genre-wise sentiment trends from movie overviews.

---

##  Tools and Libraries

- Python 3
- Pandas, Numpy
- Matplotlib, Seaborn
- Scikit-Learn
- NLTK (VADER Sentiment Analysis)

---

## Key Features Used for Prediction

- `Gross Revenue`
- `IMDB Rating`
- `No of Votes`

---

## Project Workflow

1. **Data Preprocessing:**  
   - Cleaned missing values
   - Handled numerical transformations
   - Created a custom `Success` label based on rating and revenue

2. **Sentiment Analysis:**  
   - Performed sentiment scoring on `Overview` text using VADER
   - Analyzed genre-wise average sentiment

3. **Model Building:**  
   - Trained a **Random Forest Classifier** using only key features
   - Achieved prediction accuracy and performance reports

4. **Visualizations:**  
   - Genre vs Average Sentiment Bar Plot
   - IMDB Rating vs Gross Revenue Scatter Plot
   - Feature Importance Plot

---

## Results

- **Most important features:** `Gross Revenue`, `IMDB Rating`, and `No of Votes`
- **Sentiment Trends:**  
  Some genres like *Drama* and *Action* tend to have more positive overview sentiments.
- **Model Accuracy:**  
  > Accuracy score : 1.0
