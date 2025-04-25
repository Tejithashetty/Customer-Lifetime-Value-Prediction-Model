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
