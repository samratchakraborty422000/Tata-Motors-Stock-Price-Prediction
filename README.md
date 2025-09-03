# Tata-Motors-Stock-Price-Prediction 📈
![TataLogo](https://github.com/user-attachments/assets/baa22c9a-ec45-4403-a5fc-a80545118e0c)


This project applies machine learning regression models to predict **Tata Motors Limited (TATAMOTORS)** stock closing prices using historical stock data from **NSE (1995–2025)**. It compares several models and evaluates their performance based on common regression metrics.

---

## 📊 Dataset

- **Source**: NSE – Tata Motors Stock Data (1995-01-02 to 2025-08-19)
- **Shape**: 7,804 rows × 14 columns
- **Features**:
  - `Date`: Trading day
  - `Open`: Opening price
  - `High`: Daily high
  - `Low`: Daily low
  - `Close`: Closing price (target)
  - `PrevClose`: Previous day's close
  - `Volume`: Total shares traded
  - `Turnover`: Total turnover (₹)
  - `VWAP`: Volume weighted average price
  - `Trades`: Number of trades
  - `Daily_Return_%`: Daily % change
  - `MA_20`: 20-day moving average
  - `MA_50`: 50-day moving average

- **Target**: `Close` (daily closing price)

---

## 🔍 Exploratory Data Analysis (EDA)

- Converted `Date` column to datetime format.
- Checked for null values and handled missing values using **median imputation**.
- Generated summary statistics and correlation matrix.
- Visualized:
  - Distribution of price columns (`Open`, `Close`, `High`, `Low`, `PrevClose`)
  - Time series trend of prices over time
  - Correlation heatmap of numeric variables

📌 **Key Insight**: Most price-related columns are **highly correlated**, so only a subset was used to prevent multicollinearity and overfitting.

---

## 🧪 Model Training & Evaluation

**Train-Test Split**: 80% training / 20% testing  
**Target Variable**: `Close` (stock closing price)  
**Features Used**:

```python
features = ['Open', 'High', 'Low', 'VWAP', 'Trades', 'Turnover', 'MA_50']
```

## ✅ Models Compared

| Model             | R² Score   | MSE      |
|------------------|------------|----------|
| Linear Regression | **0.999789** | **14.1051** |
| Ridge Regression  | 0.999789   | 14.1051  |
| Lasso Regression  | 0.999393   | 40.6707  |
| Random Forest     | 0.999707   | 19.6201  |
| XGBoost           | 0.999592   | 27.3615  |
| Decision Tree     | 0.999598   | 26.9397  |
| LightGBM          | 0.999631   | 24.7359  |
| CatBoost          | 0.999426   | 38.4747  |

📝 **Note**: `LinearRegression` performed the best with an R² score of **0.999789** and lowest MSE.

---

## 💾 Model Saving

Saved the best model using `joblib`:
```bash
import joblib
joblib.dump(best_model, "LinearRegression_best_model.pkl")
```

## 💻 Run the Project Locally
Install the required libraries:
```bash
pip install -r requirements.txt
```

### Example of requirements.txt:
```bash
numpy
streamlit
joblib
```
## 📷 Visualizations

### Time Series Plot

<img width="1176" height="595" alt="image" src="https://github.com/user-attachments/assets/dc5d506c-3cd2-450d-990b-97dd70f9378d" />

### Correlation Heatmap

<img width="998" height="930" alt="image" src="https://github.com/user-attachments/assets/7fa1034f-2518-48a2-9561-354a43ea238f" />

##  📷 Screenshots

<img width="1007" height="1007" alt="Screenshot 2025-09-03 123553" src="https://github.com/user-attachments/assets/3c6c1fc4-5345-44a7-a952-f46f59bde4fc" />
<img width="900" height="900" alt="Screenshot 2025-09-03 123617" src="https://github.com/user-attachments/assets/a5ecfc2b-ee2a-4299-aab4-0129d77109ec" />


