# 🚗 Car Price Prediction with Machine Learning  

![Python](https://img.shields.io/badge/Python-3.8-blue?style=for-the-badge&logo=python)  
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-orange?style=for-the-badge&logo=scikitlearn)  
![Jupyter Notebook](https://img.shields.io/badge/Notebook-Jupyter-informational?style=for-the-badge&logo=jupyter)  

## 📌 Project Overview  
This project aims to **predict the selling price of used cars** based on various features such as the car’s **age, kilometers driven, fuel type, transmission, and number of previous owners**. By using **Machine Learning models**, we can help car buyers and sellers make informed pricing decisions.  

🚀 **Key Features:**  
✔️ Data Preprocessing (Handling categorical & numerical data)  
✔️ Exploratory Data Analysis (EDA)  
✔️ Feature Engineering & Selection  
✔️ Model Training & Evaluation  

---

## 📂 Dataset Overview  
The dataset contains **301 entries** with the following **9 features**:  

| Feature | Description |
|---------|------------|
| `Car_Name` | Name of the car (string) |
| `Year` | Manufacturing year (integer) |
| `Selling_Price` | Price at which the car is being sold (Target variable) |
| `Present_Price` | Price of the car when it was new |
| `Driven_kms` | Kilometers driven |
| `Fuel_Type` | Type of fuel (Petrol, Diesel, CNG) |
| `Selling_type` | Seller type (Dealer or Individual) |
| `Transmission` | Manual or Automatic |
| `Owner` | Number of previous owners |

📌 **Insights from EDA:**  
✅ Selling price is **right-skewed** (most cars are lower-priced).  
✅ **Present Price** has the highest correlation with **Selling Price**.  
✅ **Fuel Type:** Petrol cars dominate, followed by Diesel.  
✅ **Transmission Type:** Manual cars are more common than automatic.  

---

## 🔧 Data Preprocessing  
✔️ One-hot encoding for categorical features.  
✔️ Feature scaling for numerical values.  
✔️ Dropped irrelevant features like `Car_Name`.  
✔️ Splitting dataset into **80% Training** and **20% Testing**.  

```python
# Splitting data into train and test sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

## 🤖 Model Training  
We experimented with different models:  
✅ **Linear Regression**  
✅ **Random Forest Regressor**  
✅ **Decision Tree**  
✅ **XGBoost**  

📊 **Performance Metrics Used:**  
- **R² Score** (How well the model fits the data)  
- **Mean Absolute Error (MAE)**  

---

## 📈 Results & Findings  
| Model | R² Score (Test) | MAE (Test) |
|--------|-------------|-------------|
| Linear Regression | 0.86 | 1.2 Lakhs |
| Random Forest | 0.92 | 0.9 Lakhs |
| Decision Tree | 0.88 | 1.1 Lakhs |
| XGBoost | 0.94 | 0.8 Lakhs |

📌 **Best Model:** **XGBoost** with **94% accuracy** 🎯  

---

## 🚀 How to Run the Project  
### 1️⃣ Install Dependencies  
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

### 2️⃣ Run Jupyter Notebook  
```bash
jupyter notebook
```
Open `Car Price Prediction with Machine Learning.ipynb` and run all cells.

---

## 📌 Future Improvements  
🔹 Improve feature selection & engineering.  
🔹 Try Deep Learning models.  
🔹 Build a web app using **Flask / Streamlit** for real-time predictions.  

---

## 💡 Conclusion  
This project successfully predicts used car prices with **high accuracy** using machine learning techniques. The **XGBoost model** provided the best results with a **94% R² Score**.  

---

## 🤝 Connect With Me  
💻 [GitHub](https://github.com/yuvrajsaraogi) | 🌐 [LinkedIn](https://linkedin.com/in/yuvraj-saraogi) | ✉️ [Email](yuvisaraogi@gmail.com)  
