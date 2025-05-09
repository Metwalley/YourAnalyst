# 📊 YourAnalyst – Final Project for the Digital Pioneers Scholarship 🇪🇬

Welcome to **YourAnalyst** – a streamlined, interactive data analytics toolkit developed by a passionate team of students from the **منحة رواد مصر الرقمية** (Digital Pioneers Scholarship) offered by the Egyptian Ministry of Communications.

This project began with a core objective: **Sales Forecasting and Optimization** using real-world data. But along the way, we realized we needed a better toolkit to manage messy data, generate insights, and train reliable models—so we built one!

---

## 🚀 Project Overview

### 🎯 Goal

* Forecast sales and optimize inventory using the **Rossmann Store Sales** dataset.
* Create a reusable web-based tool to automate the full data pipeline: **cleaning → visualizing → modeling**.

### 🧠 Built With Passion

We were tired of repeating the same steps—loading data, cleaning, encoding, modeling. So, we built **YourAnalyst** to simplify all of that with a few clicks.

---

## 🧩 Project Structure

```
your-analyst-project/
├── all_function.py       # Preprocessing utilities
├── visualize.py          # Charting & visual functions
├── yourAnalyst.py        # Main Streamlit web app
├── Dashboard.py          # Dash dashboard for forecasts
├── deploy.py             # Deploy trained models
├── data/                 # Raw & cleaned datasets
├── models/               # Trained model files (.pkl)
└── requirements.txt      # Project dependencies
```

---

## 🛠️ Features

### 🔍 Preprocessing Tab

* Clean data: handle missing values, detect outliers, scale/normalize
* Feature engineering: date splitting, column renaming, type conversion
* Feature selection: variance, correlation, importance
* Train/test split + live previews
* Export cleaned data in multiple formats

### 📊 Visualization Tab

* Choose from a wide variety of charts:

  * Line, Scatter, Bar, Box, Violin, Heatmap, Pairplot, Histogram
  * Word Cloud & Density Plots
* Customize axis labels, figure size, and titles

### 🤖 Modeling Tab

* Train or upload models:

  * **Regression**: Linear, RF, XGBoost, SVR, etc.
  * **Classification**: Logistic, RF, KNN, SVM, etc.
* View results:

  * Evaluation metrics (R², MAE, etc.)
  * True vs. Predicted table
* Download trained models for later use

### 📈 Dashboard.py – Forecasting & Optimization

* Forecast 90 days of sales using XGBoost Regressor (R² = 0.98)
* Calculate:

  * Safety stock = 20% of forecast
  * Reorder level = forecast + safety stock
* Interactive line chart for inventory planning

---

## 📦 Dataset

We used the [Rossmann Store Sales](https://www.kaggle.com/competitions/rossmann-store-sales/data) dataset:

* 1,100+ stores
* Includes sales, customers, holidays, promotions, competition data
* Engineered time features for forecasting

---

## 🔥 Highlights

* ✅ XGBoost Regressor with **98% accuracy**
* ✅ Streamlit app for end-to-end pipeline
* ✅ Modular design for reusability
* ✅ Inventory insights via safety stock & reorder logic

---

## 💡 Lessons Learned

* Repetition = opportunity for automation!
* Version conflicts taught us to **pin dependencies**
* Collaboration tools (shared notebooks, Git, Streamlit) kept our team aligned
* Performance tuning (Streamlit pagination, chart optimizations) mattered more than expected

---

## 🔮 Future Improvements

* 🔐 User authentication
* 🐳 Docker containerization for easy deployment
* 🔄 Real-time data ingestion or scheduling
* 🌐 Multilingual support for broader accessibility

---

## 👨‍💻 Meet the Team

Built with love by a group of students from the **Digital Pioneers Scholarship – منحة رواد مصر الرقمية** 💛
We’re proud to turn a single project into a practical tool for future analysts and teams like ours.

---

## 📎 How to Run

1. Clone the repo
   `git clone https://github.com/your-username/youranalyst.git`
   `cd youranalyst`

2. Install requirements
   `pip install -r requirements.txt`

3. Launch the app
   `streamlit run yourAnalyst.py`

4. (Optional) Launch the forecasting dashboard
   `python Dashboard.py`

---

## 🧾 License

MIT License. Feel free to use, modify, and improve this project!

---

## 🌟 Star Us!

If you found this project helpful or inspiring, please give it a ⭐️ on GitHub!
