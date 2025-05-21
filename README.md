# 📊 YourAnalyst – A Smart & Interactive Data Assistant

🎓 Final Project – Digital Pioneers Scholarship 🇪🇬

Welcome to **YourAnalyst** — a full-cycle, intelligent data assistant built by a passionate team of students from the Egyptian Ministry of Communications' scholarship program: **منحة رواد مصر الرقمية (Digital Egypt Pioneers Scholarship)**.

The project originated from a practical use case —
🎯 **Sales Forecasting and Inventory Optimization** using real-world retail data — but quickly evolved into something more powerful:

A fully interactive tool that lets you understand, clean, visualize, model, and reason about your data…
⚡ All from a browser, without writing a single line of code.

---

## 🚀 Project Overview

We realized a core truth: as data analysts, we repeat the same steps in every project — importing data, handling missing values, encoding features, training models. So instead of doing that manually every time, we asked ourselves:

> "Why not build a smart assistant that handles it all?"

Thus, **YourAnalyst** was born — part toolkit, part chatbot, and fully interactive.

---

## 🧩 Project Structure

```
your-analyst-project/
├── yourAnalyst.py         # Main Streamlit app with all tabs
├── all_function.py        # Preprocessing utilities
├── visualize.py           # Charting & visual functions
├── Dashboard.py           # Dash-based dashboard for forecasting
├── deploy.py              # Streamlit app for deployed model usage
├── data/                  # Raw & cleaned datasets
├── models/                # Trained models (.pkl)
└── requirements.docx      # Project dependencies
```

---

## 🛠️ Key Features

### 🔍 Preprocessing Tab

A rich suite of one-click preprocessing options:

* View and explore datasets
* Detect and handle:

  * Missing values
  * Outliers
  * Low-variance and multicollinear features
* Encode features (Label, One-Hot, Ordinal)
* Normalize/scale data
* Change dtypes, rename or drop columns
* Feature generation from date fields
* Target selection & Train/test split
* Live preview after every step
* Export cleaned data

---

### 📊 Visualization Tab

Easily generate a wide variety of charts to explore your data:

* Line, Scatter, Bar, Box, Violin, Histogram
* Heatmaps, Pairplots, Density Plots, Word Clouds

Each visualization is interactive and customizable.

---

### 🤖 Modeling Tab

Train or test models with a single click:

* Supports both:

  * 🔵 Classification: Logistic Regression, Random Forest, KNN, SVM, Gradient Boosting, XGBoost
  * 🔴 Regression: Linear Regression, SVR, Random Forest, XGBoost, etc.
* Model evaluation:

  * True vs. Predicted table
  * Metrics like R², MAE, Accuracy, F1, etc.
* User input section:

  * Manually enter new data and get instant predictions from trained models
* Download trained models for future use or deployment

---

### 🧠 Chatbot Assistant Tab – Ask. Analyze. Automate.

One of our standout features. The chatbot is tightly integrated with the dataset you upload and uses both prebuilt analysis logic and a connected LLM to support your workflow:

* Understands your dataset and automatically identifies:

  * Data types
  * Target column
  * Missing values or imbalanced classes
* Provides reasoning-based suggestions like:

  * “What preprocessing steps should I apply?”
  * “Which model type suits this target?”
  * “What are the top features affecting the target?”
* Lets you chat freely with your dataset:

  * Ask technical, statistical, or exploratory questions
  * Get relevant, explainable responses from your assistant

💡 Whether you’re a student or analyst, this feature is like having a mentor sitting beside you — insightful, fast, and always context-aware.

---

### 📈 Dashboard.py – Forecasting & Optimization

A dedicated Dash-powered dashboard where we applied YourAnalyst in a real business scenario:

* Dataset: Rossmann Store Sales
* Trained an XGBoost Regressor (R² = 0.98)
* Forecast next 90 days of sales per store
* Visualize predicted sales vs. historical data
* Calculate:

  * Safety stock = 20% of forecast
  * Reorder level = forecast + safety stock
* Line chart for actionable inventory planning

---

### 🚀 deploy.py – Real-World Scenario: Manager Prediction App

Built a separate Streamlit app for real-life use:

* Imagine you’re a store manager.
* You enter today’s store settings (e.g., store type, promo, holiday, competition).
* The trained model instantly predicts expected sales.

✔️ Fast, intuitive, and powered by your actual model.

---

## 📦 Dataset

We used the [Rossmann Store Sales Dataset](https://www.kaggle.com/competitions/rossmann-store-sales/data):

* 1,100+ stores
* Historical sales, promotions, holidays, and competition data
* Engineered features from date and categorical values
* Cleaned and preprocessed using YourAnalyst

---

## 🔥 Highlights

* ✅ End-to-end Streamlit app for the full data workflow
* ✅ Chatbot Assistant trained to analyze and suggest improvements
* ✅ Integrated EDA, modeling, forecasting, and inventory optimization
* ✅ User input for live prediction
* ✅ XGBoost model with 98% test accuracy (R² = 0.98)

---

## 💡 Lessons Learned

* 💭 Repetition reveals opportunities for automation
* ⚙️ Pinning dependency versions avoids headaches
* 🤝 Consistent team communication matters (we used Git, shared notebooks)
* 🚀 Optimization isn’t just for models — it matters in UX too

---

## 🔮 Future Improvements

* 🔐 Add user authentication
* 🐳 Dockerize for easier deployment
* 🗓 Add scheduled forecast updates
* 🌍 Add multi-language UI support (Arabic/English)
* 📤 Connect to real-time data sources or APIs

---

## 👨‍💻 Meet the Team

Built with passion by a team of students from the
🎓 **Digital Pioneers Scholarship – منحة رواد مصر الرقمية** 🇪🇬

🧑‍🤝‍🧑 Team Members:

* Abdelrahman Metwally (Dashboard & Deployment)
* Ahmed Helmy (Modeling)
* Mohamed Ibrahim (Preprocessing)
* Youssef Atef (Chatbot)
* Abdelrahman Farag (Visualization)

Supervised by Eng. Mostafa Atlam
With gratitude to the Ministry of Communications and the DEBI program.

---

## 📎 How to Run

1. Clone the repo
   `git clone https://github.com/your-username/youranalyst.git`
   `cd youranalyst`

2. Install dependencies
   `pip install -r requirements.docx`

3. Launch the Streamlit app
   `streamlit run yourAnalyst.py`

4. (Optional) Launch the dashboard
   `python Dashboard.py`


---

## 🌟 Like It?

If you found this useful or inspiring, drop us a ⭐ on GitHub and share it with fellow analysts!
Let’s help make data analysis more human — and a whole lot smarter.
