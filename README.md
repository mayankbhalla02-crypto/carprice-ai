# 🚗 Car Price AI (Fresh Route AI) — Vehicle Valuation Engine

An enterprise-grade, full-stack machine learning web application that predicts pre-owned car market values in India using a trained Random Forest Regressor on the Cardekho dataset.

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-000000?style=for-the-badge&logo=flask&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit_Learn-1.3-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![TailwindCSS](https://img.shields.io/badge/Tailwind_CSS-2.2-06B6D4?style=for-the-badge&logo=tailwindcss&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-3-003B57?style=for-the-badge&logo=sqlite&logoColor=white)

---

## 👥 Project Team & Credentials

* **Mayank Bhalla** — `26BAI70007`
* **Naman Sharma** — `26BAI70050`
* **Gunit** — `26BAI70032`

**Institution:** Department of Computer Science & Engineering, Chandigarh University  
**Degree:** B.Tech (Hons.) Computer Science & Engineering (Specialization in AI & ML)  
**Date:** October 2026

---

## 📌 Features

- **Real-Time Machine Learning Valuation:** Instant price predictions computed via a serialized Random Forest Regressor (`car_model_rf.pkl`).
- **Valuation Bounds:** Returns Fair Market Value, Low Bound, and High Bound estimates based on tree ensemble standard deviation.
- **Market Benchmarking:** Displays automated comparisons against historical Cardekho listings (Average, Certified, Trade-in).
- **User Authentication:** Built-in signup, login, and session persistence backed by an SQLite database (`carusers.db`) with SHA-256 password hashing.
- **Modern UI Dashboard:** Fully responsive dark-mode dashboard styled with Tailwind CSS and powered by asynchronous JavaScript `fetch()` calls.
- **One-Click Local Execution:** Executable macOS shell script (`CarPriceAI.command`) handling dynamic directory navigation to eliminate Flask path errors.

---

## 🛠️ Tech Stack & System Architecture

| Component | Technology / Library |
| :--- | :--- |
| **Backend Framework** | Python 3.9+, Flask |
| **Machine Learning** | Scikit-Learn (RandomForestRegressor), Pandas, NumPy |
| **Model Serialization** | Joblib / Pickle |
| **Database** | SQLite3 |
| **Security & Auth** | Werkzeug (`generate_password_hash`, `check_password_hash`) |
| **Frontend Layout** | HTML5, Tailwind CSS (via CDN), Vanilla JavaScript (ES6) |
| **Automation Script** | macOS Zsh / Bash Executable (`CarPriceAI.command`) |

---

## 📁 Repository Structure

```text
CarPriceAI/
├── CarPriceAI.command          # Automated macOS Terminal Execution Executable
├── app.py                      # Primary Flask Web Application & REST Controller
├── ml_model.py                 # Model Training, Engineering & Serialization Pipeline
├── car_model_rf.pkl            # Trained Random Forest Regressor Binary Asset
├── Cardekho.csv                # Historical Secondary Market Dataset
├── carusers.db                 # SQLite Relational Database (User Auth)
├── requirements.txt            # Python Dependencies Manifest
├── README.md                   # Project Documentation
└── templates/                  # Jinja2 HTML Templates Directory
    ├── index.html              # Main Valuation Dashboard & API Interface
    ├── login.html              # User Authentication Login Screen
    ├── signup.html             # User Registration Screen
    └── dashboard.html          # Secondary Analytics View
```

---

## 🚀 Quick Start Guide

### Prerequisites
Make sure you have **Python 3.9+** installed on your system.

### Option 1: One-Click Execution (macOS)
1. Open terminal and navigate to the project directory:
   ```bash
   cd ~/Downloads/CarPriceAI
   ```
2. Grant execution permission to the command script:
   ```bash
   chmod +x CarPriceAI.command
   ```
3. Run the script:
   ```bash
   ./CarPriceAI.command
   ```
4. Open your browser and go to **`http://localhost:5000`**.

---

### Option 2: Manual Terminal Setup

1. **Clone or Download Repository:**
   ```bash
   git clone https://github.com/your-username/CarPriceAI.git
   cd CarPriceAI
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **(Optional) Train/Retrain the ML Model:**
   ```bash
   python3 ml_model.py
   ```

4. **Launch the Flask Server:**
   ```bash
   python3 app.py
   ```

5. **Access Application:**
   Open browser at `http://127.0.0.1:5000` or `http://localhost:5000`.

---

## 📊 Model Performance

| Metric | Score / Value |
| :--- | :--- |
| **Algorithm** | Random Forest Regressor (`n_estimators=100`, `max_depth=20`) |
| **Determination ($R^2$)** | **0.918** |
| **Mean Absolute Error (MAE)** | **₹0.48 Lakhs** |
| **Root Mean Squared Error (RMSE)**| **₹0.94 Lakhs** |
| **Mean Absolute Percentage Error** | **8.3%** |

---

## 📄 License & Attribution

This project is created for academic research and submission at **Chandigarh University**. Dataset sourced from [Cardekho Kaggle Repository](https://www.kaggle.com/cardekho).
