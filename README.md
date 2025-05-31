
# 🧮 Income & Repayment Capability Estimator for Underserved Areas

A machine learning project to predict monthly income for India's underserved regions using alternative, regional-level data. Built for HackVortex 2025 - Open Innovation Round by Team GSV.

---

## 🚀 Problem Statement

Millions of informal workers in India are denied credit due to lack of formal income proofs. Traditional models rely on bank statements, tax filings, or salary slips — excluding a large portion of financially active individuals. We aim to simulate income visibility using region-based proxies.

---

## 💡 Our Solution

We developed a machine learning pipeline that uses pincode-level data like night light intensity, UPI transactions, mobile penetration, literacy rate, internet access, and more to predict monthly income. The model is deployed using Streamlit for real-time input and prediction.

---

## 🧠 Tech Stack

1. **Python** – Core programming language  
2. **Pandas, NumPy** – Data handling and transformation  
3. **Scikit-learn** – ML pipeline and training  
4. **RandomForestRegressor** – Income prediction model  
5. **TargetEncoder** – Encoding high-cardinality categorical data  
6. **StandardScaler** – Feature normalization  
7. **Joblib** – Saving/loading model  
8. **Streamlit** – Web-based UI for predictions  
9. **Git, GitHub** – Version control and collaboration

---

## 📁 Project Structure

```
├── data/
│   └── pincode_enriched.csv         # Dataset
├── models/
│   └── income_model.pkl             # Trained ML model
├── app/
│   └── streamlit_app.py             # Streamlit UI
├── src/
│   └── train_model.py               # Model training script
├── predict.py                       # CLI-based prediction
├── requirements.txt                 # Dependencies
└── README.md                        # Project overview
```

---

## 🧪 How to Run

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python src/train_model.py
```

### 3. Launch the Web App
```bash
streamlit run app/streamlit_app.py
```

---

## 📸 Sample UI

> _(Add a screenshot of your Streamlit app here with example inputs and predicted income)_

---

## 🎥 Demo Video

📺 Watch here: https://drive.google.com/drive/folders/19yOHn1k9qbwGQdObGMVkAx9qQShjd8y9?usp=sharing

---

## 🎯 Real-World Impact

- Enables fair credit access in low-data rural areas  
- Helps NBFCs and microfinance lenders reduce loan risk  
- Assists government bodies in targeting subsidies  
- Scalable and transparent system promoting financial inclusion

---

## 👨‍💻 Team GSV

- **Kaushal Raj** – Solo
---

## 📄 License

This project is created for HackVortex 2025 by Team GSV. All code is original and not reused from external sources.
