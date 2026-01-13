# Restaurant Prediction System

This project predicts restaurants based on their **cuisine** using **machine learning** and **NLP** techniques. It uses a **Random Forest Classifier** on **TF-IDF features** of cuisine names to predict restaurant names.

---

## Features

- **Exploratory Data Analysis (EDA):** Analyze and clean the restaurant dataset using `pandas`, `matplotlib`, and `seaborn`.
- **Text Vectorization:** Convert cuisine text into numeric vectors using **TF-IDF**.
- **Label Encoding:** Encode restaurant names into numeric labels for model training.
- **Modeling:** Train a **Random Forest Classifier** with hyperparameter tuning using `GridSearchCV`.
- **Evaluation:** Compute **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)** using a custom function.
- **Persistence:** Save trained models and encoders with `joblib`:
  - `mlmodel.pkl` – Trained Random Forest model
  - `tfidf.pkl` – TF-IDF vectorizer
  - `le.pkl` – Label encoder

---

## File Structure

| File | Description |
|------|-------------|
| `Dataset.csv` | Restaurant dataset containing restaurant names and cuisines |
| `mlmodel.pkl` | Saved trained Random Forest model |
| `tfidf.pkl` | Saved TF-IDF vectorizer for cuisines |
| `le.pkl` | Saved label encoder for restaurant names |
| `model.ipynb` or `app.py` | Python code for preprocessing, training, and predictions |
| `requirements.txt` | Required Python packages |

---

### 🔗 Large File Download

The trained model file (`mlmodel.pkl`, ~600MB) is hosted externally due to GitHub size limits.

Download here: https://drive.google.com/file/d/1ev5zSbf-bPNzFVVa5zUVrP1l0ZP7eHut/view?usp=sharing

Place the file in the project root directory before running the code.

---

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- joblib
- streamlit

All dependencies are listed in requirements.txt.

---

## Author
**Anamika Singh**
