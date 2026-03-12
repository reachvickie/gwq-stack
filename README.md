#  GWQ-Stack: Neuro-Symbolic Ensemble Framework for Global Water Quality Prediction

 Large-scale Environmental AI | Ensemble Learning | Explainable AI | Neuro-Symbolic ML

---

##  Overview

GWQ-Stack is a scalable machine learning framework designed to predict the **Water Quality Index (WQI)** using large-scale environmental datasets. The project integrates ensemble learning with neural meta-learning to capture complex nonlinear relationships between hydrological parameters.

Traditional water quality assessment relies on laboratory testing and manual sampling, which is resource-intensive. GWQ-Stack provides a data-driven alternative for high-accuracy and scalable environmental monitoring.

---

##  Architecture

GWQ-Stack implements a **two-level stacked generalization architecture**.

### 🔹 Level 0 – Base Learners
- XGBoost
- CatBoost
- LightGBM
- Random Forest

These models capture diverse error patterns using boosting and bagging techniques.

### 🔹 Level 1 – Meta Learner
- Multilayer Perceptron (MLP)

The meta-learner captures nonlinear relationships between base learner predictions and improves final output accuracy.

---

##  Dataset

- Global Water Quality Dataset (1940–2023)
- Total Samples: **2.82 Million**

### Features
- Ammonia (NH₃)
- Biochemical Oxygen Demand (BOD)
- Dissolved Oxygen (DO)
- Orthophosphate
- Nitrate
- Nitrogen
- pH
- Temperature

### Target Variable
- Canadian Council of Ministers of the Environment Water Quality Index (CCME-WQI)

---

##  Methodology

### Data Processing
- Median imputation for missing values
- Robust feature scaling
- Exploratory Data Analysis (EDA)
- Correlation analysis

### Model Training
- 5-Fold Cross Validation
- Out-of-Fold (OOF) prediction generation
- Stacked ensemble learning

### Explainability
- SHAP-based feature attribution
- Global and local interpretability analysis

---

##  Performance Results

| Metric | Score |
|---------|------------|
| R² Score | 0.99992 |
| RMSE | 0.15856 |
| MAE | 0.05456 |

 Achieved approximately **85% reduction in prediction error** compared to existing benchmark models.

---

##  Key Insights

- Orthophosphate and Ammonia are strong indicators of water quality degradation.
- Hybrid stacking improves robustness by combining bias reduction and variance reduction.
- Neural meta-learning effectively captures residual prediction errors.

---

##  Tech Stack

- Python
- Scikit-learn
- XGBoost
- CatBoost
- LightGBM
- TensorFlow / PyTorch
- SHAP
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

---

##  Applications

- Environmental monitoring
- Smart water resource management
- Pollution prediction systems
- Sustainability and policy planning
- IoT-based water quality monitoring

---


---

## 📜 License

This project is licensed under the MIT License.

