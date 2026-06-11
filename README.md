# Breast Cancer Diagnostic Classification using Support Vector Machines (SVM)

## 📊 Dataset & Source
This project utilizes the **Breast Cancer Wisconsin (Diagnostic) Dataset** to classify breast masses as either benign or malignant based on geometric features computed from a digitized image of a fine needle aspirate (FNA).

🔗 **Dataset Link:** [Kaggle - Breast Cancer Dataset](https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset)

---

## 🛠️ Code Workflow & Implementation Steps

The Python pipeline is executed sequentially to transform raw medical metrics into reliable predictive outputs:

1. **Exploratory Data Analysis (EDA):** Evaluated data shapes, distributions, and collinear relationships across 30 clinical features using `Seaborn` and `Matplotlib`.
2. **Target Variable Encoding:** Cleaned the text-based target column (`diagnosis`) by mapping categorical labels `'B'` (Benign) to `0` and `'M'` (Malignant) to `1` to align with standard clinical conventions.
3. **Stratified Train-Test Split:** Divided the dataset into training ($80\%$) and testing ($20\%$) subsets. Employed `stratify=y` to ensure that class proportions remained perfectly identical across both splits.
4. **Feature Standardization:** Initialized a `StandardScaler` to fit strictly on the training data and transform both sets. This eliminates scale discrepancies among variables (like area vs. smoothness) that would otherwise distort the SVM's hyperplane.
5. **Model Training:** Configured and trained a **Support Vector Machine (SVM)** classifier using a Radial Basis Function (RBF) kernel to manage non-linear decision spaces.
6. **Performance Assessment:** Extracted predictions using `X_test_scaled` and compiled a full metric suite including accuracy scoring, a detailed classification report, and an interactive confusion matrix display.

---

## 📈 Visualizations & Graph Analysis

### 1. Feature Correlation Heatmap
<img width="377" height="742" alt="breast_cancer_correlation_heatmap" src="https://github.com/user-attachments/assets/716b2fd2-b1d5-407e-9137-36fabbe1746a" />

**Analysis:** This heatmap visualizes the Pearson correlation coefficients across the feature matrix. It highlights widespread **multicollinearity** (such as near-perfect correlations of `1.00` among tumor radius, perimeter, and area). Highly correlated features are mathematically redundant. Isolating them helps streamline model input, minimizes overfitting, and enables the SVM algorithm to locate optimal decision boundaries efficiently.

### 2. Feature Distribution and Density Analysis
<img width="996" height="372" alt="graphs" src="https://github.com/user-attachments/assets/08cf0dc5-4a83-4cce-920c-11757465b9c0" />

**Analysis:** A matrix of individual density plots illustrating the separation curves between Benign (0) and Malignant (1) classes across key clinical measurements. Distinct, non-overlapping peaks indicate high feature importance, signaling exactly which data attributes give the model the clearest instructions for drawing structural classification splits.

### 3. 3D Spatial Feature Decision Space
<img width="595" height="621" alt="3d graph" src="https://github.com/user-attachments/assets/8fc9cc5c-2a0c-47b3-842c-50b4711406ea" />

**Analysis:** A 3-dimensional scatter plot displaying data distribution across three major geometric components. By mapping features into a three-dimensional geometric space, this graph simulates how an SVM identifies multi-dimensional hyperplanes. It visually demonstrates how a non-linear kernel warps data coordinates to cleanly separate benign and malignant clusters.

---

## 🔬 Model Performance Evaluation

<img width="400" height="482" alt="confusion_matrix" src="https://github.com/user-attachments/assets/d0474e74-9252-4804-9877-c039edad8a64" />

## 🧩 Confusion Matrix Analysis

A Confusion Matrix provides a detailed look at where the model made correct predictions and where it faltered by mapping predicted categories directly against true clinical diagnoses.

Based on the testing split evaluation ($114$ samples), our SVM model produced the following distribution:

### 📈 Visual Breakdown of the Matrix

* **True Negatives (TN) = 72:** The model predicted **Benign** ($0$) for 72 samples, and they were truly benign. This represents a perfect evaluation for the negative class (**100% Sensitivity/Recall** for Benign).
* **True Positives (TP) = 39:** The model predicted **Malignant** ($1$) for 39 samples, and they were truly malignant.
* **False Positives (FP) = 0:** The model made **zero** false alarms. It never mistakenly flagged a healthy/benign patient sample as malignant (**100% Precision** for Malignant).
* **False Negatives (FN) = 3:** The model missed 3 actual malignant cases, mistakenly predicting them as benign. 

---

### 🩺 Diagnostic & Clinical Significance

In medical machine learning, analyzing the quadrants of a confusion matrix is far more vital than checking a single macro accuracy score:

1. **The Cost of False Negatives (Type II Error):** The 3 missed cases represent the most critical risk in medical diagnostics. A false negative means a patient with a malignant condition leaves with an incorrect "clean bill of health." Lowering this number to zero is the primary goal of future parameter tuning (e.g., adjusting class weights or choosing a higher-recall threshold).
2. **The Power of Zero False Positives (Type I Error):** Having zero false positives means that every single time this model raises an alarm for a malignant case, it is correct. In a healthcare framework, this minimizes unnecessary patient anxiety and avoids wasting clinical resources on secondary diagnostic confirmation procedures.

---

## 🔬 Final SVM Model Performance Analysis

The Support Vector Machine (SVM) model achieved an exceptional overall classification accuracy of **97.37%** (rendered as 0.97 in the rounded report summary) on the test dataset, correctly identifying 111 out of 114 unseen patient samples.

### 📑 Class-by-Class Metric Breakdown

#### 🟢 Class 0: Benign Tumors (72 Samples)
* **Precision (0.96):** Out of all cases the model predicted as Benign, 96% of them were truly benign. This indicates a minor false-alarm rate where a few malignant cases were misclassified as benign.
* **Recall (1.00):** **Perfect Sensitivity.** The model successfully caught 100% of the actual benign tumors in the test set. Zero benign cases were missed.
* **F1-Score (0.98):** Reflects a near-perfect balance between precision and recall for this category.

#### 🔴 Class 1: Malignant Tumors (42 Samples)
* **Precision (1.00):** **Perfect Positive Predictive Value.** When the model flags a tumor as Malignant, it is correct 100% of the time. There are zero False Positives (healthy patients given false cancer alarms).
* **Recall (0.93):** Out of 42 patients who actually had malignant tumors, the model correctly identified 39 of them (93%). The remaining 7% represent 3 False Negatives (missed cancer cases). In medical analytics, this is the most critical metric to monitor and continuously optimize.
* **F1-Score (0.96):** Confirms robust diagnostic strength for the positive cancer class.

---

### 🧮 Global Averages & Structural Integrity

* **Macro Average (0.98 Precision / 0.96 Recall):** This calculates the unweighted mean of the metrics across both classes. Its high value proves that the model maintains strong, independent predictive power for both benign and malignant cases rather than performing well on only one.
* **Weighted Average (0.97):** This factors in the underlying class volume (Support: 72 vs 42). Because the Macro and Weighted metrics track so closely together, it proves our stratified train-test split kept the data perfectly balanced, preventing the majority class from biasing the model's decision boundary.
---

## 💻 Tech Stack Used
**Language:** Python
**Data Libraries:** Pandas, NumPy
**Machine Learning:** Scikit-Learn (SVC, StandardScaler, train_test_split)
**Visualization:** Matplotlib, Seaborn
