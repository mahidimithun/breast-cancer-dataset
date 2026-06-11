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
* **Analysis:** This heatmap visualizes the Pearson correlation coefficients across the feature matrix. It highlights widespread **multicollinearity** (such as near-perfect correlations of `1.00` among tumor radius, perimeter, and area). Highly correlated features are mathematically redundant. Isolating them helps streamline model input, minimizes overfitting, and enables the SVM algorithm to locate optimal decision boundaries efficiently.

### 2. Feature Distribution and Density Analysis
<img width="996" height="372" alt="graphs" src="https://github.com/user-attachments/assets/08cf0dc5-4a83-4cce-920c-11757465b9c0" />
* **Analysis:** A matrix of individual density plots illustrating the separation curves between Benign (0) and Malignant (1) classes across key clinical measurements. Distinct, non-overlapping peaks indicate high feature importance, signaling exactly which data attributes give the model the clearest instructions for drawing structural classification splits.

### 3. 3D Spatial Feature Decision Space
<img width="595" height="621" alt="3d graph" src="https://github.com/user-attachments/assets/8fc9cc5c-2a0c-47b3-842c-50b4711406ea" />
* **Analysis:** A 3-dimensional scatter plot displaying data distribution across three major geometric components. By mapping features into a three-dimensional geometric space, this graph simulates how an SVM identifies multi-dimensional hyperplanes. It visually demonstrates how a non-linear kernel warps data coordinates to cleanly separate benign and malignant clusters.

---

## 🔬 Model Performance Evaluation
<img width="400" height="482" alt="confusion_matrix" src="https://github.com/user-attachments/assets/d0474e74-9252-4804-9877-c039edad8a64" />
### 1. Confusion Matrix Breakdown
Based on the testing split evaluation, the model's predictions are categorized into the following matrix:

* **True Negatives (TN) = 71:** The model correctly identified 71 benign tumors as Benign ($0$).
* **True Positives (TP) = 39:** The model correctly identified 39 malignant tumors as Malignant ($1$).
* **False Positives (FP) = 1:** The model made 1 "false alarm," predicting a tumor was malignant when it was actually benign.
* **False Negatives (FN) = 3:** **The most critical metric in diagnostics.** The model missed 3 malignant cases, predicting them as benign. In medical AI, the goal is to drive this number as close to 0 as possible to avoid leaving an active condition undetected.

---

### 2. Classification Report Metrics

| Metric | Class 0 (Benign) | Class 1 (Malignant) | Overall Model Summary |
| :--- | :--- | :--- | :--- |
| **Precision** | **0.96** | **0.97** | Out of all predicted Malignant cases, **97%** were correct. |
| **Recall** | **0.99** | **0.93** | The model successfully caught **93%** of the actual Malignant cases. |
| **F1-Score** | **0.97** | **0.95** | Represents a balanced harmonic mean between precision and recall. |
| **Accuracy** | - | - | **🏆 Overall Accuracy: 96.49%** (110 / 114 correct predictions) |

#### Metric Significance:
* **Recall (Sensitivity) for Class 1 (0.93):** This indicates high clinical reliability. The model minimized false negatives effectively, successfully isolating 93% of the malignant instances.
* **Macro vs. Weighted Averages (0.96 / 0.96):** The tight alignment between these averages confirms that the model is performing stably across both classes without being silently biased by the majority class distribution.

---

## 💻 Tech Stack Used
* **Language:** Python
* **Data Libraries:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (SVC, StandardScaler, train_test_split)
* **Visualization:** Matplotlib, Seaborn
