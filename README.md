# Heart-Disease-Indicator
Machine learning project focused on heart disease prediction using Python, Pandas, Scikit-learn, and TensorFlow/Keras. Explores data preprocessing, model training (NN, RF, DT, SVC, KNN, GBC), evaluation, and hyperparameter tuning.
## Description

This project aims to predict the likelihood of a patient having heart disease based on various medical attributes. It explores different machine learning classification models to determine the most effective approach for this dataset. This was developed as a group project for a university machine learning course.

## Dataset

The project utilizes the "Heart Disease UCI" dataset (or a similar variant). You can typically find this dataset on platforms like Kaggle or the UCI Machine Learning Repository.

*(**Note:** You should either add the `heart.csv` file to your repository or provide a direct link here to where someone can download it.)*

The dataset contains patient attributes such as:
*   Age
*   Sex
*   ChestPainType
*   RestingBP (Resting Blood Pressure)
*   Cholesterol
*   FastingBS (Fasting Blood Sugar)
*   RestingECG (Resting Electrocardiogram results)
*   MaxHR (Maximum Heart Rate achieved)
*   ExerciseAngina (Exercise-induced angina)
*   Oldpeak (ST depression induced by exercise relative to rest)
*   ST_Slope (Slope of the peak exercise ST segment)
*   HeartDisease (Target variable: 0 = Normal, 1 = Heart Disease)

## Project Workflow

1.  **Data Loading & Exploration:** The dataset (`heart.csv`) is loaded using Pandas. Initial exploration includes checking shape, data types (`.info()`), and viewing sample rows (`.head()`).
2.  **Data Preprocessing:**
    *   Checked for missing values using heatmaps and `.isnull().sum()`.
    *   Encoded categorical features (Sex, ChestPainType, FastingBS, RestingECG, ExerciseAngina, ST_Slope) into numerical representations using Scikit-learn's `LabelEncoder`.
    *   Visualized feature correlations using a heatmap (Seaborn).
    *   Analyzed relationships between key features and the target variable using pairplots and countplots.
3.  **Data Splitting & Scaling:**
    *   The data was split into training (70%) and testing (30%) sets using `train_test_split`.
    *   Features were scaled using Scikit-learn's `StandardScaler` to normalize the data distribution.
4.  **Model Training & Evaluation:**
    *   Several classification models were trained and evaluated:
        *   **Neural Network (NN):** Built using TensorFlow/Keras `Sequential` API with Dense layers and Dropout for regularization. Compiled with Adam optimizer and binary crossentropy loss.
        *   **Random Forest Classifier:** Implemented using Scikit-learn.
        *   **Decision Tree Classifier:** Implemented using Scikit-learn.
        *   **Support Vector Classification (SVC):** Implemented using Scikit-learn.
        *   **K Neighbors Classifier (KNN):** Implemented using Scikit-learn.
        *   **Gradient Boosting Classifier:** Implemented using Scikit-learn.
    *   Models were evaluated using accuracy score, classification reports (precision, recall, F1-score), and confusion matrices.
5.  **Cross-Validation:** Stratified K-Fold cross-validation (3 folds) was performed on multiple models (DT, RF, Logistic Regression, KNN, SVC, GBC) to assess model robustness and generalization.
6.  **Hyperparameter Tuning:** GridSearchCV was used to find optimal hyperparameters (batch size, epochs) for the Neural Network model, further improving its performance.
7.  **Prediction on New Data:** Demonstrated how to preprocess and predict the heart disease likelihood for a new, unseen data point using the trained Neural Network.

## Technologies Used

*   **Language:** Python 3
*   **Libraries:**
    *   Pandas (Data manipulation and analysis)
    *   Scikit-learn (Machine learning: splitting, preprocessing, models, evaluation, cross-validation, grid search)
    *   TensorFlow / Keras (Neural Network implementation)
    *   Matplotlib (Data visualization)
    *   Seaborn (Enhanced data visualization)
*   **Environment:** Jupyter Notebook (likely run in Google Colab based on file paths)

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repository-name.git
    cd your-repository-name
    ```
2.  **Ensure Python is installed.** (Python 3.7+ recommended)
3.  **Install required libraries:**
    ```bash
    pip install pandas scikit-learn tensorflow matplotlib seaborn jupyter
    ```
    *(Optional but recommended: Create a `requirements.txt` file with library versions and instruct users to run `pip install -r requirements.txt`)*
4.  **Dataset:** Download the `heart.csv` dataset and place it in the project directory, or update the path in the notebook if using a different location. *(Link to dataset if available)*
5.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook ML_Group_Project_G78_Heart_Disease_Indicator.ipynb
    ```
    or open it using JupyterLab or another compatible IDE like VS Code.

## Usage

Open the Jupyter Notebook and run the cells sequentially to see the data analysis, preprocessing steps, model training, evaluation, and prediction process.

## Results

*   Multiple classification models were compared based on accuracy, precision, recall, and F1-score.
*   Cross-validation provided insights into model generalization.
*   The Neural Network model, after hyperparameter tuning using GridSearchCV, achieved the highest accuracy on the test set (approximately 91.3%).
*   The Random Forest model also performed well (approximately 89.1%).

*(Optional: Insert key results tables or confusion matrix images here)*

## Visualizations

The notebook includes visualizations such as:
*   Null value heatmap
*   Target variable distribution (countplot)
*   Feature correlation heatmap
*   Pairplots showing relationships between selected features and the target variable
*   Confusion matrices for evaluated models
