# Linear Regression with Gradient Descent on the Advertising Dataset

This project demonstrates how to implement a linear regression model using gradient descent on the "Advertising.csv" dataset from Kaggle. The dataset contains advertising budgets (TV, Radio, Newspaper) and corresponding sales figures. The goal is to predict sales based on these advertising budgets.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Data Loading and Exploration](#data-loading-and-exploration)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Implementation (Gradient Descent)](#model-implementation-gradient-descent)
5. [Model Evaluation and Analysis](#model-evaluation-and-analysis)
6. [Printed Outputs and Analysis](#printed-outputs-and-analysis)
7. [Conclusion](#conclusion)

## Project Overview

In this project, we:
- **Load the dataset** and perform extensive exploratory data analysis (EDA) using multiple visualizations.
- **Preprocess the data** by scaling features and adding an intercept term.
- **Implement a linear regression model** using gradient descent.
- **Evaluate the model** using performance metrics and diagnostic plots.
- **Analyze printed outputs** from the training process to assess convergence and overall model performance.

## Data Loading and Exploration

### What We Did
- **Loaded the Data:**  
  The dataset "Advertising.csv" was loaded into a Pandas DataFrame.
- **Checked for Missing Values:**  
  Verified that there are no missing values.
- **Displayed Summary Statistics:**  
  Provided basic statistics (mean, median, quartiles) to understand the data distribution.

### Visualizations and Their Interpretations

1. **Histograms:**  
   - **Purpose:** Visualize the distribution of each variable (TV, Radio, Newspaper, Sales).  
   - **Interpretation:** Reveals the skewness or symmetry of data, helping identify if any variable is heavily skewed.

2. **Boxplots:**  
   - **Purpose:** Identify outliers and observe the spread of the data.  
   - **Interpretation:** Outliers, if present, are visible as points outside the whiskers.

3. **Correlation Heatmap:**  
   - **Purpose:** Assess the pairwise relationships between variables.  
   - **Interpretation:** Shows which features are strongly or weakly correlated with Sales. For example, a high correlation between TV and Sales suggests that TV is an important predictor.

4. **Pairplot:**  
   - **Purpose:** Visualize pairwise relationships across all variables.  
   - **Interpretation:** Provides insights into potential linear relationships between features and Sales.

5. **Scatter and Joint Plots:**  
   - **Purpose:** Explore individual relationships between each advertising medium and Sales.  
   - **Interpretation:**  
     - **Scatter Plots:** Highlight the direct relationship between one feature and Sales.  
     - **Joint Plots:** Combine scatter plots with histograms/regression lines to illustrate both the trend and distribution.

## Data Preprocessing

### What We Did
- **Feature Selection:**  
  Selected `TV`, `Radio`, and `Newspaper` as features and `Sales` as the target variable.
- **Feature Scaling:**  
  Used `StandardScaler` to standardize the features, ensuring they have a mean of 0 and a standard deviation of 1. This is crucial for the efficiency of gradient descent.
- **Adding an Intercept:**  
  Added a column of ones to account for the bias term in the linear regression model.
- **Train-Test Split:**  
  Split the data into 80% training and 20% testing sets to validate the model on unseen data.

## Model Implementation (Gradient Descent)

### What We Did
- **Cost Function (`compute_cost`):**  
  Calculates the Mean Squared Error (MSE) between predictions and actual values. This function is used to evaluate the model's performance.
- **Gradient Descent (`gradient_descent`):**  
  Iteratively updates the model parameters (`theta`) to minimize the cost function.  
  - **Learning Rate:** Determines the step size in the parameter update.
  - **Iterations:** The number of updates performed.  
  The cost is printed every 100 iterations to monitor convergence.

### Visualizations
- **Cost Convergence Plot:**  
  - **Purpose:** Display the cost function value over iterations.
  - **Interpretation:** A steadily decreasing cost indicates that the gradient descent algorithm is converging to a minimum.

## Model Evaluation and Analysis

### What We Did
- **Predictions:**  
  Generated predictions for the test set using the optimized parameters.
- **Performance Metrics:**  
  - **Mean Squared Error (MSE):** Quantifies the average squared differences between the actual and predicted values.
  - **R² Score:** Measures the proportion of variance in the target variable explained by the features.

### Visualizations and Their Interpretations

1. **Actual vs. Predicted Sales Scatter Plot:**  
   - **Purpose:** Compare actual sales against predicted sales.
   - **Interpretation:** Points close to the diagonal line indicate good predictive accuracy.

2. **Residual Analysis:**  
   - **Residual Histogram with KDE:**  
     - **Purpose:** Examine the distribution of prediction errors.
     - **Interpretation:** A roughly normal distribution of residuals suggests that errors are random.
   - **Residual Scatter Plot:**  
     - **Purpose:** Plot residuals against predicted values.
     - **Interpretation:** A random scatter around zero indicates that the model errors are not patterned (homoscedastic).
   - **Q-Q Plot:**  
     - **Purpose:** Assess whether the residuals follow a normal distribution.
     - **Interpretation:** Points falling close to the reference line indicate normality.

## Printed Outputs and Analysis

Below are the printed outputs from the training and evaluation process:

```markdown
Iteration 0: Cost 110.1334  
Iteration 100: Cost 15.6080  
Iteration 200: Cost 3.2521  
Iteration 300: Cost 1.6140  
Iteration 400: Cost 1.3909  
Iteration 500: Cost 1.3589  
Iteration 600: Cost 1.3538  
Iteration 700: Cost 1.3529  
Iteration 800: Cost 1.3526  
Iteration 900: Cost 1.3526  

Optimized Theta Parameters: [14.04142097 3.83093398 2.79851601 0.06385596]

Model Evaluation on Test Data:  
Mean Squared Error: 3.1757  
R² Score: 0.8994


### Analysis of the Outputs

- **Cost Convergence:**  
  The cost starts at **110.1334** and decreases rapidly, leveling off at around **1.3526**. This indicates effective convergence of the gradient descent algorithm.

- **Optimized Theta Parameters:**  
  - **Intercept:** ~14.04  
  - **TV coefficient:** ~3.83  
  - **Radio coefficient:** ~2.80  
  - **Newspaper coefficient:** ~0.064  
  These values suggest that TV and Radio budgets have a strong positive influence on Sales, while Newspaper has a negligible effect.

- **Model Evaluation Metrics:**  
  - **Mean Squared Error (MSE):** 3.1757  
    A low MSE indicates that the average error between predicted and actual sales is relatively small.
  - **R² Score:** 0.8994  
    An R² score close to 0.90 implies that approximately 90% of the variance in Sales is explained by the model, demonstrating strong performance.

## Conclusion

In summary, this project provides a comprehensive workflow for building a linear regression model using gradient descent on the Advertising dataset. Key takeaways include:

- **Exploratory Data Analysis:**  
  Multiple visualizations (histograms, boxplots, heatmaps, pairplots) helped in understanding the data distribution, identifying outliers, and revealing correlations.

- **Data Preprocessing:**  
  Feature scaling and the addition of an intercept term were critical for efficient model training.

- **Model Training:**  
  Gradient descent effectively minimized the cost, as evidenced by the decreasing cost values and convergence plot.

- **Model Performance:**  
  The model performs well with an R² score of 0.8994 and low MSE. Diagnostic plots, such as the actual vs. predicted scatter plot and residual analysis, further confirm that the model captures the underlying data patterns accurately.

Overall, this approach demonstrates a robust and interpretable method for linear regression, with clear insights drawn from both data exploration and model evaluation.

