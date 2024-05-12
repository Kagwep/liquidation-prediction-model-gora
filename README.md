
# Gora ML Liquidation Prediction Model

## Project Overview
This project develops a machine learning model to build models which predict the likelihood and the ratio of individual liquidations, in comparison to the amount borrowed [total_liquidation_to_total_borrow]

## Dataset
The dataset includes transaction histories from several DeFi lending platforms such as Aave, Compound, and MakerDAO, among others. It features various financial transactions including borrows, repays, and deposits, detailed across multiple blockchain networks.

## Data Exploration
Data exploration is a crucial step in any data science project as it allows us to understand the underlying patterns and relationships within the data. Here's how we explored the dataset:

- **Descriptive Statistics**: We began by generating descriptive statistics using `df.describe()`. This provided a summary of the central tendency, dispersion, and shape of the dataset's distribution, including means, medians, quartile ranges, and standard deviations. This step is essential to get a preliminary understanding of the data scale and variability.



- **Visualization of Log-Transformed Data**:
    ![log](https://res.cloudinary.com/dydj8hnhz/image/upload/v1715492825/trdukzldsjwbnmsk3lu1.png)
    We plotted the log-transformed distribution of `total_borrow` to assess its skewness and overall spread. The histogram, enhanced with a Kernel Density Estimate (KDE), helped us estimate the probability density function of the variable, providing insights into the data's distribution and confirming the effectiveness of the log transformation in normalizing the data.

- **Correlation Analysis**:
    
    ![cor](https://res.cloudinary.com/dydj8hnhz/image/upload/v1715493399/uxldyspmnowqpzualwgd.png)
    We utilized Spearman correlation instead of Pearson as it is more adept at handling non-linear relationships and less sensitive to outliers. This analysis provided valuable insights into how features related to each other, particularly in terms of borrow, repay, and liquidation activities.

These exploratory steps enabled us to make informed decisions in subsequent preprocessing and feature engineering stages, tailoring our approach based on the characteristics and relationships revealed through this explorative analysis.


## Data Preprocessing
Effective data preprocessing is critical for ensuring the quality and reliability of the model's predictions. Here's a detailed breakdown of the preprocessing steps applied to the dataset:

- **Handling Missing Values and Anomalies**: ().

- **Transforming Skewed Data**: We applied logarithmic transformations to `total_borrow` and other skewed features. This transformation helps in stabilizing the variance and normalizing the distribution, which enhances the model's performance as many machine learning algorithms assume normally distributed data.

- **Clipping Outliers**: To mitigate the influence of extreme outlier values in `total_borrow`, we applied clipping at the 99th percentile. This reduces the risk of the model being overly influenced by rare, extreme values and improves its robustness.

- **Encoding Categorical Variables**: Features like `token_borrow_mode` were transformed using label encoding, converting them from categorical to numeric formats. This is necessary as machine learning models inherently require numerical input for calculations.

- **Scaling Features**: All numerical features were scaled using `StandardScaler`. This standardization process adjusts the data to have a mean of zero and a standard deviation of one, ensuring that each feature contributes equally to the prediction, preventing bias towards features with larger scales.




## Feature Engineering
Effective feature engineering enhances model performance by introducing new information or simplifying existing information, which helps the machine learning model learn better patterns from the data. Here are the key steps we undertook:

- **Log Transformations**:
    - **Purpose**: We applied logarithmic transformations to features like `total_borrow`. This transformation is crucial for reducing skewness in variables that are highly skewed. It stabilizes variance and normalizes the distribution, making these features more model-friendly as many algorithms assume data is normally distributed.
    - **Implementation**: `df['log_total_borrow'] = np.log1p(df['total_borrow'])`

- **Interaction Features**:
    - **Purpose**: Interaction features allow models to understand and leverage the interaction between two or more variables. For instance, the interaction between total borrowed amounts and total repaid amounts can provide insights into the financial habits of users.
    - **Implementation**: `df['borrow_repay_interaction'] = df['log_total_borrow'] * df['log_total_repay']`
    - This feature captures the multiplicative effect of borrowing and repaying behaviors, which could be indicative of risk levels or financial stability.

- **Aggregated Metrics**:
    - **Purpose**: Aggregating data by user addresses helps in creating a comprehensive financial profile for each user. This is useful in identifying overarching patterns like consistent borrowing behavior or irregular repayment trends.
    - **Implementation**: We grouped transactional data by 'address' and calculated aggregate statistics such as sum, mean, and standard deviation for `total_borrow`, `total_repay`, and `total_liquidation`.
    - These aggregated features provide a macro view of each user's financial activity, which is essential for assessing creditworthiness or default risk.

- **Feature Selection via RFECV**:
    - **Purpose**: To enhance model performance and efficiency by reducing the number of features, thereby minimizing complexity and potential overfitting. Recursive Feature Elimination with Cross-Validation (RFECV) helps in identifying features that contribute the most to predicting the target variable.
    - **Implementation**: We used a `DecisionTreeRegressor` as the estimator in RFECV to iteratively remove the least important features based on their impact on model performance, using cross-validated selection to ensure robustness.
    
    ![features](https://res.cloudinary.com/dydj8hnhz/image/upload/v1715493978/sf2050b3uwgtdvjhris3.png)

```python
Optimal number of features : 23
Best features : Index(['log_total_borrow', 'total_borrow_clipped', 'count_borrow',
       'avg_borrow_amount', 'std_borrow_amount', 'borrow_amount_cv',
       'repay_amount_cv', 'total_deposit', 'count_deposit',
       'avg_deposit_amount', 'std_deposit_amount', 'deposit_amount_cv',
       'days_since_first_borrow', 'net_outstanding', 'net_deposits',
       'count_repays_to_count_borrows', 'avg_repay_to_avg_borrow',
       'net_outstanding_to_total_borrowed', 'net_outstanding_to_total_repaid',
       'count_redeems_to_count_deposits', 'total_redeemed_to_total_deposits',
       'avg_redeem_to_avg_deposit', 'net_deposits_to_total_redeemed'],
      dtype='object')

```


## Model Development

Breakdown of our model development process:

- **Model Architecture**:
    - **Configuration**: A deep feed-forward neural network consisting of multiple layers. The architecture includes:
        - An input layer sized according to the number of features.
        - Five hidden layers with 128, 64, 32, 16, and 8 neurons respectively. This gradual reduction in layer size helps in refining the features to the most crucial elements before making a prediction.
        - A single neuron in the output layer to predict the continuous target variable, the liquidation ratio.
    - **Activation Functions**: 
        - **ReLU (Rectified Linear Unit)**: Used in all hidden layers, ReLU helps to introduce non-linearity to the model, enabling it to learn more complex patterns in the data. It is preferred for its efficiency and effectiveness in reducing the vanishing gradient problem.

- **Compilation and Optimization**:
    - **Optimizer**: We chose the Adam optimizer for training the model. Adam is widely recognized for its efficiency in handling sparse gradients and adaptive learning rate capabilities, which makes it superior for datasets with diverse feature scales and complexities.
    - **Loss Function**: Mean Squared Error (MSE) was used as the loss function. It measures the average of the squares of the errors—that is, the average squared difference between the estimated values and the actual value. This loss function is well-suited for regression problems and helps in minimizing the prediction error.

- **Model Training**:
    - **Epochs and Batch Size**: The model was trained for 100 epochs with a batch size of 32. This configuration was chosen to balance between computational efficiency and the need for the model to learn from different subsets of data adequately.
    - **Validation Split**: During training, 20% of the training data was used as a validation set. This helps in monitoring the model's performance on unseen data, preventing overfitting and ensuring that the model generalizes well.

    loss vs epochs

    ![loss vs epochs](https://res.cloudinary.com/dydj8hnhz/image/upload/v1715494468/amiw32svvsn7v3z6soqj.png)


## Evaluation


### Test Set Evaluation
The model was first evaluated using a reserved portion of the dataset, the test set. Here are the key performance metrics observed:

- **Mean Squared Error (MSE)**: The MSE on the test set was 0.005560641176998615. This metric helps in quantifying the average of the squares of the prediction errors, which is the difference between the predicted and actual values.
- **Root Mean Squared Error (RMSE)**: The RMSE, derived from the MSE, was 0.0745697068319208. RMSE provides a more interpretable measure as it is in the same units as the target variable. It indicates how closely the model's predictions match the actual values.

### Classification Metrics
Given the project's goal to predict credit risk levels, classification metrics were computed to assess the model's accuracy in classifying the liquidation risk as high or low:
- **Precision**: Achieved a precision of 0.9672816728167282, indicating the model's accuracy in identifying truly high-risk cases among all predicted high-risk cases.
- **Accuracy**: The overall accuracy was 0.9898637219544889, reflecting the model's effectiveness across all classifications.
- **Recall**: The recall rate was 0.9551571685371423, measuring the model's ability to identify all actual high-risk cases.

### Loss Trends
During training, the loss on both the training set and the validation set was monitored:

- **Training and Validation Loss**: The model displayed a consistent decrease in loss over epochs, confirming learning progress. Notably, the validation loss exhibited some fluctuations ('spikiness'), but overall, it trended downwards in conjunction with the training loss. This spikiness in the validation loss could be attributed to the model encountering slightly different or more challenging patterns in the validation data, suggesting areas where model robustness could potentially be improved.


## Results
The model demonstrated high accuracy and precision, indicating a strong capability to identify high-risk transactions effectively. RMSE values on test and evaluation datasets were low, suggesting good predictive performance.

