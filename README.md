# Analysis of an E-commerce Dataset - README

**Name:** Taha Naveed Shibli  

## Introduction

This README provides an overview of the "Analysis of an E-commerce Dataset." In this analysis, I focused on training linear regression models to predict user ratings based on various features within the dataset. The primary objectives of this assignment were to explore the dataset, examine the impact of feature selection, and assess how different proportions of training and testing data affect model performance.

## Dataset Import

The analysis begins with the import of the e-commerce dataset from the 'ecommerce_dataset.csv' file. This step is essential for all subsequent tasks. 

## Dataset Exploration

To better understand the dataset, an exploration was conducted:

- The structure and data types of the dataset were examined, offering insights into the number of columns and their respective data types.
- Correlations between various features and user ratings were calculated. To perform this, the dataset's categorical features were converted into numerical values.

### Explanations and Analysis on the Correlations

The correlation analysis unveiled critical insights:

- **Correlation Between Helpfulness and Rating:** A weak negative correlation suggests that the helpfulness of a review has little to no impact on the rating assigned to it.

- **Correlation Between Gender and Rating:** Another weak correlation indicates that the gender of the reviewer has negligible influence on the rating.

- **Correlation Between Category and Rating:** A moderate negative correlation implies that the category of the product being reviewed influences the rating. Different product categories may exhibit distinct rating patterns.

- **Correlation Between Review and Rating:** A moderate correlation suggests that the specific words or phrases used in a review have a reasonable influence on the rating.

## Analysis on the Correlations

This section aims to determine which features are the most and least correlated with the ratings:

- **Most Correlated Features:** The analysis identified "Category" and "Review" as the most correlated features with the user ratings.

- **Least Correlated Features:** "Helpfulness" and "Gender" were found to be the least correlated with user ratings. These features are less likely to significantly improve rating prediction accuracy.

### Understanding the Impact of Correlation on Prediction Results

The correlations provided insights into how specific features may affect rating prediction:

- The weak correlation between "Helpfulness" and "Rating" suggests that this feature has minimal impact on prediction accuracy.

- Similarly, the weak correlation between "Gender" and "Rating" indicates that the gender of the reviewer has limited influence on rating prediction.

- In contrast, the moderate correlation between "Category" and "Rating" implies that the product category significantly influences the predicted rating. 

- The moderate correlation between "Review" and "Rating" shows that the words and phrases used in reviews have a reasonable impact on rating prediction.

In summary, based on the correlations, it is suggested that "Review" and "Category" are the most informative features for predicting ratings, while "Gender" and "Helpfulness" may have less impact on the final rating prediction. However, it is essential to remember that correlation does not imply causation, and other factors may influence prediction results.

## Splitting Training and Testing Data

To train machine learning models, the dataset was divided into training and testing sets. Two scenarios were explored:

- **Case 1:** Training data containing 10% of the entire dataset.
- **Case 2:** Training data containing 90% of the entire dataset.

## Train Linear Regression Models with Feature Selection under Cases 1 & 2

Four linear regression models were trained using different combinations of features and training data scenarios:

- **Model A:** Trained with 10% training data using the two most correlated features.
- **Model B:** Trained with 10% training data using the two least correlated features.
- **Model C:** Trained with 90% training data using the two most correlated features.
- **Model D:** Trained with 90% training data using the two least correlated features.

## Evaluate Models

The performance of these models was evaluated using Mean Squared Error (MSE) and Root Mean Squared Error (RMSE).

## Visualize, Compare, and Analyze the Results

The results were visualized using bar plots, and insightful analysis was provided. The comparison of Model A and Model C with Model B and Model D was conducted to understand the impact of feature selection and the size of training and testing data on model performance.

The results showed that models trained with the most correlated features and more training data had lower MSE and RMSE values, aligning with the expectation that relevant features and a larger dataset lead to better performance. In cases where this wasn't observed, potential factors such as overfitting, feature relevance, and data quality were considered as explanations.

In conclusion, this analysis demonstrates the importance of feature selection, correlation analysis, and data splitting in predicting user ratings in an e-commerce dataset. Careful consideration of these factors is crucial for making informed decisions in data science.

********************************************************************************************************************************************

### Final Results:

**Task Completion:**
1. **Data Import:** The dataset has been successfully imported, and its length has been displayed.
2. **Data Exploration:** The dataset structure, including the number of columns and data types, has been explored. Correlations between key features and ratings have been calculated.
3. **Analysis of Correlations:** Correlations between helpfulness, gender, category, review, and rating have been analyzed. The most and least correlated features regarding rating have been identified.
4. **Understanding the Impact:** The impact of these correlations on rating predictions has been discussed, providing insights into which features are likely to be more influential.

**Training and Testing Data Split:**
- The dataset has been split into two scenarios: 10% training data (Case 1) and 90% training data (Case 2). The shape of training and testing sets for each case has been printed.

**Training Linear Regression Models:**
- Four linear regression models (A, B, C, D) have been trained with different combinations of features and training data.

**Model Evaluation:**
- The performance of each model has been evaluated using Mean Squared Error (MSE) and Root Mean Squared Error (RMSE). The results are as follows:

    - Model A (10% training data, most correlated features):
      - MSE: 1.769
      - RMSE: 1.330

    - Model B (10% training data, least correlated features):
      - MSE: 1.841
      - RMSE: 1.357

    - Model C (90% training data, most correlated features):
      - MSE: 1.759
      - RMSE: 1.326

    - Model D (90% training data, least correlated features):
      - MSE: 1.811
      - RMSE: 1.346

### Conclusions:

1. **Impact of Correlations:**
   - The analysis of correlations revealed that category and review are the most correlated features regarding ratings. This suggests that these features have a significant influence on the prediction of ratings.
   - In contrast, helpfulness and gender are the least correlated features. Their impact on rating prediction is relatively weak.

2. **Data Splitting:**
   - Splitting the dataset into different proportions of training and testing data (10% and 90%) allowed for a comparison of model performance under different conditions.

3. **Model Performance:**
   - Model A, which used 10% training data with the two most correlated features, demonstrated the lowest MSE and RMSE among all models.
   - Model B, with the two least correlated features, showed the highest MSE and RMSE among the 10% training data models.
   - Model C, trained with 90% of the data using the most correlated features, had a lower MSE and RMSE compared to Model D, which used 90% data with the least correlated features.

### Valuable Insights:

1. **Feature Importance:** The results highlight the importance of feature selection in model building. Using the most correlated features for training (category and review) improved model performance, while using less correlated features (helpfulness and gender) led to poorer predictions.

2. **Data Split Impact:** Splitting data into varying proportions for training and testing can significantly affect model performance. In this case, using a larger training dataset (Case 2) generally resulted in better models, emphasizing the importance of having sufficient data for training.

3. **Overfitting Consideration:** While models with more data and highly correlated features performed better in this case, it's essential to consider the possibility of overfitting. Overfitting occurs when a model learns noise in the training data, leading to poor generalization. Close monitoring of overfitting is crucial.

4. **Causation vs. Correlation:** The analysis underscores the need to understand that correlation does not imply causation. Even highly correlated features may not necessarily be causal factors in rating predictions. Additional domain knowledge may be required to make more informed decisions.

Overall, the task successfully achieved the stated tasks, providing valuable insights into the importance of feature selection and data splitting in the context of rating prediction in e-commerce datasets. It also emphasizes the need for data scientists to carefully analyze and evaluate their models to draw meaningful conclusions and make informed decisions.