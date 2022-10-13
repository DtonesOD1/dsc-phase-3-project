# dsc-phase-3-project
Flatiron Data Science Phase 3 Project
# Business Understanings:
From our initial look at Telecom's data, we see that they have an average churn rate of essentially 15%. There was no time frame given but this is usually measured over the course of a year. A general rule of thumb in business is that anything above 10% in terms of churning is too high.
This leads us to believe that the company is losing money.
There has been data that shows that acquiring new customers can cost 5x as much as retaining existing ones. Keeping this adage in mind is what is driving this project to attempt to lower the churn rate to anywhere from 10%-7%. If this can be successfully done the company can greatly increase it's profit margins.

# Data Understanding:
We gathered our needed data and explored what information was available to us to work with. A minimal amount of cleaning was needed, but now we need to continue our process of exploratory data analysis. We will inspect the dataframe and see what features are tied to the customer churn rate, and then attempt to exploit the results into better customer retention.
# Data Preperation:
we saw that there are 20 features and 3,333 observations/customers with an approximate churn rate of 15%. We will continue our exploratory data analysis with more visuals and looking at each individual feature to see what will best work for our modeling in terms of predicting and preventing customer turnover.

  ### EDA
  
 Moving forward it would be good to have an approach in terms of attempting to find what features are most likely to indicate customer churn rate. In the past we have used a correlation table which shows what features are most correlated to a chosen feature. We will use that again and work from there.
 
churn                    1.000
international_plan       0.260
customer_service_calls   0.209
total_day_minutes        0.205
total_day_charge         0.205
voice_mail_plan          0.102
total_eve_minutes        0.093
total_eve_charge         0.093
number_vmail_messages    0.090
total_intl_charge        0.068
total_intl_minutes       0.068
total_intl_calls         0.053
total_night_charge       0.035
total_night_minutes      0.035
total_day_calls          0.018
account_length           0.017
total_eve_calls          0.009
state                    0.008
total_night_calls        0.006

A strong visual tool available for us is 'sweetviz' which takes a deep look at each feature and plots them out against churn, it also shows things such as if there are missing values and associations.

# Data Modeling:

### Vanilla Model Findings:
The initial vanilla model was no more than a Logistic Regression model. A logistic regression model, simply, is used to estimate the relationship between a dependent and one or more independent variables. In this case we are using it to model the probability of a customer churning.
#### Metrics Findings
The metrics printed off in the previous kernel are used to quantify the performance of the classifiers.
log loss is difficult to interpret but generally indicates how close the probability and predictions are, good to use as a baseline and lower is generally better
Precision is a straightforward measurement of true positives divided by true negatives
Recall is essentially the inverse of precision
Accuracy measures both true positives and true negatives by total observations
F1 is a harmonic mean of both precision and recall
#### Confusion Matrix
A confusion matrix is a good way to evaluate a classifier. In order to understand these matrix results I had the percentages displayed in each quadrant instead of the actual number since the dataset is not balanced. A good model will have a high percentage of True Positives and True Negatives. This initial model accurately predicted True Positives 77% of the time and True Negatives 73% of the time. Not a bad start but we would like to see higher.
#### *A Note on Metrics:
A general rule of thumb with metrics is that some are better than others depending on the type of datasets and models you are working with. In our case we have an imbalanced data set and log-loss metrics are usually not used here due to them being difficult to interpret. Additionally, the accuracy score should not be used by itself for the same reason. We also mentioned that F1 is a harmonic mean of both precision and recall, which makes them useful for both imbalanced data sets and working with a binary class, therefore F1 is the metric we will focus on going forward. A good article on using the F1 metric on imbalanced datasets: https://neptune.ai/blog/balanced-accuracy
#### *A Note on Weighted Data:
It should be mentioned that if the imbalance of this dataset was not taken into account when splitting the data and instantiating the logistic regression model, these results would be significantly different and far worse. A good reference article on this topic can be found here: https://towardsdatascience.com/weighted-logistic-regression-for-imbalanced-dataset-9a5cd88e68b

### Choosing a Model:
The classifiers that we will explore in this next step are:
#### Random Forest: is an ensemble of decision trees, where each tree uses a process called 'Bagging' to ensure each tree is trained on different samples of data.
#### K-Nearest Neighbors: is a distance-based classifier, meaning that it implicitly assumes that the smaller the distance between two points, the more similar they are.
#### Decision Tree: is a supervised learning method used for classification and regression with the goal of creating a model that predicts the value of a target variable by learning simple decision rules inferred from the data features.
These models will be compared along with the Logistic Regression model already created.

#### XGBoost
So we are making some progress in terms of finding out which classifiers work best for our case, but it would be remiss to not include one of the newer and most powerful concepts in machine learning right now, XGBoost.
We mentioned that our classic classifiers use methods like bagging and leaves/branches to create decision trees, but when it comes to XGBoost it also includes a method called Gradient Descent Boosting. A good introduction to this method and algorithms used can be found here: https://www.nvidia.com/en-us/glossary/data-science/xgboost/.

#### Tuning:
Moving forward we will tune and iterate our models and perform grid searches that will ensure optimal results in our search for the best hyperparameters.
There will be a function written to return our new tuned test and training scores and we will take a closer look at the corresponding matrices.

# Evaluation

Our last model is still showing a strong F1 score, albeit a little lower than the previous 2 models. But what we did see an improvement in is the Type 1 error of False Positives dropped to 19% and it also showed an increase in predicting true positives at 81%.
The final XGBClassifier model will be the model that we choose to move forward with, we will go over the reasons why.
F1: The final F1 scores were technically slightly lower than the some of the other models but still very strong. We see with the delta that the scores are very close together which may indicate slight overfitting which XGB tends to do, but the numbers are not too good to be true which is promising that the model still has a very good fit and we took several precautions with our models to help avoid the issue.
Confusion Matrix: This is where I made the decision to pick this model. The correct prediction of True Negatives at over 96% is incredibly strong and close to the other models, but the 81% of correctly predicting True Positives is significantly stronger than the other choices. It was also showed an improvement of predicting Type 1 errors at 19%. As mentioned earlier, from an economical perspective this has the potential to save the company money.
According to the Sklearn XGBoost manual, with an imbalanced data set a good evaluator of the model is the Area Under Curve score, which came in at a strong 89%. Anything at 90% or above is considered excellent, so we are happy with this score and is another indicator that this is a good model.

# Conclusion:
Conclusion
XGBoost uses supervised learning to create parallel tree boosting models that have proven to be some of the fastest and most accurate in the industry. From our model we created a strong F1 score, which as mentioned earlier, is good for both imbalanced data sets and binary classes. Also noted was that according to the XGBoost manual the AUC score is one of the best evaluators of the model. Both our F1 and AUC scores were strong and show minimal overfitting which we attempted to avoid by addressing the imbalance by paying attention to items such as 'class_weight' and 'scale_pos_weight'. The confusion matrix is a very important tool in our modeling, but can sometimes turn into a pick your poison scenario. Very rarely will all the quadrants meet your expectations as tuning the model may improve one quadrant at the expense of another. Therefore a decision must be made specific to the business problem at hand. That is why in our case I selected the model with the lowest type 1 error since the entire business problem is based around avoiding customers leaving. We want to be able to predict what is making customers leave to create an optimal strategy, with as few customers as possible slipping through the cracks.
For future work it would be beneficial to continue to add to the dataset to keep a track of current trends. It would also be helpful to know more information pertaining to the more important features of the model, for example:
How much extra does the voicemail plan cost for customers and what is the profit margin for the company
What is the average length of time a customer spends on the phone with customer service
A customer survey could also be a good idea.



