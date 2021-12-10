## MLZoomCamp Capstone Project

# **Building & Deploying a Python ML Model for Motor Vehicle Collision Prediction**

## Introduction:

Motor Vehicle collisions occur when a vehicle collides with another vehicle, pedestrian, animal or any other object on the road. Such collisions often result in injury, disability, death, damage to life as well property. This project is based on collision accident between a person and a motor vehicle that occurred in New York city during 2021. The data was sourced from NYC Open Data[https://opendata.cityofnewyork.us/]. The purpose of this project was to understand issues like - what is the frequency and type of injuries in such collisions; which are the most common factors influencing such motor vehicle collisions in New York etc. I studied a publicly available dataset on Kaggle and built a model for predicting collision incidents using Supervised Machine learning techniques in Python.

Firstly, I prepared my data by cleaning and formatting the dataset. Then, I used Exploratory Data Analysis techniques in Python to visualize relationships between different features in dataset like BODILY_INJURY, SAFETY_EQUIPMENT, EJECTION, EMOTIONAL_STATUS and their relation with the target variable, PERSON_INJURY. After cleaning, I pre-processed my dataset to train multiple Supervised Machine Learning models by splitting the data into subsets, identifying & segregating the feature matrix from the target variable and one-hot encoding of categorical variables. I used multiple Binary Classification models from scikit-learn library in Python to train my dataset and make predictions. I used both Regression-based & Tree-based models. Each model was evaluated using classification metrics AUC score and their performances compared. Thereafter the models were tuned using different parameters to find the most optimal parameters. Lastly, multiple final models with optimal parameters were run to select the Best Model for our NYC Motor Vehicle Collisions dataset. For each algorithms used, I also analyzed which were the most important features influencing the predictions made about the target variable, PERSON_INJURY.   

Once the best model for my dataset was selected its code was exported from a Python Notebook into a Python script. Thereafter, this model was put into a web service using Flask. Then, a Python virtual environment was created using Pipenv containing all the necessary dependencies and packages for running the model. The model was first deployed locally using Docker as a containerized service. Then, this locally deployed model was used to predict the incidence of collision injury for a new 'sample collided person' with unseen data as input. Finally, our deployed model gave as output, details regarding the risk of collision_injury (as True or False) and probability of collision_injury for a 'sample collided person' as inputs to the model. 

Lastly, I also tried deploying the collision injury prediction service to the cloud using AWS Elastic Beanstalk. This cloud environment created for the collision injury prediction service was then used to make predictions about a new 'sample collided person' as input to our model.




## Data Sources and Preparation: 

### Sources of Data -

For this project, I retrieved data from Kaggle Public Dataset - [NYC Motor Vehicle Collisions to Person](https://www.kaggle.com/kukuroo3/nyc-motor-vehicle-collisions-to-person) Data was shared on Kaggle as a csv file (*NYC_Motor_Vehicle_Collisions_to_Person.csv*).

### Data Processing Steps - 

Following steps were used in preparing the data for this project:-

* I processed the dataset using Python in-built functions and libraries such as **Pandas** for manipulation and structuring them as DataFrames.**Numpy** functions along with statistical analysis techniques like mean used to fill for missing values in the dataset. 
* Once the data was cleaned & formatted it was used to perform **Exploratory Data Analysis (EDA)** and create a series of visualizations for drawing insights.
* Different categorical features in the dataset like PERSON_TYPE, COMPLAINT, PERSON_SEX etc. were converted into dictionaries using one-hot encoder **DictVectorizer**.
* **Correlation** among categorical features was computed to understand their relation with target variable, PERSON_INJURY using Chi-square test.
* **Feature Importance** techniques were implemented for different ML algorithms to understand which features affected predictions about PERSON_INJURY variable the most.
* **Binary Classification (Supervised) ML algorithms** both regression-based (like *Logistic Regression*) and tree-based (like *Decision Trees*) were used from **scikit-learn** library to train and make predictions. **Ensemble methods** using bagging like *Random Forest* and boosting like *eXtreme Gradient Boosting or XGBoost* were also used. 
* Models were compared and their performance evaluated using metrics like - roc_auc_score, f1_score, classification_report etc. 
* **GridSearchCV** parameter-tuning techniques were used to tune the parameters of each model, improve their performance and select the most optimal parameters.
* **Cross-Validation** technique like **K-Fold** was also used to split the dataset into ‘k’ number of subsets, then use k-1 subsets to train the model and use the last subset as a validation set to test the model. Then the score of the model on each fold was averaged to evaluate the performance of the model.
* **CatBoost for Classification** algorithm was used as an additional exercise in this project, to explore more from this dataset. CatBoost or **Categorical Boosting** is an open-source boosting library developed by Yandex. The greatest advantage of CatBoost is that it automatically handles categorical features, using Ordered target statistics. As this dataset mostly has categorical features, making predictions using CatBoost and tuning its parameters acted as a good experiment to learn.
* After evaluating multiple models the Best Model was chosen as **Random Forest for Classification**. Hereafter, this best model was used (as a Python script) in a Web service (using **Flask**) and deployed locally (using **Docker**).
* Lastly, the Collision Injury Prediction service, was also deployed in the cloud using **AWS Elastic Beanstalk**. For this, Elastic Beanstalk command-line interface (CLI) was added as a development dependency (only for the project). Then an environment for the collision injury prediction service was created in AWS which successfully launched the environment and provided a public url to reach the application. This url was finally used to make collision injury predictions about details of the new 'sample collided person' as input. 




