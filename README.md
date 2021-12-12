## MLZoomCamp Capstone Project

# **Building & Deploying a Model for Predicting Collision Injuries from Motor Vehicles**

## Introduction:

Motor Vehicle collisions occur when a vehicle collides with another vehicle, pedestrian, animal or any other object on the road. Such collisions often result in injury, disability, death, damage to life as well property. The purpose of this project was to study the collisions of persons with Motor vehicles in New York during 2021. I analyzed issues like, the frequency and type of injuries in such collisions; the most common factors influencing such motor vehicle collisions and outcomes for life and property as a result of such collisions.

This project is based on a publicly available dataset on Kaggle, [NYC Motor Vehicle Collisions to Person](https://www.kaggle.com/kukuroo3/nyc-motor-vehicle-collisions-to-person) which sources its data from [NYC Open Data](https://opendata.cityofnewyork.us/).  I studied this dataset on Kaggle and built a Web service for predicting collision injuries using Supervised Machine learning techniques in Python.

Below is a summary of steps undertaken in this project - 

* Firstly, I prepared my data by cleaning and formatting the dataset. Then, I used Exploratory Data Analysis techniques in Python to visualize relationships between different features in dataset like BODILY_INJURY, SAFETY_EQUIPMENT, EJECTION, EMOTIONAL_STATUS and their relation with the target variable, PERSON_INJURY. 
* After cleaning, I pre-processed my dataset to train multiple Supervised Machine Learning models by splitting the data into subsets, identifying & segregating the feature matrix from the target variable and one-hot encoding of categorical variables. I used multiple Binary Classification models from scikit-learn library in Python to train my dataset and make predictions. I used both Regression-based & Tree-based models. 
* Each model was evaluated using classification metrics AUC score and their performances compared. Thereafter the models were tuned using different parameters to find the most optimal parameters. 
* Lastly, multiple final models with optimal parameters were run to select the Best Model for our NYC Motor Vehicle Collisions dataset. For each algorithms used, I also analyzed which were the most important features influencing the predictions made about the target variable, PERSON_INJURY.   
* Once the best model for my dataset was selected its code was exported from a Python Notebook into a Python script. Thereafter, this model was put into a web service using Flask. Then, a Python virtual environment was created using Pipenv containing all the necessary dependencies and packages for running the model. 
* The model was first deployed locally using Docker as a containerized service. Then, this locally deployed model was used to predict the incidence of collision injury for a new 'sample collided person' with unseen data as input. Finally, our deployed model gave as output, details regarding the risk of collision_injury (as True or False) and probability of collision_injury for a 'sample collided person' as inputs to the model. 
* Lastly, I also tried deploying the collision injury prediction service to the cloud using AWS Elastic Beanstalk. This cloud environment created for the collision injury prediction service was then used to make predictions about a new 'sample collided person' as input to our model.


Now let us go into the details of each step:-

## Data Sources and Preparation: 

### Sources of Data -

For this project, I retrieved data from Kaggle Public Dataset - [NYC Motor Vehicle Collisions to Person](https://www.kaggle.com/kukuroo3/nyc-motor-vehicle-collisions-to-person) This dataset was shared on Kaggle as a csv file (*NYC_Motor_Vehicle_Collisions_to_Person.csv*).

### Data Processing Steps - 

Following steps were used in preparing the data for this project:-

* I processed the dataset using Python in-built functions and libraries such as **Pandas** for manipulation and structuring them as DataFrames.**Numpy** functions along with statistical analysis techniques like mean used to fill for missing values in the dataset. 
* Once the data was cleaned & formatted it was used to perform **Exploratory Data Analysis (EDA)** and create a series of visualizations for drawing insights.
* Different categorical features in the dataset like PERSON_TYPE, COMPLAINT, PERSON_SEX , PERSON_INJURY etc. were converted into dictionaries using one-hot encoder **DictVectorizer** and **LabelEncoder**.
* **Correlation** among categorical features was computed using **Chi-square test** to understand their relation with the target variable, PERSON_INJURY.
* **Feature Importance** techniques were implemented for different ML algorithms to understand which features affected predictions about PERSON_INJURY variable the most.
* **Binary Classification (Supervised) ML algorithms** both regression-based (like *Logistic Regression*) and tree-based (like *Decision Trees*) were used from **scikit-learn** library to train and make predictions. **Ensemble methods** using bagging like *Random Forest* and boosting like *eXtreme Gradient Boosting or XGBoost* were also used. 
Models were compared and their performance evaluated using metrics like - roc_auc_score, f1_score, classification_report etc. 
* **Hyperparameter-tuning techniques** like **GridSearchCV** were used to tune the parameters of each model, improve their performance and select the most optimal parameters.
* **Cross-Validation** technique like **K-Fold** was also used to split the dataset into ‘k’ number of subsets, then use k-1 subsets to train the model and use the last subset as a validation set to test the model. Then the score of the model on each fold was averaged to evaluate the performance of the model.
* **CatBoost for Classification** algorithm was used as an additional exercise in this project, to explore more from this dataset. CatBoost or **Categorical Boosting** is an open-source boosting library developed by Yandex. The greatest advantage of CatBoost is that it automatically handles categorical features, using Ordered target statistics. As this dataset mostly has categorical features, making predictions using CatBoost and tuning its parameters acted as a good experiment to learn.
* After evaluating multiple models the Best Model was chosen as **Random Forest for Classification**. Hereafter, this best model was used (as a Python script) in a Web service (using **Flask**) and deployed locally (using **Docker**).
* Lastly, the Collision Injury Prediction service, was also deployed in the cloud using **AWS Elastic Beanstalk**. For this, Elastic Beanstalk command-line interface (CLI) was added as a development dependency (only for the project). Then an environment for the collision injury prediction service was created in AWS which successfully launched the environment and provided a public url to reach the application. This url was finally used to make collision injury predictions about details of the new 'sample collided person' as input. 




## Exploratory Data Analysis (EDA), Correlation & Feature Importance: 


Name of the Python Notebook - [ML_ZoomCamp_Capstone_Project_NYC_MV_Collisions.ipynb](https://github.com/sukritishuk/ML_ZoomCamp_Capstone_Project/blob/main/ML_ZoomCamp_Capstone_Project_NYC_MV_Collisions.ipynb)

Below are the important components from this Notebook:-
* **Data Loading** - Firstly, I loaded all the basic libraries used in the project. Then, I imported the data from the .csv file into a Pandas DataFrame and got a snapshot of data (like exploring the unique values in each column of dataset, getting a statistical summary of dataset).

![image](https://user-images.githubusercontent.com/50409210/145678001-3c5a240a-f94e-4a5c-92e3-d9204ccf1ced.png)

This dataset mostly had categorical features (as shown below) and the target variable was PERSON_INJURY' with primarily binary outcomes 'Injured' and 'Killed'. There were 2-3 time-series related features (like CRASH_DATE & CRASH_TIME) as well.  

![image](https://user-images.githubusercontent.com/50409210/145683746-7a173296-60b7-448d-a9b6-f5cf2d9b5a32.png)


*  **Data Cleaning and Formatting** - I cleaned the dataset by taking the following steps:-
   a) Removing Irrelevant data or Columns - like dropping columns VEHICLE_ID, PERSON_ID, UNIQUE_ID, COLLISION_ID
   b) Imputing Missing rows with mean or mode of values - like imputing missing PERSON_AGE with mean PERSON_AGE
   c) Changing Data Type for Columns - like changing CRASH_DATE to datetime format or PERSON_AGE to integers
   d) Replacing Column Values with Specific Values - like replacing 'Does Not Apply' values with 'Unknown'
   e) Feature Creation from existing feature columns - like creating CRASH_Mnth_Name column from CRASH_DATE column to extract Months from Dates


* **Exploratory Data Analysis (EDA)** - After this, I performed EDA using Python libraries like Matplotlib, Pandas and Seaborn to analyze the data and visualize its key components. Using multiple visuals and subplots I tried to answer important questions like ***When do the most traffic accidents occur in NYC during 2021?***
      
![image](https://user-images.githubusercontent.com/50409210/145678529-b005475b-e4bf-4c51-83f9-b509865b68ba.png)

As my dataset contained more categorical variables countplot was the primary plot used for most visualizations to understand the relation of different features like EJECTION, PERSON_TYPE etc. with the target variable, PERSON_INJURY. For time-series specific columns like CRASH_Mnth_Name and CRASH_TIME I used barplots to plot the count of Hourly Crashes and Monthly Collisions in 2021. PERSON_AGE column was also analyzed with other features using barplots to study, ***Which age and gender people were involved in most of the NYC collisions?*** or ***Which were most common type of BODILY_INJURY and COMPLAINT faced by those injured after the collision?***

![image](https://user-images.githubusercontent.com/50409210/145678877-e601148b-d9a7-48be-9e81-33fa352a20f9.png)

Following **interesting insights** were drawn from these plots - 
* The month of June appeared to have the most number of crashes on NYC roads during 2021, while November had the least.
* Also, most number of collisions in 2021 on NYC roads, seem to have happened around 16:00 or 4:00 pm in the evening.
* In thw NYC collisions more Females were killed than Males. Also, the proportion of Females injured were quite similar to that of injured Males.
* The count of collisions for different types of Ejection shows that most of the people were Injured and that too because they could Not Eject. Also, of those injured in collisions most people were Occupants of the motor vehicle themselves or Pedestrians.
* Most Females killed were in age group 50-60 years while, most Males killed were in age group 40-50 years.

![image](https://user-images.githubusercontent.com/50409210/145679010-49818c6e-ccb2-4dd9-b6ed-6f11bc50a730.png)

* The Pedestrian Location of most of the people injured in NYC collisions in 2021 were not Known (Unknown). Most of the injured people were in Conscious state and in most cases those injured were in the role of Driver at the time of collision.
* The injured faced mostly Back, Knee-Lower Leg Foot and Neck injuries after collision. Those injured most often complained about Pain or Nausea after meeting with accident.
 
 ![image](https://user-images.githubusercontent.com/50409210/145683308-a596322c-fddb-46b4-a908-fcd15bb49c59.png)

* Mostly the injured persons during collision were at Driver's position in the vehicle. The Lap Belt & Harness were used as Safety Equipment by most injured during the NYC Collisions in 2021. This is ironical as it shows that, people got injured even after using safety equiments. However, the second highest of those injured were those about whom the use of Safety Equipment was Unknown.


Hereafter, I used Chi-square test to compute the Correlation between each of the Categorical Features ('BODILY_INJURY','CONTRIBUTING_FACTOR_2', 'EMOTIONAL_STATUS', 'PERSON_SEX' etc.) and the target variable, PERSON_INJURY. As only one feature PERSON_AGE was numerical in our dataset we did not find its correlation with our target separately.
The following **insights** were drawn from them -
* Only 'CONTRIBUTING_FACTOR_2' feature had a p-value > 0.5 hence, accepting our Null Hypothesis we found that only this feature is not correlated to our target variable, PERSON_INJURY.
* For all other features, their respective p-value < 0.5 so, we rejected the Null Hypothesis and declared that they are correlated with our target variable.


* **One-hot Encoding of Categorical Data Using DictVectorizer & LabelEncoder** - As this dataset contained mostly categorical features like PERSON_TYPE, COMPLAINT, EMOTIONAL_STATUS etc. these were encoded using DictVectorizer before being used further in training ML algorithms and making predictions. DictVectorizer helped in transforming lists of feature-value mappings to vectors i.e., feature matrix into dictionaries for training and predicting on subsets. When feature values were strings, this transformer would do a binary one-hot coding. 

In addition, to DictVectorizer, LabelEncoder was also used to encode the target variable column , 'PERSON_INJURY' into integers for 'Injured' and 'Killed' values. This would help in training different models and making predictions.The fit_transform methos was used from LabelEncoder() to encode the data and keep a memory of encodings simultaneously.


* **Feature Importance Using Mutual Information Score** - To understand the importance of fetaures in dataset Mutual Information metric was computed for different features with the PERSON_INJURY variable. It was found that the knowledge about EMOTIONAL_STATUS will be the most certain while knowledge about CONTRIBUTING_FACTOR_2 will be the least certain in giving information about our target variable PERSON_INJURY. I also computed the important features in the dataset for each of the models Decision tree, Random Forest and XGBoost so as to identify how they differed among models.

![image](https://user-images.githubusercontent.com/50409210/145687426-0856a1ca-1c30-44ab-b7d9-440b9eb0e5be.png)

* **Computing Difference & Risk Ratio for Features** - Risk Ratios or relative risk, is a metric that measures the risk-taking place in a particular group and comparing the results with the risk-taking place in another group. Here it helped in finding interesting facts about those injured or killed in NYC collisions - like people with Lap Belt & Harness as safety equipment, Occupants of vehicles, those in Driver seat or those who could Not Eject were more likely to get injured. Most people post collision Complained of Pain or Nausea, were Conscious. These facts would further help us find categories or variables which would make predictions about Person's Injury status after a collision using ML algorithms.



## Model Selection and Tuning & Evaluation:


* **Setting up the validation framework** - Firstly, I split the dataset into training, validation and testing subsets in the ratio of 60:20:20. Then, I defined the feature matrix containing all the factor columns and defined the 'PERSON_INJURY' column as the target variable. I also ensured that the target column was removed from each of the 3 data subsets.

* **Model Selection & Evaluation** - Once the data was split and pre-processed for machine learning algorithms I implemented different models by training them on the full_train set and made predictions on the validation set. The models were then evaluated using Classification metrics like roc_auc_score, confusion_matrix, classification_report etc. to compare their performances. 

Following were the different modelling algorithms I used in this project:

      * Logistic Regression
      * Decision Trees for Classification
      * Random Forest for Classification (using Ensemble learning, bagging)
      * XGBoost for Classification (using Gradient boosting)
      * Additional exploration - using CatBoost or Categorical Bossting  for Classification 
      
For each of the model feature importance was computed to identify which features contributed most to the predictions about collision-related injuries. AUC scores were computed both for the training set & the validation set separately to compare model's performance.

![image](https://user-images.githubusercontent.com/50409210/145686737-8c15f03d-ffbc-48c1-8321-5151036101c2.png)

* **Parameter Tuning of Models** - The parameters for each of the above models were also tuned using **Grid Search Cross Validation (GridSearchCV)** to find the most optimal parameters giving the following as outputs:
   a) .best_params_ - gives the best combination of tuned hyperparameters (Parameter setting that gave the best results on the hold out data)
   b) .best_score_ - gives the Mean cross-validated score of the best_estimator 
   c) .best_estimator_ - estimator which gave highest score (or smallest loss if specified) on the left out data
   
Following were the parameters tuned for each model.
     * Decision Trees for Classification - max_depth , min_samples_leaf and max_features 
     * Random Forest for Classification - n_estimators, max_depth, min_samples_leaf, max_features
     * XGBoost for Classification - eta, max_depth, min_child_weight
     * CatBoost for Classification - learning_rate, max_depth
 
After tuning each model with different parameters the most optimal parameters were selected for the model. This became the Final Model after Hyperparameter Tuning yielding the best AUC score:-

    * Final Decision Tree Model - max_depth = 5, min_sample_leaf = 20 and max_features = 10
    * Final Random Forest Model - max_depth = 15, min_sample_leaf = 1, and n_estimators = 70 and max_features = 8
    * Final XGBoost Model -(training for 200 iterations) - max_depth = 4, min_child_weight = 1 and eta = 0.5
    * Final CatBoost Model - max_depth = 5 and learning_rate = 0.5

* **Selecting the Best Model** - Once final models were built next step was choosing the Best Model among Final Decision tree, Random forest and XGBoost for Classification models. This was done by evaluating each of the final models on the validation set and comparing the AUC scores. 

By doing so, I found that, ***Random Forest for Classification*** model gave the best AUC score on validation set hence, it was selected as the **Best Model for NYC MV Collision Prediction dataset**. 

Thereafter, I used this best model to make predictions on the testing set (unseen data). Here also it performed fairly close to the validation set scores. Finally, this best model was saved as a Python script and used for further deployment as a web service.



## Exporting Best Model to Python script:

The code for best model i.e., Random Forest for Classification was first saved as a Python Notebook [Capstone_Final_Model_Code.ipynb](https://github.com/sukritishuk/ML_ZoomCamp_Capstone_Project/blob/main/Capstone_Final_Model_Code.ipynb) then saved as a Python script [Capstone_Final_Model_train.py](https://github.com/sukritishuk/ML_ZoomCamp_Capstone_Project/blob/main/Capstone_Final_Model_train.py) for further deployment as a web service.

Name of Python script used - [Capstone_Final_Model_train.py](https://github.com/sukritishuk/ML_ZoomCamp_Capstone_Project/blob/main/Capstone_Final_Model_train.py)

Above script would be used for training the final model and further deployment in a server. 

**Saving and Loading the Collision Injury Prediction Model** - 
* It contains all the necessary libraries & Python packages for the model like pandas, numpy, scikit-learn, seaborn etc. It contains the parameters used for training the model. It also has steps about data preparation, data cleaning and formatting like the once we used in Python Notebook for Kaggle dataset. Then it lists the steps to create a validation framework (splitting dataset into 3 subsets, identifying feature matrix and target variables etc.). Thereafter, it performs one-hot encoding using DictVectorizer on data subsets, trains on training or validation subsets and finally lists steps for making predictions. It also performs KFold Cross-Validation on subsets before making predictions.

* After training, validation and making model ready for predictions it saves the model to a binary file using the **Pickle** library. This enables, to use the model in future without training and evaluating the code. Here, I have used pickle to make a binary file named ***model.bin***. It contains the one-hot encoder (DictVectorizer) and the model details as an array in it.

![image](https://user-images.githubusercontent.com/50409210/139581819-0fe4351e-f48f-4c2c-910d-4945506bf1ba.png)

With unpacking the model and the DictVectorizer here, I would be able to predict the collision injuries for any new input values (sample_collided_person) or unseen data without training a new model, just by re-running the code.



## Putting the Model into a Web Service and Local Deployment using Docker: 

**Creating a Web service for Model using Flask** - 

Name of Python script used - [Capstone_Final_Model_predict.py](https://github.com/sukritishuk/ML_ZoomCamp_Capstone_Project/blob/main/Capstone_Final_Model_predict.py)

* Here I use the **Flask** library in Python to create a web service. The script used here would be implementing the functionality of prediction to our collision_injury web service and would be making it usable in a development environment. To use the model without running the code, I firstly opened and loaded the saved binary file as shown below.

![image](https://user-images.githubusercontent.com/50409210/139581873-4b6058f3-05e6-405d-b2c0-cb0f6f4aa74f.png)

* Finally a function was used for creating the web service. Now, we can run this code to post a new person's data and see the response of our model.

![image](https://user-images.githubusercontent.com/50409210/145710418-275fca66-a4e6-4142-8578-339a85d25126.png)

The details of a new 'sample collided person' are provided in JSON format. These details are sent as a POST request to the web service. The web service sends back a response in JSON format which is converted back into a Python dictionary. Finally, a response message is printed based on the collision_injury decision provided by the model (threshold as 0.55 for collision_injury decision) for the new person.


Name of Python script used - [Capstone_Final_Model_predict_test.py](https://github.com/sukritishuk/ML_ZoomCamp_Capstone_Project/blob/main/Capstone_Final_Model_predict_test.py)

![image](https://user-images.githubusercontent.com/50409210/145710601-a96c9280-dffe-4be8-9c40-b18fc90f0f1f.png)

As shown above, I made a simple web server that predicts the collision_injury for every new collided person. When I ran the app I got a warning (see below) that this server is not a WGSI server, hence not suitable for production environmnets. 

![image](https://user-images.githubusercontent.com/50409210/145714366-5fe4a12f-8c50-49c6-a66f-57cb1d00b7aa.png)

To fix this issue for my Windows machine, I used the library **waitress** to run the waitress WSGI server. This fix helped me make a production server that predicts the collision_injury for every new collided person.



**Creating a Python virtual environment using Pipenv** - 

Names of files used - [Pipfile](https://github.com/sukritishuk/ML_ZoomCamp_Capstone_Project/blob/main/Pipfile) and [Pipfile.lock](https://github.com/sukritishuk/ML_ZoomCamp_Capstone_Project/blob/main/Pipfile.lock)

Virtual environments can help solve library version conflictions in each machine and manage the dependencies for production environments. I used the **Pipenv** library to create a virtual environment for my Collision Injury Prediction project. 

This was done using the following steps:-
* Firstly, I installed pipenv library using *pip install pipenv*. 
* Then, I installed all the necessary libraries for my project in the new virtual environment like numpy, flask, pandas (also specifying exact versions in some cases) using pipenv command like, *pipenv install numpy sklearn==0.24.1 flask*. 

![image](https://user-images.githubusercontent.com/50409210/145714393-93a513bf-caeb-4007-b9b0-2c16f79d7aac.png)

Pipenv command created two files named Pipfile and Pipfile.lock. Both these files contain library-related details, version names and hash files. (In future, if I want to run the project in another machine, I can easily install the libraries using command *pipenv install*, which would look into Pipfile and Pipfile.lock to install all the relevant libraries for my project).

* After installing the libraries I can run the project in a virtual environment with *pipenv shell* command. This will go to the virtual environment's shell and run any command using the virtual environment's libraries.

* Next I installed and used the libraries such as waitress (like before).



**Environment Management using Containerization in Docker** - 

Name of file used - [Dockerfile](https://github.com/sukritishuk/ML_ZoomCamp_Capstone_Project/blob/main/Dockerfile)

Docker allows to separate or isolate my project from any system machine and enables running it smoothly on any machine using a container. To use Docker after installing Docker Desktop for Windows I had to perform the following steps:- 
* Firstly, I had to build a Docker image. I used a Python Docker image [3.8.12-slim](https://github.com/docker-library/python/blob/9242c448c7e50d5671e53a393fc2c464683f35dd/3.8/bullseye/slim/Dockerfile) from the [Docker website](https://hub.docker.com/_/python). 

This Docker image file would have all the dependencies for my project. 

![image](https://user-images.githubusercontent.com/50409210/145710911-37653ec4-a99e-4a70-afba-380f386b0f17.png)

* After creating the Dockerfile and writing the settings in it (as shown above), I built it and specified the **tag name** *capstone_collision-test* for this Dockerfile using the command - *docker build -t capstone_collision-test .*
* Then, I ran the image built and launched waitress service using command - *docker run -it --rm --entrypoint=bash capstone_collision-test*
* Thereafter, I mapped the port 5000 of the Docker to 5000 port of my machine for successful run of project app using Docker container with the command - *docker run -it --rm -p 5000:5000 capstone_collision-test*
 
 This finally deployed my collision_injury prediction app inside a Docker container.



## Instructions for Local Deployment of Collision Injury Prediction Service:


In this section I have summarized the steps for converting my Best model (**Random Forest for Classification**) into a script and deploying it locally as an app using Docker.

Below are the steps in this regard:-
* Changing the directory to the desired one using the command prompt terminal.
* Running the train script [Capstone_Final_Model_train.py](https://github.com/sukritishuk/ML_ZoomCamp_Capstone_Project/blob/main/Capstone_Final_Model_train.py) used for training the best model using command ----> *python Capstone_Final_Model_train.py* 

It resulted in the below output.

![image](https://user-images.githubusercontent.com/50409210/145714322-25d5b772-1d32-4948-97d9-f281d309e30f.png)

* Installing flask library using command -----> *pip install flask*
* Running the predict script [Capstone_Final_Model_predict.py](https://github.com/sukritishuk/ML_ZoomCamp_Capstone_Project/blob/main/Capstone_Final_Model_predict.py) used for loading the best model using command ----> *python Capstone_Final_Model_predict.py*
* Installing waitress library (for Windows) using command ------> *pip install waitress*
* Telling waitress service where the collision_injury predict app is using command ----> *waitress-serve --listen=127.0.0.1:5000 Capstone_Final_Model_predict:app*
* In a new cmd terminal, running the script [Capstone_Final_Model_predict_test.py](https://github.com/sukritishuk/ML_ZoomCamp_Capstone_Project/blob/main/Capstone_Final_Model_predict_test.py) with new sample collided person's details for testing the best model after deployment on new unseen data using command ------> *python Capstone_Final_Model_predict_test.py*

  This would give the collision injury prediction and probability of injury for the new collided person (unseen data) as input.

* Installing the pipenv library for creating virtual environment using command -----> *pip install pipenv* 
* Installing other dependent libraries for collision injury prediction model using command -----> *pipenv install numpy scikit-learn==0.24.2 flask pandas requests*
* Installing python 3.8 version to match python version in the docker image and that in Pipfile using command -----> *pipenv install --python 3.8*
  
  This would update our [Pipfile](https://github.com/sukritishuk/ML_ZoomCamp_Capstone_Project/blob/main/Pipfile) and [Pipfile.lock](https://github.com/sukritishuk/ML_ZoomCamp_Capstone_Project/blob/main/Pipfile.lock) with all requisite library details.
  
* Next, getting into the newly created virtual environment using command ----> *pipenv shell*
* Now running the script with sample collided person details using command ----> *python Capstone_Final_Model_predict_test.py*

  This launches pipenv shell first then runs waitress service. It results with injury predictions and probabilities from our model.

* Changing the directory to the desired one using a new cmd terminal. Downloading a desired Docker Python image [3.8.12-slim](https://github.com/docker-library/python/blob/9242c448c7e50d5671e53a393fc2c464683f35dd/3.8/bullseye/slim/Dockerfile) using command ----> *docker run -it --rm   python:3.8.12-slim*
* Getting into this image using command ----> *docker run -it --rm --entrypoint=bash python:3.8.12-slim*
* Building the Docker image with **tag** *capstone_collision-test* (while still within the virtual env shell) using command -----> *docker build -t capstone_collision-test .*
* Running the image once built using command ----> *docker run -it --rm --entrypoint=bash capstone_collision-test*
  
* This brings us inside the app terminal where we get a list of files in app terminal using command -----> *ls*
* Then launching the waitress service within app terminal using commnad -----> *waitress-serve --listen=127.0.0.1:5000 Capstone_Final_Model_predict:app*
* Thereafter, mapping the port in our container (5000) to the port in our host machine (5000) using command ----> *docker run -it --rm -p 5000:5000 capstone_collision-test*
  
  Both the above steps would launch our waitress service giving below output:-
  
  ![image](https://user-images.githubusercontent.com/50409210/145713912-66af381f-a6bd-4126-bb8a-ba1c4d38a103.png)
  
* Then in fresh cmd terminal changing the directory to the desired one and getting into the virtual environment created earlier using command ----> *pipenv shell*
* Finally, running the script with sample collided person's details using command ------> *python Capstone_Final_Model_predict_test.py*

This gives the collision injury predictions and probabilities from our model for the new sample collided person as input. 

![image](https://user-images.githubusercontent.com/50409210/145714109-c2ce86c4-8593-48a9-85af-10488b5a549d.png)

At last our Collision Injury Prediction service has been deployed locally using a Docker container.




## Cloud Deployment of Collision Injury Prediction Service using AWS Elastic Beanstalk:

Once my Collision Injury prediction service was deloyed locally I also tried to deploy it to the cloud using AWS Elastic Beanstalk. This was done using a special utility **Elastic Beanstalk command-line interface (CLI)**.

Following steps were undertaken in this regard:-

* Firstly, the CLI  was installed and added as a development dependency only for the project using command ---> *pipenv install awsebcli --dev*

![image](https://user-images.githubusercontent.com/50409210/145714436-95954ee6-0ef0-4f71-a59b-88ba56c35621.png)

* Then, I entered the virtual environment for this project using command -----> *pipenv shell* 
* Here, I checked the version for Elastic Beanstalk available using CLI using command -----> *eb --version*
* Next, I initialized the Docker-based **collision-prediction-serving** platform using command -----> *eb init -p docker -r eu-west-1 collision-prediction-serving*

![image](https://user-images.githubusercontent.com/50409210/145714462-c6987843-fc33-4026-aa59-f85a131549c0.png)

Output on AWS Console for Elastic Beanstalk - 

![image](https://user-images.githubusercontent.com/50409210/145714465-0b38d972-ae74-4ecc-b7a9-a59ac577dd69.png)

* Now I first tested my application locally by specifying the port 5000 using command -----> *eb local run --port 5000*. 

It will first build an image and then run the container. 

* Then I made predictions about new 'sample collided person' using the command as earlier -----> *python Capstone_Final_Model_predict_test.py*
* After this, I created the **collision-prediction-serving environment** in AWS which sets up **EC2 instance**, applying auto-scaling rules using command -----> *eb create collision-prediction-serving-env*

It creates the application and launches the collision-prediction-serving environment. It also generates a **public url** which can be used to reach the application and make predictions.

![image](https://user-images.githubusercontent.com/50409210/145711772-bc0e137e-b832-436f-9ce3-461958c16eed.png)

* Finally, I could test the cloud service to make predictions about new 'sample collided person' as input using commnad -----> *python Capstone_Final_Model_predict_test.py* 

It resulted in giving us collision injury predictions and probabilities from our model for the new sample collided person (as shown below).

![image](https://user-images.githubusercontent.com/50409210/145711847-ba62007d-2f70-4e35-901a-9d118d4f817d.png)

Thus, our collision prediction service was deployed inside a container on AWS Elastic Beanstalk (as shown below). To reach it, we could use its public URL.

![image](https://user-images.githubusercontent.com/50409210/145711881-5c48c2eb-addc-4715-ad75-fe98b69207cb.png)



