# Credit Approval Prediction Program
Machine learning prediction model combined with simple data structure program about credit approval



## A. Background
Credit Approval Prediction Program is a simple program combining machine learning model prediction and Object-Oriented Programming (OOP) concepts in Python. The main goal of this project is to implement the best machine learning model that could predict credit approval in a simple prediction system. By providing this credit approval prediction program, users can easily make predictions of credit approval.



## B. Objectives
This project is built to implement data wrangling, feature engineering, machine learning classification models, and Object-Oriented Programming (OOP) concepts in Python.



## C. Dataset
The dataset is obtained from the UC Irvine Machine Learning Repository named [__Credit Approval__](https://archive.ics.uci.edu/dataset/27/credit+approval). This project uses the file [`crx.data`](). Based on [the analysis regarding the dataset by Ryan Kuhn](https://rstudio-pubs-static.s3.amazonaws.com/73039_9946de135c0a49daa7a0a9eda4a67a72.html#exploratory-analysis-and-data-transformations), the features on the dataset includes:
- `0` = `Gender` : where 'a' represents male and 'b' represents female
- `1` = `Age` : the applicant's age
- `2` = `Debt` : the applicant's debt
- `3` = `Married` : marital status of the applicant
- `4` = `BankCustomer` : applicant's bank which has value 'g', 'p', and 'gg'
- `5` = `EducationLevel` : applicant's educational level which has 'w', 'q', 'm', 'r', 'cc', 'k', 'c', 'd', 'x', 'i', 'e', 'aa', 'ff', and 'j'
- `6` = `Ethnicity` : ethnicity of the applicant which has value 'v', 'h', 'bb', 'ff', 'j', 'z', 'o', 'dd', and 'n'
- `7` = `YearsEmployed` : years of employment
- `8` = `PriorDefault` : prior defaulting on credit where 't' represents that the applicant has previously defaulted on credit while 'f' represents the applicant has no record regarding defaulting
- `9` = `Employed` : employment status where 't' represents that the applicant is employed while 'f' represents the applicant is unemployed
- `10` = `CreditScore` : numerical representation of an applicant's creditworthiness,
- `11` = `DriverLicense` : ownership of driver's license where 't' represents that the person has driver's license while 'f' represents the person has no driver's license
- `12` = `Citizen` : the person's citizenship which has value 'g', 's', and 'p'. Those value probably represents country name's
- `13` = `ZipCode` : the zip code of where the person resides in
- `14` = `Income` : the person's income 
- `15` = `Approved` : where '+' represents approved while '-' represents disapproved

The feature `15` (`Approved`) acted as the label for this classification case



## D. Requirements & Program Flow
This program contains one analysis notebook and four modules for the prediction program:
1. Notebook [__`model_training_and_analysis.ipynb`__](https://github.com/novitaaryanti/credit_approval_prediction_project/blob/main/model_training_and_analysis.ipynb) as the notebook to analyse the dataset and search for the best model to predict the credit approval with the highest accuracy and F1-score. This notebook helps in decision-making for building the program.
2. Module [__`main.py`__](https://github.com/novitaaryanti/credit_approval_prediction_project/blob/main/main.py) includes function `main()` as the main point of running the program and function `menu()` as the credit approval menu.
3. Module [__`credit_application.py`__](https://github.com/novitaaryanti/credit_approval_prediction_project/blob/main/credit_application.py) includes all functions to do the task of credit approval menu option and also the class of applicant object:
   - Function `get_idx_ori_encode_dict()`: obtain the index of encoded feature value for new applicant's data
   - Method `__init__()` in class `Application`: make applicant list in the class Application
   - Function `add_applicant()` in class `Application`: add new applicant to the applicant list
   - Function `display_applicant()` in class `Application`: display the DataFrame of applicant details
   - Function `delete_applicant()` in class `Application`: remove specific applicant based on the given name
   - Function `reset_applicant()` in class `Application`: remove all applicants from the applicant list
   - Function `show_pred_res()` in class `Application`: show the prediction result of applicant's credit approval
   - Function `get_pred_res()` in class `Application`: do prediction using the trained model on new applicant's details
4. Module [__`training.py`__](https://github.com/novitaaryanti/credit_approval_prediction_project/blob/main/training.py) includes all functions to do model training:
   - Function `feature_engineering()`: do feature engineering for the dataset
   - Function `model_training()`: do model training using the Random Forest model. This model is chosen based on the analysis in the notebook `model_training_and_analysis.ipynb`
   - Function `model_evaluation()`: evaluate the model on the test set
5. Module [__`data_wrangling.py`__](https://github.com/novitaaryanti/credit_approval_prediction_project/blob/main/data_wrangling.py) includes all functions to do data wrangling:
   - Function `feature_engineering()`: open dataset and select necessary features based on analysis done in `model_training_and_analysis.ipynb`
   - Function `missing_data_handling()`: handle missing values which are indicated with '?' (based on analysis done in `model_training_and_analysis.ipynb`)
   - Function `incosistent_format_handling()`: change the data type for dataset features based on analysis is done in `model_training_and_analysis.ipynb`
   - Function `get_dict_unique_val()`: get unique value before and after label encoding
   - Function `data_encoding()`: do categorical features encoding (manually or automatically using LabelEncoder())
   - Function `outlier_imputation()`: handle outlier based on the analysis done in `model_training_and_analysis.ipynb`.


### 1. Analyse the dataset by doing data wrangling
After analysing the feature in the dataset, missing data and inconsistent format of the data type are handled. After that, do feature encoding for categorical features. From the dataset, it is shown that the dataset for the label is slightly imbalanced with the proportion of 54.67% for label `0` (disapproved) and the remaining 45.33% for label `1` (approved). For the trained features, the top 10 independent features were selected based on the correlation to the label. Those 10 features (unsorted) are:
- `1` (Age)
- `2` (Debt)
- `3` (Married)
- `4` (BankCustomer)
- `5` (EducationalLevel)
- `7` (YearsEmployed)
- `8` (PriorDefault)
- `9` (Employed)
- `10` (CreditScore)
- `14` (Income)

The top 3 independent features based on the correlation are `8` (PriorDefault) with 0.74 correlation, followed by `9` (Employed) with 0.45 correlation, and `10` (CreditScore) with 0.41 correlation. The correlation on the other independent features is less the 0.4 which varies around 0.1 - 0.3. Next, the outlier is handled. Outlier appears on several features, which are `1` (Age), `2` (Debt), `7` (YearsEmployed), `10` (CreditScore), and `14` (Income). The outliers in those four features are handled by setting the outlier value as the upper extreme or lower extreme in order to maintain the information of the data. After that, the data is split into train and test set with ratio train:test = 80:20. Lastly, the feature set (X) is scaled using MinMaxScaler.


### 2. Train the dataset and evaluate the model
For the model, this project uses three models, which are Logistic Regression, Random Forest, and XGBoost. Using GridSearchCV with 5-fold cross-validation, the best parameters obtained from those models for this dataset are:
1. __Logistic Regression__
  - `C`: 10
  - `tol`: 0.01
  - `max_iter`: 100
  - `penalty`: L1
  - `solver`:liblinear
2. __Random Forest__
  - `criterion`: gini
  - `max_depth`: 30
  - `max_features`: sqrt
  - `min_samples_leaf`: 1
  - `min_samples_split`: 6
  - `n_estimators`: 50
3. __XGBoost__
  - `learning_rate`: 0.3
  - `max_depth`: 10
  - `min_child_weight`: 7 

Predicting the credit approval using the test set, the obtained performance scores are:

| Metrics   	| Logistic Regression 	| Random Forest 	| XGBoost  	|
|-----------	|---------------------	|---------------	|----------	|
| Accuracy  	| 0.839695            	| 0.862595      	| 0.816794 	|
| Precision 	| 0.757576            	| 0.836364      	| 0.771930 	|
| Recall    	| 0.909091            	| 0.836364      	| 0.800000 	|
| F1        	| 0.826446            	| 0.836364      	| 0.785714 	|

From the metrics above, Random Forest can be considered the best model due to the high accuracy and F1 Score. As the data is slightly imbalanced, the F1 Score might be the best consideration for model performance metrics. Based on this result, the model which will be used for the program is Random Forest.


### 3. Add applicant data to the Credit Approval Prediction Program
Users can add several applicants to do predictions at once. For adding applicant data, the user can choose option __1 (Add Applicant Data)__ on the menu which will direct the user to function `add_applicant()` in class `Application`. When adding the applicant details, users will be asked to input several features which are:
- `1` (Age)
- `2` (Debt)
- `3` (Married)
- `4` (BankCustomer)
- `5` (EducationalLevel)
- `7` (YearsEmployed)
- `8` (PriorDefault)
- `9` (Employed)
- `10` (CreditScore)
- `14` (Income)

Users can also remove specific applicants or remove all applicants from the list. To remove a specific applicant, the user can choose option __2 (Remove Data)__ on the menu which will direct the user to function `delete_applicant()` in class `Application`. To remove all applicants, the user can choose option __3 (Remove All Applicant)__ which will direct the user to function `reset_applicant()` in class `Application`.


### 4. Predict the credit approval on the inputted applicant's data
Before predicting the credit approval, the user can see all inputted applicant details on the list by choosing option __4 (Show Credit Applicant List)__ which will direct the user to function `display_applicant()` in class `Application`. To predict the applicant's credit approval, the user can choose option __5 (Predict Credit Approval)__ which will direct the user to function `get_pred_res()` in class `Application`.



## F. Test Cases

### Test 1: Add Applicant
Add new applicant with the name 'Felix'

<img src="https://github.com/novitaaryanti/credit_approval_prediction_project/assets/138101831/6a5ce990-3090-4f44-90cf-4bcb846a343c" width="500"/>
<img src="https://github.com/novitaaryanti/credit_approval_prediction_project/assets/138101831/e6b2fd61-b36d-4094-80e7-676ba8feb0b0" width="500"/>


### Test 2: Remove Applicant
Remove the applicant with the name 'Felix'

<img src="https://github.com/novitaaryanti/credit_approval_prediction_project/assets/138101831/344abea1-84ed-4e4f-97f6-0eb73a763ce3" width="500"/>


### Test 3: Remove All Applicant (Reset)
<img src="https://github.com/novitaaryanti/credit_approval_prediction_project/assets/138101831/dd712d01-802b-4263-9d31-e2cb6c2de7f6" width="500"/>


### Test 4: Show Credit Applicant List
<img src="https://github.com/novitaaryanti/credit_approval_prediction_project/assets/138101831/f6ae6d4a-19fa-45f9-b3e2-d96889da1f15" width="500"/>


### Test 5: Predict Credit Approval
<img src="https://github.com/novitaaryanti/credit_approval_prediction_project/assets/138101831/ed8489aa-79ec-456d-9df2-703dca6ff1f2" width="500"/>



## E. Conclusions
This project covers the minimum requirements of a credit approval prediction program. From the analysis, the best model for doing credit card approval prediction using the best 10 features based on the correlation is __Random Forest__ with accuracy = 0.862595 and F1-score = 0.836364. However, the result might be different as the dataset is randomized before the training.
