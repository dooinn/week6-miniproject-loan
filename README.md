# Loan Eligibility

## Context

Dream Housing Finance company deals in all kinds of home loans. They have presence across all urban, semi urban and rural areas. Customer first applies for home loan and after that company validates the customer eligibility for loan.
Company wants to automate the loan eligibility process (real time) based on customer detail provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To automate this process, they have provided a dataset to identify the customers segments that are eligible for loan amount so that they can specifically target these 



## Feature Description

- **Loan_ID**:	Unique Loan ID
- **Gender**:	Male/ Female
- **Married**:	Applicant married (Y/N)
- **Dependents**:	Number of dependents
- **Education**:	Applicant Education (Graduate/ Under Graduate)
- **Self_Employed**:	Self employed (Y/N)
- **Applicant Income**:	Applicant income
- **Coapplicant Income**:	Coapplicant income
- **Loan Amount**:	Loan amount in thousands
- **Loan_Amount_Term**:	Term of loan in months
- **Credit_History**:	credit history meets guidelines
- **Property_Area**:	Urban/ Semi Urban/ Rural
- **Loan_Status**:	(Target) Loan approved (Y/N)


## Project Goal
Using the provided dataset, our study will aim to determine which particular customer profiles are more likely to receive loan approval. Subsequently, we will develop a machine learning model using Logistic Regression, which will automate the binary classification of loan status (approved/denied).

## Project Strucuture
1. Import libraries & Load Dataset
2. Overview of Data
3. Data Cleansing
    - 3-1. Remove Duplicates
    - 3-2. Standardization of Headers
    - 3-3. Handling Null Values
4. Exploratory Data Analysis (EDA)
5. Logistic Regression Analysis
    - 5-1. Data Processing
    - 5-2. Data Encoding
    - 5-3. Splitting the Data and Fitting the Model
    - 5-4. Model Building - Logistic Regression
    - 5-5. Model Validation - Confusion Matrix



<img src="images/1.jpg" width = 300 alt="Alt text that describes the graphic" title="Title text" />


