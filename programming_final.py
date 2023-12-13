#!/usr/bin/env python
# coding: utf-8


import streamlit as st 

import pandas as pd
import numpy as np
import altair as alt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix





s = pd.read_csv("social_media_usage.csv")





#s


def clean_sm (x):
    x = np.where(x==1,1,0)

    return x



sample_data = {'Column_1': [1,2,3],
             'Column_2': [1,4,7]}
df = pd.DataFrame(sample_data)





print(df)





print(clean_sm(df))





ss = pd.DataFrame({
    "sm_li":clean_sm(s["web1h"]),
    "income":np.where(s["income"] >9, np.nan,
                          s["income"]),
    "education":np.where(s["educ2"] > 8, np.nan,s["educ2"]),
    "par":np.where(s["par"] == 1, 1,0),
    "marital":np.where(s["marital"] == 1, 1, 0),
    "gender":np.where(s["gender"] == 2, 1,0),#Female=1,non-female = 0
    "age": np.where(s["age"] >98, np.nan,
                    s["age"])})
ss = ss.dropna()

ss["income"]=ss["income"].astype(int)
ss["age"]= ss["age"].astype(int)
ss["education"]=ss["education"].astype(int)

#ss.head(10)




#Create plots to explore relationships in the data
#group_ss = ss.groupby(["gender","age"], as_index=False)["sm_li"].mean()
#Plot the age, gender, and mean LinkedIn usage
#alt.Chart(group_ss).mark_bar().encode(
   # x="age:Q",
   # y="sm_li:Q",
   # color="gender:N")



#Creates target(y) and feature variables(x)
y = ss["sm_li"]
x = ss[["income","education","par","marital","gender","age"]]


#Split the data into train and test sets

x_train,x_test,y_train,y_test = train_test_split(x,
                                                y,
                                                stratify=y,
                                                test_size=0.2,
                                                random_state=721)
#x_train contains the explanatory variables we will use in our train dataset, it contains about 80% of the data
#x_test contains 20% of the data and the features used to test the unseen data and evaluate the performance
#y_train contains the response variable sm_li we will use in our train dataset, it contains about 80% of the data
#y_test contains 20% of the unseen response test data we will use to evaluate the performance of our model 


# In[172]:


#Initiate the algorithm 
lr = LogisticRegression()
#Set class weight to balanced
LogisticRegression (class_weight = 'balanced')
#Fit the training data to the logistic regression algorithm
fit_model=lr.fit(x_train,y_train)
#print(fit_model)




### 7 Evaluate model performance

y_pred = lr.predict(x_test)

#confusion_matrix(y_test, y_pred)

pd.DataFrame(confusion_matrix(y_test, y_pred),
            columns=["Predicted negative", "Predicted positive"],
            index=["Actual negative","Actual positive"]).style.background_gradient(cmap="PiYG")
#+ **Accuracy**
#+ Other evaluation metrics:
# + **Recall**: Recall is calculated as $\frac{TP}{(TP+FN)}$ and is important when the goal is to minimze the chance of missing positive cases. E.g. fraud
# + **Precision**: calculated as $\frac{TP}{(TP+FP)}$ and is important when the goal is to minimize incorrectly predicting positive cases. E.g. cancer screening
# + **F1 score**: F1 score is the weighted average of recall and precision calculated as $2\times\frac{(precision x recall)}{(precision+recall)}$


#Recall: 
#36/(36+48) 




#Precision:
#36/(36+19) 

#F1 score: 
#2*(0.6545454545454545*0.42857142857142855)/(0.6545454545454545+0.42857142857142855)



print(classification_report(y_test, y_pred))


# In[192]:
st.title("LinkedIn User Prediction App")

#


#newdata["y_test"] = lr.predict(newdata)


#newdata

new_person = [7,6,1,1,0,30]


# In[203]:


predicted_class = lr.predict([new_person])




probability = lr.predict_proba([new_person])




print(f"predicted class:{predicted_class[0]}")




print(f"Probability that this person is a LinkedIn: {[probability[0][1]]}")

#Code to create application on streamlit

income = st.slider(label="Income level (1: less than 10k - 9: over $150k)", 
          min_value=1,
           max_value=9,
         value=5)
education = st.slider(label = "College degree (0=no,1=yes)",
                          min_value= 0,
                          max_value=1,
                          value = 1)
par = st.slider(label = "Parent of child under 18? (0=no,1=yes)",
                          min_value= 0,
                          max_value=1,
                          value= 0)
marital = st.slider(label = "Married? (0=no,1=yes)",
                          min_value= 0,
                          max_value=1,
                          value = 0)
gender = st.slider(label = "Gender (0=male,1=female)",
                          min_value= 0,
                          max_value=1,
                          value = 0)
age = st.slider(label = "Age? (18-97)",
                          min_value= 18,
                          max_value=97,
                          value = 50
                          )
new_person = [income, education, par, marital, gender, age]

# Load the trained model
lr = LogisticRegression(class_weight="balanced")
x = ss[["income", "education", "par", "marital", "gender", "age"]]
y = ss["sm_li"]
x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=721)
fit_model = lr.fit(x_train, y_train)

new_person = [income, education, par, marital, gender, age]

# Make prediction for the new person
predicted_class = lr.predict([new_person])
probability = lr.predict_proba([new_person])

# Display results
st.subheader("Prediction:")
st.write("The person is classified as a LinkedIn user." if predicted_class[0] == 1 else "The person is not classified as a LinkedIn user.")

st.subheader("Probability:")
st.write(f"The probability of being a LinkedIn user is: {probability[0][1]:.2%}")
