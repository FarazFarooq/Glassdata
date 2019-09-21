#!/usr/bin/env python
# coding: utf-8

# # Beginning of Project for Predicting Glass types

# We will be looking at the glass dataset from the UCI Machine learning repository.The goal of this Project is to build a model that successfully predicts the type of glass based on the evidence of it remaining at the scene. https://archive.ics.uci.edu/ml/datasets/glass+identification
# 
# The Attributes are as follows:
# 
# Id number: 1 to 214
# RI: refractive index
# Na: Sodium (unit measurement: weight percent in corresponding oxide, as are attributes 4-10)
# Mg: Magnesium
# Al: Aluminum
# Si: Silicon
# K: Potassium
# Ca: Calcium
# Ba: Barium
# Fe: Iron
# Type of glass: (class attribute) -- 1 building_windows_float_processed -- 2 building_windows_non_float_processed -- 3 vehicle_windows_float_processed -- 4 vehicle_windows_non_float_processed (none in this database) -- 5 containers -- 6 tableware -- 7 headlamps
# Goal: To build a model that accurately classifies types of glass characteristics into the type of glass they came from.

# # # Importing the necessary libraries and setting up the directory.
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


import os
path = "C:/Users/Faraz/Downloads/PythonProjects"
os.chdir(path)
os.getcwd()


# # # Importing the Data

# In[3]:


glassdata = pd.read_csv("glass.csv")
glassdata.head()


# # # Initial Exploration

# In[4]:


print(glassdata.isnull())  # Confirming that there are no missing values


# In[5]:


glassdata.describe()


# In[6]:


glassdata.shape


# In[7]:


glassdata.dtypes


# # # Exploratory Data Analysis

# In[8]:


glassdata.hist(figsize= (10,10))  # Understanding where most of the values line up 


# Looking at the histograms, we clearly see none of these exhibit a standard normal distribution. K and Ba are the elements that are the most off.

# In[9]:


boxplot = sns.boxplot(data=glassdata, orient="h", palette="Set2")


# It seems most of the values for the materials are consistent, however NA(Sodium has a lot of outliers). Silicon is also very much high in terms of amount but also straying away from the rest of the materials and values. Sodium might be a high amount in Glass?

# In[10]:


corr = glassdata.corr()
corr


# In[11]:


# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# Looking at the correlation we see that the elements that are the most correlated with the type of the glass are from lowest to highest, Na(Positive), Al(Positive), and Mg(Negative) These 3 all also had data points with a right skew in the histogram. Something else to note Ca has a high correlation with the reserved index.

# In[12]:


sns.boxplot('Type', 'Fe', data=glassdata)
# Iron mainly present in types 1-3 and outliers in type 7


# In[13]:


import seaborn as sns
sns.boxplot(x=glassdata['Fe'])


# In[14]:


sns.boxplot('Type', 'Mg', data=glassdata)  # Outliers in glass type 2 and 7


# In[15]:


sns.boxplot('Type', 'Na', data=glassdata)  # Main outliers are present in glass type 2


# In[16]:


sns.boxplot('Type', 'Al', data=glassdata)  # Outliers present in types 1 and 2


# In[17]:


sns.boxplot('Type', 'Ba', data=glassdata)  # Very few outliers but it seems Barium only exists in glass type 7


# In[18]:


sns.boxplot('Type', 'Ca', data=glassdata)  # Outliers present for glass types 2 and 7


# In[19]:


sns.boxplot('Type', 'K', data=glassdata)  # Outliers mainly present for glass type 7


# In[20]:


sns.boxplot('Type', 'RI', data=glassdata)  # Outliers present when predicting glass types 1,2,3, and 7


# In[21]:


sns.boxplot('Type', 'Si', data=glassdata)  # Outliers present when predicting glass types 2 and 7


# Given that type 2 had the highest number of outliers, it is not surprising because type 2 was also where most of the observations came from. Because of this, the next goal is to remove the outliers we have observed. As it stands removing outliers for Na and Al.

# In[22]:


glassdata.boxplot(return_type='dict')
plt.plot()


# # # Model Selection (Random Forest)

# In[23]:


glassdata.target = glassdata['Type']


# In[24]:


from sklearn.model_selection import train_test_split  # Importing library for split
xTrain, xTest, yTrain, yTest = train_test_split(glassdata.drop(['Type'], axis=1), 
glassdata.target, random_state=0, test_size=0.25)


# In[25]:


from sklearn.ensemble import RandomForestClassifier
# Create a Gaussian Classifier
clf = RandomForestClassifier(n_estimators=100)
# Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(xTrain, yTrain)


# In[26]:


y_pred = clf.predict(xTest)


# In[27]:


#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(yTest, y_pred))


# In[28]:


feature_importances = pd.DataFrame(clf.feature_importances_, 
                    index = xTrain.columns,
                    columns=['importance']).sort_values('importance',ascending=False)


# In[29]:


feature_importances


# Looking at feature importance, we see that Aluminum is the highest and contributes the most to the Random Forest Model, next being Magnesium. This makes sense as these were 2 of the 3 highly correlated values when looking at type that we observed earlier. NA which was the third is 5th on this list meaning it does not contribute as much to the random forest model as we thought. We do know that there is a healthy sense of bias in using feature importance, it might be that Al had the most number of complete observations?
# 
# This also hints that removing outliers especially from Al and Na should contribute to a positive effect on the model.

# # # Tuning the Model

# ###  Outlier Removal

# In[30]:


from scipy import stats
import numpy as np
z = np.abs(stats.zscore(glassdata))
print(z)


# In[31]:


threshold = 3
print(np.where(z > 3))


# In[32]:


glassdatanoout = glassdata[(z < 3).all(axis=1)]


# In[33]:


glassdatanoout.shape


# In[34]:


glassdata.shape


# In[35]:


glassdatanoout.target = glassdatanoout['Type'] 


# ## Re-Training after Ouliers have been removed
# 

# In[36]:


from sklearn.model_selection import train_test_split  # Importing library for split
xTrainnoout, xTestnoout, yTrainnoout, yTestnoout = train_test_split(glassdatanoout.drop(['Type'], axis=1), 
glassdatanoout.target, random_state=0, test_size=0.25)


# In[37]:


clf.fit(xTrainnoout,yTrainnoout)


# In[38]:


y_prednoout = clf.predict(xTestnoout)


# In[39]:


# Model Accuracy, how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(yTestnoout, y_prednoout))


# In[40]:


clf.fit(xTrainnoout,yTrainnoout).score(xTrainnoout, yTrainnoout)


# In[41]:


feature_importances = pd.DataFrame(clf.feature_importances_,
                            index = xTrainnoout.columns,
                            columns=['importance']).sort_values('importance', ascending=False)


# In[42]:


feature_importances


# In[43]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(yTest, y_pred)
print(confusion_matrix)


# Next we will use Grid Search CV to sample from the best values provided by the random forest process. This will help to see if we can improve the accuracy further.

# ## Hyper Parameter Tuning with Grid Search CV

# In[44]:


from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


# In[45]:


param_grid = {
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [4, 5, 6, 7, 8],
    'criterion': ['gini', 'entropy']
}


# In[46]:


CV_rfc = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10)
CV_rfc.fit(xTrainnoout, yTrainnoout)


# In[47]:


CV_rfc.best_params_


# In[48]:


rfc1 = RandomForestClassifier(random_state=42, max_features='auto', n_estimators=200, max_depth=8, criterion='gini')


# In[49]:


rfc1.fit(xTrainnoout, yTrainnoout)


# In[50]:


pred = rfc1.predict(xTestnoout)
print("Accuracy for Random Forest on CV data: ", metrics.accuracy_score(yTestnoout, pred))


# In[51]:


rfc1.fit(xTrainnoout, yTrainnoout).score(xTrainnoout, yTrainnoout)


# The GridSearch CV has increased the Random Forest by 2% thus improving the accuracy of the prediction. It does this by finding the best number of forests (number of trees) and splits the data across it to achieve a prediction.

# # # SVM Model

# We know for random forest, the more trees there are the better the model. But we need to find the optimal amount as anything above it could lead to issues with the model such as overfitting. Perhaps SVM is a good one?

# In[52]:


from sklearn import svm 
clf = svm.SVC(kernel='rbf') 
clf.fit(xTrainnoout, yTrainnoout)
y_pred = clf.predict(xTestnoout)


# In[53]:


# Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Model Accuracy: how often is the classifier correct?
print("Accuracy:", metrics.accuracy_score(yTestnoout, y_pred))


# RBF Kernel gave us the best accuracy and that might be due to the fact that with all the different types of classes, we are working with a non-linear plane. Let's try to visualize below

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




