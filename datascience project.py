#!/usr/bin/env python
# coding: utf-8

# In[42]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[43]:


data_set = pd.read_csv("newproject.csv")
data_set


# In[30]:


meta_data=pd.read_csv("newproject.csv")


# In[31]:


meta_data


# In[32]:


data_set.shape


# In[33]:


data_set.head().T  #to transform the data so that it is easily viewable


# In[45]:


x=pd.DataFrame(data_set['Bank Balance'])
y=pd.DataFrame(data_set['Defaulted'])
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(criterion='mse',random_state=100,max_depth=4,min_samples_leaf=1)
reg=regressor.fit(x_train, y_train) 
from sklearn.tree import export_graphviz
from sklearn import tree
import pydotplus
export_graphviz(regressor,out_file='reg_tree.dot')
y_pred=regressor.predict(x_test)
print(y_pred)


# In[47]:


plt.scatter(x=data_set['Bank Balance'],y=data_set['Defaulted'],color='red')
plt.xlabel('BANK BALANCE OF CUSTOMER')
plt.ylabel('DEFAULTED CUSTOMER')
plt.plot(x,y)


# In[36]:


tree.plot_tree(reg)


# In[49]:


plt.figure(figsize=(10,10))
plt.hist(data_set['Bank Balance'],bins=20)
plt.hist(data_set['Defaulted'],bins=10)


# In[38]:


data_set= np.random.random(( 12 , 12 ))
plt.imshow( data_set , cmap = 'viridis', interpolation = 'nearest' ) 
plt.title( "2-D Heat Map" )
plt.show()


# In[12]:


#data_set.groupby("CREDIT SCORE")("BANK LOAN").describe()


# In[52]:


from sklearn.linear_model import LogisticRegression  
classifier= LogisticRegression(random_state=0)  
classifier.fit(x_train, y_train)  

y_pred= classifier.predict(x_test)  
y_pred


# In[22]:


full_health_data = pd.read_csv("newproject.csv", header=0, sep=",")
full_health_data


# In[65]:


data_set.shape
from sklearn.metrics import confusion_matrix  
d=data_set.head(2000)
cm=confusion_matrix(d,y_pred)
cm


# In[51]:


import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

full_health_data = pd.read_csv("newproject.csv", header=0, sep=",")
model = smf.ols("BankBalance ~ Defaulted", data = full_health_data)
results = model.fit()
print(results.summary())


# In[ ]:





# In[ ]:





# In[ ]:




