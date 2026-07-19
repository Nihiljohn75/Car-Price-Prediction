#!/usr/bin/env python
# coding: utf-8

# # Car Price Prediction using Machine Learning
# 
# ## Author
# Nihil John
# 
# ## Business Problem
# A company wants to estimate the selling price of a used car based on vehicle characteristics.
# 
# ## Objective
# Build a machine learning model that predicts the selling price of a car.
# 
# # 1. Data Cleaning
# 
# # 2. Exploratory Data Analysis
# 
# # 3. Feature Engineering
# 
# # 4. Model Building
# 
# ## 4.1 Linear Regression
# 
# ## 4.2 Random Forest Regressor
# 
# # 5. Model Evaluation
# 
# # 6. Cross Validation
# 
# # 7. Feature Importance
# 
# # 8. Model Saving
# 
# # 9. Conclusion

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
import pickle


# In[2]:


df=pd.read_csv('car_data_augmented.csv')


# In[3]:


df


# # Business problem:
# ## A company wants to estimate the selling price of a used car based on vehicle characteristics.

# ## Objective:
# 
# ### Build a machine learning model that predicts the selling price of a car.

# In[4]:


print(df.shape)



# In[5]:


print(df.info())


# In[6]:


print(df.isnull().sum())


# In[7]:


df.columns


# In[8]:


df.duplicated().sum()


# In[9]:


df=df.drop_duplicates()


# In[10]:


df.duplicated().sum()


# In[11]:


print(df.describe())


# In[12]:


df['Selling_Price'].describe()


# In[13]:


sns.scatterplot(x='Present_Price', y='Selling_Price', data=df)
plt.show()


# In[14]:


df.corr(numeric_only=True)['Selling_Price']


# In[15]:


plt.figure(figsize=(8,5))
sns.boxplot(x='Selling_Price', y='Fuel_Type', data=df)
plt.show()


# In[16]:


sns.boxplot(x='Selling_Price', y='Transmission', data=df)
plt.show()


# In[17]:


sns.boxplot(x='Selling_Price', y='Seller_Type', data=df)
plt.show()


# In[18]:


Current_year = datetime.now().year


# In[19]:


df['Car_age']= Current_year - df['Year']


# In[20]:


sns.scatterplot(x='Car_age', y='Selling_Price', data=df)
plt.show()


# In[21]:


df.corr(numeric_only=True)['Selling_Price']


# In[22]:


df.isnull().sum()


# In[23]:


df.duplicated().sum()


# In[24]:


corr = df.corr(numeric_only=True)
plt.figure(figsize=(8,5))
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.show()


# In[25]:


df.drop('Year', axis=1, inplace=True)


# In[26]:


df.head()


# In[27]:


df.select_dtypes(include='object').columns


# In[28]:


df['Car_Name'].nunique()


# In[29]:


df.drop('Car_Name', axis=1, inplace=True)


# In[30]:


df = pd.get_dummies(
    df,
    columns=['Fuel_Type','Seller_Type','Transmission'],
    drop_first=True)


# In[31]:


df.head()


# In[32]:


df.dtypes


# In[33]:


bool_cols = df.select_dtypes(include='bool').columns
df[bool_cols] = df[bool_cols].astype(int)


# In[34]:


df.dtypes


# In[35]:


df.head()


# In[36]:


x=df.drop('Selling_Price', axis=1)
y=df['Selling_Price']


# In[37]:


x_train, x_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,
    random_state=42
)


# In[38]:


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[39]:


lr = LinearRegression()
lr.fit(x_train, y_train)


# In[40]:


y_pred=lr.predict(x_test)


# In[41]:


print(len(y_pred))
print(len(y_test))


# In[42]:


print("R2 Score:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("RMSE:",
      np.sqrt(mean_squared_error(y_test,y_pred)))


# In[43]:


rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

rf.fit(x_train, y_train)

y_pred_rf = rf.predict(x_test)

print("R2 Score:", r2_score(y_test, y_pred_rf))
print("MAE:", mean_absolute_error(y_test, y_pred_rf))


# In[44]:


result = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred_rf
})

print(result.head(20))


# In[45]:


result.to_csv('prediction_results.csv', index=False)

print("prediction_results.csv created successfully")


# In[46]:


y_pred = np.maximum(y_pred, 0)


# In[47]:


importance = pd.DataFrame({
    'Feature': x_train.columns,
    'Importance': rf.feature_importances_
}).sort_values('Importance', ascending=False)

print(importance)


# In[48]:


pred = rf.predict(x_test)

print("R2:", r2_score(y_test, pred))
print("MAE:", mean_absolute_error(y_test, pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, pred)))


# In[49]:


models = pd.DataFrame({
    'Model':['Linear Regression','Random Forest'],
    'R2':[r2_score(y_test,y_pred),
          r2_score(y_test,y_pred_rf)],
    'MAE':[mean_absolute_error(y_test,y_pred),
           mean_absolute_error(y_test,y_pred_rf)]
})

print(models)


# In[50]:


sample = pd.DataFrame({
    'Present_Price':[6.5],
    'Kms_Driven':[30000],
    'Owner':[0],
    'Car_age':[4],
    'Fuel_Type_Diesel':[0],
    'Fuel_Type_Petrol':[1],
    'Seller_Type_Individual':[1],
    'Transmission_Manual':[1]
})

rf.predict(sample)


# In[51]:


print(len(['Linear Regression','Random Forest']))
print(len([r2_score(y_test,y_pred),
           r2_score(y_test,y_pred_rf)]))
print(len([mean_absolute_error(y_test,y_pred),
           mean_absolute_error(y_test,y_pred_rf)]))


# In[52]:


models = pd.DataFrame({
    'Model':['Linear Regression','Random Forest'],
    'R2':[
        r2_score(y_test,y_pred),
        r2_score(y_test,y_pred_rf)
    ],
    'MAE':[
        mean_absolute_error(y_test,y_pred),
        mean_absolute_error(y_test,y_pred_rf)
    ],
    'RMSE':[
        np.sqrt(mean_squared_error(y_test,y_pred)),
        np.sqrt(mean_squared_error(y_test,y_pred_rf))
    ]
})

print(models)


# In[53]:


sample = pd.DataFrame({
    'Present_Price':[12],
    'Kms_Driven':[3000],
    'Owner':[1],
    'Car_age':[6],
    'Fuel_Type_Diesel':[0],
    'Fuel_Type_Petrol':[1],
    'Seller_Type_Individual':[1],
    'Transmission_Manual':[1]
})

rf.predict(sample)


# In[54]:


with open('car_price_model.pkl', 'wb') as file:
    pickle.dump(rf, file)

print("Model saved successfully")


# In[55]:


residuals = y_test - pred

plt.figure(figsize=(8,5))
plt.scatter(pred,residuals)
plt.axhline(y=0)
plt.xlabel("Predicted Price")
plt.ylabel("Residuals")
plt.title("Residual Analysis")
plt.show()


# In[56]:


from sklearn.model_selection import KFold

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

scores = cross_val_score(
    rf,
    x,
    y,
    cv=kf,
    scoring='r2'
)

print("CV Scores:", scores)
print("Average CV Score:", scores.mean())


# In[57]:


plt.figure(figsize=(8,5))
sns.barplot(
    data=importance,
    x='Importance',
    y='Feature'
)

plt.title("Feature Importance")
plt.tight_layout()

plt.savefig("feature_importance.png")
plt.show()


# In[58]:


plt.figure(figsize=(8,5))
plt.scatter(y_test,pred)

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted")

plt.savefig("actual_vs_predicted.png")
plt.show()


# In[59]:


print("""
Conclusion

• Built a machine learning model to predict used car selling prices.
• Random Forest Regressor achieved the best performance.
• Test R² Score: 0.922
• MAE: 0.636
• RMSE: 1.115
• Present Price (77%) and Car Age (17%) were the most influential features.
• The model can be used to estimate resale values of used cars.
""")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




