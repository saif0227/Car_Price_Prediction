#!/usr/bin/env python
# coding: utf-8

# In[52]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
from statsmodels.formula.api import ols

from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# import dataset
df = pd.read_csv(r'C:\Users\hp\OneDrive\Desktop\Stats And ML\Data sets\car_price.csv')


# In[3]:


df.head()


# In[4]:


df.info()


# In[5]:


df.describe(include='all')


# In[6]:


df.drop(columns=['Unnamed: 0'], axis = 1, inplace = True)


# In[7]:


df.head()


# In[8]:


# Find the unique value in give dataset

for i in df.columns:
    print("*********************************************", i ,
         "******************************************************")
    print()
    print(set(df[i].tolist()))
    print()


# In[9]:


print(df.shape)


# In[10]:


df.duplicated().sum()


# In[11]:


# Number and percentage of unique value in each column 

unique_counts = []

for col in df.columns:
    #print(col)
    unique_counts.append((col, df[col].nunique() ))
unique_counts = sorted(unique_counts, key = lambda x:x[1], reverse = True)
print("No of unique values in each columns are as follows :(In Descending order)\n")

for col, nunique in unique_counts:
    print(f"{col}:{nunique}:{round(nunique/5512 * 100, 2)}%")


# In[12]:


target = 'car_prices_in_rupee'


# In[ ]:





# In[ ]:





# In[13]:


df.isnull().sum()


# In[14]:


# Data Preprocessing

# Part -1 : Handle Missing Values

def get_missing_data_details(df):
    sns.heatmap(df.isnull(), yticklabels= False, cbar= False)
    total = df.isnull().sum().sort_values(ascending = False)
    percent = ((df.isnull().sum()/df.isnull()).count()*100).sort_values(ascending = False)
    missing = pd.concat([total, percent], axis = 1, keys = ['Total','Percent'])
    missing = missing[missing['Percent']>0]
    


# In[15]:


get_missing_data_details(df)


# In[16]:


df['car_name'][100]


# In[17]:


df.head(2)


# In[18]:


def converted_price(s):
    if type(s) == str:
        s = s.lower()
        if 'lakh' in s:
            s = s.strip('lakh')
            s = float(s)*100000
        return s


# In[19]:


df['car_prices_in_rupee'] = df['car_prices_in_rupee'].apply(converted_price)
df['car_prices_in_rupee'] = pd.to_numeric(df['car_prices_in_rupee'], errors = 'coerce')


# In[20]:


df['kms_driven'] = pd.to_numeric(df['kms_driven'].str.replace('[^\d]','',regex=True))


# In[21]:


df.info()


# In[22]:


company = []
brand = []
model = []
for i in df['car_name']:
    car = i.split(' ')
    company.append(car[0])
    brand.append(car[1])
    model.append(' '.join(car[1:]))
else:
    df['company'] = company
    df['brand'] = brand
    df['model'] = model
    df.drop(columns= 'car_name',inplace=True)
df.head()


# # Checking for missing value

# In[23]:


def Missing_Features_values(df):
    missing_ = df.isnull().sum()[df.isnull().sum()>0]
    if len(missing_):
        for i in missing_.index:
            print(f'\n\033[1m {missing_[i]} Missing Values in the feature "{i}"')
            continue
    else:
         print('\n\033[1m No Missing Values')

Missing_Features_values(df)


# # Filling missing value

# In[24]:


df['car_prices_in_rupee'] = df['car_prices_in_rupee'].fillna(df['car_prices_in_rupee'].median())


# # Part -2 : Handling Encoding

# In[25]:


def classifire_preprocesor(df,numeric):
    for col in df.columns:
        if col in numeric:
            df[col] = df[col].astype('float64')
            continue
        df[col] = df[col].astype('category')
    return df

numeric =  ['car_prices_in_rupee','kms_driven']
df = classifire_preprocesor(df,numeric=numeric)


# In[26]:


Encoding_targets = df.select_dtypes(exclude=(int,float))
Encoding_targets.head()


# In[27]:


Enc_target_freq = Encoding_targets.nunique()
Enc_target_freq


# In[28]:


Enc_target_freq[Enc_target_freq<=5].index


# In[29]:


for col in Enc_target_freq.index:
    
    if col in Enc_target_freq[Enc_target_freq<=5].index:  
        print("\033[1mOne-Hot Encoding on features: \033[0m",col)
        df = pd.concat((df,pd.get_dummies(df[col],prefix='enc_',prefix_sep='',drop_first=True,dtype=int)),axis=1)
        df.drop(columns=col,inplace=True)
        continue
    print("\033[1mLabel Encoding on features: \033[0m",col)
    df[f'enc_{col}'] = LabelEncoder().fit_transform(df[col])
    df.drop(columns=col,inplace=True)


df.head()


# In[30]:


# Encoding done


# In[31]:


df.describe()


# In[32]:


Q1,Q3 = df[target].quantile([0.25,0.75])
IQR = Q3 - Q1
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR
print("Q1:",Q1)
print("Q3:",Q3)
print("IQR:",IQR)
print("lower limit:",lower_limit)
print('Upper limit:',upper_limit)


# In[33]:


df[(df[target]<upper_limit)]


# In[34]:


print("Skew :",df[target].skew())


plt.figure(figsize=(5,8))
plt.subplot(2,1,1)
sns.distplot(df[target])


plt.subplot(2,1,2)
sns.boxplot(df[target])
plt.show()


# In[35]:


orignaldf = df.copy()


# In[36]:


df=df[(df[target]<upper_limit)]


# In[37]:


df.shape


# In[38]:


sns.pairplot(data=df,y_vars=target,x_vars=df.columns)


# In[39]:


print("Skew :",df[target].skew())


plt.figure(figsize=(5,8))
plt.subplot(2,1,1)
sns.distplot(df[target])


plt.subplot(2,1,2)
sns.boxplot(df[target])
plt.show()


# # cheking for outliers 

# In[41]:


numerical='kms_driven'
print("Skew :",df['kms_driven'].skew())


plt.figure(figsize=(5,8))
plt.subplot(2,1,1)
sns.distplot(df['kms_driven'])


plt.subplot(2,1,2)
sns.boxplot(df['kms_driven'])
plt.show()


# In[42]:


Q1,Q3 = df['kms_driven'].quantile([0.25,0.75])
IQR = Q3 - Q1
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR
print("Q1:",Q1)
print("Q3:",Q3)
print("IQR:",IQR)
print("lower limit:",lower_limit)
print('Upper limit:',upper_limit)


# In[43]:


df = df[df['kms_driven']<upper_limit]


# In[44]:


df.describe()


# In[45]:


numerical='kms_driven'
print("Skew :",df['kms_driven'].skew())


plt.figure(figsize=(5,8))
plt.subplot(2,1,1)
sns.distplot(df['kms_driven'])


plt.subplot(2,1,2)
sns.boxplot(df['kms_driven'])
plt.show()


# In[46]:


Q1,Q3 = df['kms_driven'].quantile([0.25,0.75])
IQR = Q3 - Q1
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR
print("Q1:",Q1)
print("Q3:",Q3)
print("IQR:",IQR)
print("lower limit:",lower_limit)
print('Upper limit:',upper_limit)


# In[47]:


(df['kms_driven']>upper_limit).sum()


# In[48]:


df['kms_driven'] = np.clip(df['kms_driven'],lower_limit,upper_limit)


# In[49]:


y = pd.DataFrame(df[target])
x = df.drop(columns = target)


# In[50]:


y.head()


# # Splitting the data into x_train, y_train

# In[53]:


x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=.25,random_state=101)

print("Orignal Data Size --------", x.shape)
print("Training Data Size: --------", x_train.shape)
print("Testing Data Size: --------", x_test.shape)


# # Feature Scaling

# In[54]:


print('\033[1m  Scaling Training Data\033[22m\n'.center(120))
Scaled_x_train = StandardScaler().fit_transform(x_train)
display(pd.DataFrame(Scaled_x_train,columns=x_train.columns).describe())


# In[55]:


print('\033[1m  Scaling Testing Data\033[22m\n'.center(120))
Scaled_x_test = StandardScaler().fit_transform(x_test)
display(pd.DataFrame(Scaled_x_test,columns=x_test.columns).describe())


# In[56]:


print('\033[1m  Scaling Orignal X Data\033[22m\n'.center(120))
Scaled_x = StandardScaler().fit_transform(x)
Scaled_x = pd.DataFrame(Scaled_x,columns=x.columns)
display(Scaled_x.describe())


# In[57]:


# Checking the Correlation among all the variabels
print('\033[1mCorrelation Matrix'.center(120))
plt.figure(figsize=(25,25))
sns.heatmap(data=df.corr(),vmin=-1,vmax=1,annot=True,cmap='coolwarm')
plt.show()


# # Linear Regression Model

# In[58]:


linearModel = LinearRegression()
linearModel.fit(Scaled_x_train,y_train)

print("Raw Training r2 Score ",r2_score(y_train,linearModel.predict(Scaled_x_train)))
print("Raw Testing r2 Score ",r2_score(y_test,linearModel.predict(Scaled_x_test)))
print('Slopes : ',linearModel.coef_)
print('Intercept',linearModel.intercept_)


# In[59]:


def Significance_test(df,target='y',exclude=False):
    """exclude that column which you are not interested in,
    to compare with target variable
    
    Keyword arguments:
    argument -- description
    Return: pandas dataframe with Featurename and pr(>F) score
    """
    
    if exclude:
        df = df.drop(columns=exclude)

    formula = f'{target} ~ ' + ' + '.join([f'{col}' for col in df.columns.difference([target])])


    model = ols(formula, data=df).fit()

    anova_table = sm.stats.anova_lm(model, typ=2)

    feature_Signific_score = pd.DataFrame(anova_table).iloc[:-1]['PR(>F)']

    return pd.DataFrame({'Feature':feature_Signific_score.index,'PR(>F)':feature_Signific_score.values},)


# In[60]:


results = Significance_test(df,target=target)

print("\033[1m -- Anova Tabel -- \033[0m".center(35))
display(results)
print(end='\n\n')

print("\033[1m -- Significant Features -- \033[0m".center(35))
Significant = results[results['PR(>F)']<=0.05]
display(Significant)
print(end='\n\n')

print("\033[1m-- Non-Significant Features -- \033[0m".center(35))
Non_significant = results[results['PR(>F)']>0.05]
display(Non_significant)


# In[61]:


def Vif_score(x,exclude=None):
    if exclude:
        x = x.drop(columns=exclude)
    scaled_x = StandardScaler().fit_transform(x)
    vif=pd.DataFrame()
    vif['Features'] = x.columns
    vif['VIF'] = [variance_inflation_factor(scaled_x, i) for i in range(scaled_x.shape[1])]
    return vif


# In[62]:


Vif_score(x,exclude=False)


# In[63]:


exclude=['enc_model','enc_Petrol'] # by neglecting these we solved the multicoliarity issue, now lets solve non significant issue

Vif_score(x,exclude=exclude) # droped the high correlated columns


# # Linear Regression Model After removing highest VIF variable

# In[64]:


def evaluate_model(x, y, model, test_size=0.2):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=42)

    model.fit(x_train, y_train)

    training_y = model.predict(x_train)
    testing_y = model.predict(x_test)

    training_mse = mean_squared_error(y_train, model.predict(x_train))
    testing_mse = mean_squared_error(y_test, model.predict(x_test))
    
    training_r2 = r2_score(y_train, model.predict(x_train))
    testing_r2 = r2_score(y_test, model.predict(x_test))
    
    return training_mse,testing_mse, training_r2,testing_r2


# In[65]:


Scaled_x.columns


# In[66]:


training_mse,testing_mse, training_r2,testing_r2 = evaluate_model(Scaled_x,y,LinearRegression(),test_size=0.20)

print("Training Mean Square Error :",training_mse)
print("Testing Mean Square Error :",testing_mse)
print("Training R2 Score :",training_r2)
print("Testing R2 Score :",testing_r2)



# In[67]:


exclude # we get these in final iteration


# In[68]:


x.drop(columns=exclude)


# In[69]:


training_mse,testing_mse, training_r2,testing_r2 = evaluate_model(x.drop(columns=exclude),y,LinearRegression(),test_size=0.25)

print("Training Mean Square Error :",training_mse)
print("Testing Mean Square Error :",testing_mse)
print("Training R2 Score :",training_r2)
print("Testing R2 Score :",testing_r2)


# # OLS Method

# In[70]:


x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=.25,random_state=101)

olsModel = OLS(y_train,x_train).fit()
olsModel.summary()


# In[71]:


x_train,x_test, y_train, y_test = train_test_split(x.drop(columns=['enc_engine','enc_ownership']),y,test_size=.25,random_state=101)

olsModel = OLS(y_train,x_train).fit()
olsModel.summary()


# In[73]:


x_train,x_test, y_train, y_test = train_test_split(x.drop(columns=['enc_Electric','enc_engine','enc_ownership']),y,test_size=.25,random_state=101)

olsModel = OLS(y_train,x_train).fit()
olsModel.summary()


# In[74]:


Significant['Feature'].values


# In[75]:


evaluate_model(x,y['car_prices_in_rupee'].values,LinearRegression())


# In[76]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression()
lm.fit(x_train, y_train)
print("lm model", lm.coef_)


# In[77]:


yt= y_train['car_prices_in_rupee']


# # Cross validation check -- K-Fold

# In[78]:


from sklearn.model_selection import cross_val_score
training_accuracy = cross_val_score(LinearRegression() , x,y, cv=500)


# In[79]:


print('training_accuracy for all 10 indivisual :', training_accuracy)
print()
print("training_accuracy with mean value :", training_accuracy.mean())
print()
print("training_accuracy max value :", training_accuracy.max())


# # Regularization --

# ### Lasso Method

# In[80]:


x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=.25,random_state=101)
lassoModel = Lasso(alpha=.5)
lassoModel.fit(x_train,y_train,)

print("Raw Training r2 Score ",r2_score(y_train,lassoModel.predict(x_train)))
print("Raw Testing r2 Score ",r2_score(y_test,lassoModel.predict(x_test)))
print('Slopes :',lassoModel.coef_)
print('Intercept',lassoModel.intercept_)


# ### Ridge Method

# In[81]:


x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=.25,random_state=101)

ridge = Ridge(alpha=.8)
ridge.fit(x_train, y_train)

print("Raw Training r2 Score ",r2_score(y_train,ridge.predict(x_train)))
print("Raw Testing r2 Score ",r2_score(y_test,ridge.predict(x_test)))
print('Slopes :',ridge.coef_)
print('Intercept',ridge.intercept_)


# ### Elastic Net

# In[82]:


x_train,x_test, y_train, y_test = train_test_split(x,y,test_size=.25,random_state=101)

elastic = ElasticNet(alpha=0.1,l1_ratio=0.1)
elastic.fit(x_train, y_train)

print("Raw Training r2 Score ",r2_score(y_train,elastic.predict(x_train)))
print("Raw Testing r2 Score ",r2_score(y_test,elastic.predict(x_test)))
print('Slopes :',elastic.coef_)
print('Intercept',elastic.intercept_)


# ### Gradient Descent

# In[83]:


from sklearn.linear_model import SGDRegressor
gdmodel = SGDRegressor(alpha=0.1,penalty='l1',)
gdmodel.fit(x_train, y_train)

print("Raw Training r2 Score ",r2_score(y_train,gdmodel.predict(x_train)))
print("Raw Testing r2 Score ",r2_score(y_test,gdmodel.predict(x_test)))
print('Slopes :',gdmodel.coef_)
print('Intercept',gdmodel.intercept_)


# In[ ]:




