#!/usr/bin/env python
# coding: utf-8

# # Laptop Price EDA

# ![image.png](attachment:image.png)

# In[3]:


from warnings import filterwarnings
filterwarnings('ignore')


# ## Step 1: Read the dataset

# In[4]:


import os
os.chdir('c:/Datasets/')


# In[5]:


import pandas as pd
df = pd.read_csv('laptopPrice.csv')
df.head()


# ## Step 2:Data Quality Check

# In[6]:


df.info()


# In[7]:


df.isna().sum()


# In[8]:


df.duplicated().sum()


# ## Drop the duplicates

# In[9]:


df = df.drop_duplicates(keep='first')
df.head()


# ## Cat con seperation for df

# In[10]:


df.columns


# In[11]:


df.dtypes


# In[12]:


cat = list(df.columns[df.dtypes=='object'])
cat


# In[13]:


con = list(df.columns[df.dtypes!='object'])


# In[14]:


con


# ### Perform descriptive analytics for cat and con features

# In[15]:


df[cat].describe().T


# In[16]:


df['brand'].value_counts()


# In[17]:


df['processor_brand'].value_counts()


# In[18]:


df[con].describe().T


# ![image.png](attachment:image.png)

# ## Perform Data Visualization

# In[19]:


import matplotlib.pyplot as plt
import seaborn as sns


# ![image.png](attachment:image.png)

# In[20]:


for i in cat:
    plt.figure(figsize=(12,6))
    sns.countplot(data=df, x=i)
    plt.title(f'Countplot for {i}')
    plt.show()


# In[21]:


for i in con:
    plt.figure(figsize=(12,6))
    sns.histplot(data=df, x=i, kde=True)
    plt.title(f'Histogram for {i}')
    plt.show()


# ## Multivariate analysis

# ![image.png](attachment:image.png)

# In[22]:


con


# In[23]:


plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='Number of Ratings', y='Price')
plt.title(f'Scatterplot for Number of Ratings vs Price')
plt.show()


# In[24]:


plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='Number of Reviews', y='Price')
plt.title(f'Scatterplot for Number of Reviews vs Price')
plt.show()


# In[25]:


plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='Number of Ratings', y='Number of Reviews')
plt.title(f'Scatterplot for Number of Ratings vs Number of Reviews')
plt.show()


# ## Number of ratings and Number of Reviews are Linearly dependent on each other

# ### Correlation Plot

# In[26]:


cor = df[con].corr()
cor


# In[27]:


sns.heatmap(cor, annot=True)


# ## Cat  vs con - Boxplot

# In[28]:


cat


# In[29]:


con


# In[30]:


plt.figure(figsize=(16,8))
sns.boxplot(data=df, x='brand', y='Price')
plt.title('Boxplot for Laptop brand vs price')
plt.show()


# In[31]:


for i in cat:
    plt.figure(figsize=(12,6))
    sns.boxplot(data=df, x=i, y='Price')
    plt.title(f'Boxplot for {i} vs Price')
    plt.show()


# ## Cateogical vs Categorical - Crosstab heatmap

# In[32]:


cat


# In[33]:


ctab1 = pd.crosstab(df['processor_brand'], df['processor_gnrtn'])
ctab1


# In[34]:


sns.heatmap(ctab1, annot=True, fmt='d')


# In[35]:


ctab2 = pd.crosstab(df['brand'], df['rating'])
ctab2


# In[36]:


sns.heatmap(ctab2, annot=True, fmt='d')


# In[38]:


ctab3 = pd.crosstab(df['ram_gb'], df['ram_type'])
ctab3


# In[39]:


sns.heatmap(ctab3, annot=True, fmt='d')


# ## Multivarite analysis-Pairplot

# In[40]:


sns.pairplot(data=df)


# In[41]:


cat


# In[42]:


sns.pairplot(data=df, hue='processor_brand')


# In[43]:


sns.pairplot(data=df, hue='ram_gb')


# In[ ]:




