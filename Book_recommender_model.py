#!/usr/bin/env python
# coding: utf-8

# ### Import the libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ### Reading the datatsets

# In[2]:


books = pd.read_csv('Data/Books.csv')
books.head(2)


# In[3]:


ratings = pd.read_csv('Data/Ratings.csv')
ratings.head(2)


# In[4]:


users = pd.read_csv('Data/Users.csv')
users.head(2)


# #### fix the null values and other mistakes

# In[5]:


books['Book-Author'].fillna('Larissa Anne Downes',inplace=True)
books['Publisher'].fillna('Novelbooks Inc',limit=1,inplace=True)
books['Publisher'].fillna('Bantam',inplace=True)
books['https'] = books['Image-URL-M'].apply(lambda x:str(x).split(":")[0]).replace('http','https:')
books['remaining'] = books['Image-URL-M'].apply(lambda x:str(x).split(":")[1])
books['Image-URL'] = books['https'].astype(str) + books['remaining'].astype(str)
books.drop(['https','remaining','Image-URL-L','Image-URL-M','Image-URL-S'],axis=1,inplace=True)


# In[6]:


merged_df = ratings.merge(books,on = 'ISBN')
merged_df = merged_df.replace(['TokyoPop','J. R. R. Tolkien','J.R.R. TOLKIEN','Antoine de Saint-ExupÃ©ry'],['Tokyopop','J.R.R. Tolkien','J.R.R. Tolkien','Antoine de saint-exupery'])#After finding top publishers i realised Tokyopop at some places has been misspelled
merged_df['Book-Author'] = merged_df['Book-Author'].str.replace('[Ãx][^A-Za-z]+','',regex=True)


# ### Lets build a collaborative filtering model

# #### The conditions we are putting forward are :
# 
# ##### 1. Ony consider the users who have rated atleast 200 books
# ##### 2. only consider the books who have been rated by ateast 50 users

# In[7]:


z = merged_df.groupby('User-ID')['Book-Rating'].count() > 200
users_with_ratings = z[z].index
users_merged = merged_df[merged_df['User-ID'].isin(users_with_ratings)]


# In[8]:


u =users_merged.groupby('Book-Title')['Book-Rating'].count() >= 50
books_with_ratings = u[u].index
total_df = users_merged[users_merged['Book-Title'].isin(books_with_ratings)]


# #### Now we will convert this into a pivot table

# In[9]:


total_df.head(1)


# In[10]:


pivot_df = total_df.pivot_table(index='Book-Title',columns='User-ID',values='Book-Rating').fillna(0)


# In[11]:


from sklearn.metrics.pairwise import cosine_similarity

sim_scores = cosine_similarity(pivot_df)

def recommend_book(book):
    index = np.where(pivot_df.index == book)[0][0]
    similarity = sorted(list(enumerate(sim_scores[index])),key = lambda x:x[1],reverse = True)[1:6]
    
    
    data = []
    for i in similarity:
        item = []
        temp_df = books[books['Book-Title'] == pivot_df.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL'].values))

        data.append(item)

    print(data)
        
        
    
    


# In[12]:


recommend_book('1984')


# In[13]:


import pickle 

pickle.dump(pivot_df,open('pivot_df.pkl','wb'))
pickle.dump(sim_scores,open('sim_scores.pkl','wb'))
pickle.dump(books,open('books.pkl','wb'))

