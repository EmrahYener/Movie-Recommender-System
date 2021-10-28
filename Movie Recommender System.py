#!/usr/bin/env python
# coding: utf-8

# # PROBLEM STATEMENT

# - This notebook implements a movie recommender system. 
# - Recommender systems are used to suggest movies or songs to users based on their interest or usage history. 
# - For example, Netflix recommends movies to watch based on the previous movies you've watched.  
# - In this example, we will use Item-based Collaborative Filter 
# 
# 
# 
# - Dataset MovieLens: https://grouplens.org/datasets/movielens/100k/ 
# - Photo Credit: https://pxhere.com/en/photo/1588369

# ![image.png](attachment:image.png)

# # STEP #0: LIBRARIES IMPORT
# 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # STEP #1: IMPORT DATASET

# In[2]:


# Two datasets are available, let's load the first one:
movie_titles_df = pd.read_csv("Movie_Id_Titles")
movie_titles_df.head(20)


# In[3]:


# Let's load the second one!
movies_rating_df = pd.read_csv('u.data', sep='\t', names=['user_id', 'item_id', 'rating', 'timestamp'])


# In[4]:


movies_rating_df.head(10)


# In[5]:


movies_rating_df.tail()


# In[6]:


# Let's drop the timestamp 
movies_rating_df.drop(['timestamp'], axis = 1, inplace = True)


# In[7]:


movies_rating_df


# In[8]:


movies_rating_df.describe()


# In[9]:


movies_rating_df.info()


# In[10]:


# Let's merge both dataframes together so we can have ID with the movie name
movies_rating_df = pd.merge(movies_rating_df, movie_titles_df, on = 'item_id') 


# In[11]:


movies_rating_df


# In[12]:


movies_rating_df.shape


# # STEP #2: VISUALIZE DATASET

# In[13]:


movies_rating_df.groupby('title')['rating'].describe()


# In[14]:


ratings_df_mean = movies_rating_df.groupby('title')['rating'].describe()['mean']


# In[15]:


ratings_df_count = movies_rating_df.groupby('title')['rating'].describe()['count']


# In[16]:


ratings_df_count


# In[17]:


ratings_mean_count_df = pd.concat([ratings_df_count, ratings_df_mean], axis = 1)


# In[18]:


ratings_mean_count_df.reset_index()


# In[19]:


ratings_mean_count_df['mean'].plot(bins=100, kind='hist', color = 'r') 


# In[20]:


ratings_mean_count_df['count'].plot(bins=100, kind='hist', color = 'r') 


# In[21]:


# Let's see the highest rated movies!
# Apparently these movies does not have many reviews (i.e.: small number of ratings)
ratings_mean_count_df[ratings_mean_count_df['mean'] == 5]


# In[22]:


# List all the movies that are most rated
# Please note that they are not necessarily have the highest rating (mean)
ratings_mean_count_df.sort_values('count', ascending = False).head(100)


# # STEP #3: PERFORM ITEM-BASED COLLABORATIVE FILTERING ON ONE MOVIE SAMPLE

# In[23]:


userid_movietitle_matrix = movies_rating_df.pivot_table(index = 'user_id', columns = 'title', values = 'rating')


# In[24]:


userid_movietitle_matrix


# In[25]:


titanic = userid_movietitle_matrix['Titanic (1997)']


# In[26]:


titanic


# In[27]:


# Let's calculate the correlations
titanic_correlations = pd.DataFrame(userid_movietitle_matrix.corrwith(titanic), columns=['Correlation'])
titanic_correlations = titanic_correlations.join(ratings_mean_count_df['count'])


# In[28]:


titanic_correlations


# In[29]:


titanic_correlations.dropna(inplace=True)
titanic_correlations


# In[30]:


# Let's sort the correlations vector
titanic_correlations.sort_values('Correlation', ascending=False)


# In[31]:


titanic_correlations[titanic_correlations['count']>80].sort_values('Correlation',ascending=False).head()


# In[32]:


# Pick up star wars movie and repeat the excerise


# # STEP#4: CREATE AN ITEM-BASED COLLABORATIVE FILTER ON THE ENTIRE DATASET 

# In[33]:


# Recall this matrix that we created earlier of all movies and their user ID/ratings
userid_movietitle_matrix


# In[34]:


movie_correlations = userid_movietitle_matrix.corr(method = 'pearson', min_periods = 80)
# pearson : standard correlation coefficient
# Obtain the correlations between all movies in the dataframe


# In[35]:


movie_correlations


# In[36]:


# Let's create our own dataframe with our own ratings!
myRatings = pd.read_csv("My_Ratings.csv")
#myRatings.reset_index


# In[37]:


myRatings


# In[38]:


len(myRatings.index)


# In[39]:


myRatings['Movie Name'][0]


# In[40]:


similar_movies_list = pd.Series()
for i in range(0, 2):
    similar_movie = movie_correlations[myRatings['Movie Name'][i]].dropna() # Get same movies with same ratings
    similar_movie = similar_movie.map(lambda x: x * myRatings['Ratings'][i]) # Scale the similarity by your given ratings
    similar_movies_list = similar_movies_list.append(similar_movie)


# In[41]:


similar_movies_list.sort_values(inplace = True, ascending = False)
print (similar_movies_list.head(10))

