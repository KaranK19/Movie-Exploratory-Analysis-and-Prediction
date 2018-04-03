
# coding: utf-8

# # MOVIE EXPLORATORY ANALYSIS

# ### <span style="color:Maroon">Our movie dataset consists of about 4000 rows and spanned across17 columns.

# ### PROBLEM STATEMENT
# * What can we say about the success of a movie before it is released? 
# * What are the main contributors to a movie being successful? 
# * Are there certain companies that have found a consistent formula?. 
# * Can we predict which films have earned the maximum profit in the last five years? 
# * What is the genre that makes a movie click?
# * Using our dataset we have tried to dig into the above questions with data on the budget, production houses, release dates and revenues, voter average and so on of several thousand films.

# In[1]:

get_ipython().magic('matplotlib inline')

import os                                  #importing os
import numpy as np                         #importing package numpy
import pandas as pd                        #importing package pandas
import seaborn as sns                      #importing package seaborn 
import matplotlib.pyplot as plt            #importing package matplotlib 


# In[2]:

movies = pd.read_csv('C:/Users/Karan Kanwal/Downloads/tmdb_5000_moviess.csv')
                                           #Reads a CSV file as a dataframe


# In[3]:

movies                                     #To view dataframe movies


# ## Data Cleaning
# ### <span style="color:Maroon">Cleaning data to fill in the missing values, smooth out the noise and correct inconsistencies in our data.

# ### <span style="color:Green">To calculate missing values

# In[4]:

def missing_values_table(df): 
        mis_val = df.isnull().sum()                                   #Calculates sum of null values i.e if rows have null values calculate the sum.
        mis_val_percent = 100 * df.isnull().sum()/len(df)             #Calculates missing value percentage where we divide sum by the total length of the dataframe
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1) #We create a missing value table that has columns for missing values and missing value percentage
        mis_val_table_ren_columns = mis_val_table.rename(             #We rename the column names in missing value table
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})    
        return mis_val_table_ren_columns
    
print(missing_values_table(movies))                                   #print the missing values table


# ### <span style="color:Green">Renaming column names

# In[5]:

#In order to improve the consistency of data,we rename columns in our dataset to make them more readable.
movies=movies.rename(columns = {'budget':'Budget','genres':'Genres' ,'homepage':'Website'  ,'id':'MovieID',
                'keywords':'Keywords','original_language':'Original_Language',
                'original_title':'Title','overview':'Overview','popularity':'Popularity',
                'production_companies':'Production_Houses','production_countries':'Production_Countries',
                'release_date':'Release_Date', 'revenue':'Revenue','runtime':'Runtime',
                'status':'Movie_Status','tagline':'Tagline','title':'Movie_Name',
                'vote_average':'Average_Vote','vote_count':'Vote_Count'})
movies.head(5)                                                       #Returns the top 5 rows of the frame


# ### <span style="color:Green">We replace the inconsistencies of various values present in the columns.

# In[6]:

s = movies['Production_Houses'].str.replace('{"name": "',"")
r=s.str.replace('[',"")
t=r.str.replace(']',"")
u=t.str.replace('}',"")
v=u.str.replace('", "id":',"")  
w=v.str.replace("\d","")
movies['Production_Houses']=w                


# In[7]:

s = movies['Production_Countries'].str.replace('"}'," ")
t=s.str.replace('[',"")
u=t.str.replace(']',"")
v=u.str.replace('{"iso_3166_1": "',"")
x=v.str.replace('", "name": "',"")
movies['Production_Countries']=x


# In[8]:

a = movies['Keywords'].str.replace('{"id": ',"")
b=a.str.replace('[',"")
c=b.str.replace(']',"")
d=c.str.replace('"}',"")
e=d.str.replace(', "name": "',"") 
f=e.str.replace("\d","")
movies['Keywords']=f 


# In[9]:

movies.head()


# ### <span style="color:Green">Replacing blank or null values with an appropriate message

# In[10]:

#Below we handle missing data,as there are certain rows in the Website and Tagline column as missing,we replace these columns
#with values which will enable us to go head with out analysis
movies['Website'].fillna('No Website available', inplace=True)
movies['Tagline'].fillna('No Tagline available', inplace=True)


# ### <span style="color:Green"> Converting float datatype to integer datatype

# In[11]:

#We removie the na values and also convert the columns into suitable data types
movies['Popularity']=movies['Popularity'].fillna(0).astype(np.int64)


# In[12]:

movies['Original_Language']=movies['Original_Language'].str.upper()  #Changing the values to uppercase
movies['Release_Date'] = movies['Release_Date'].str.replace('-','/') #Replacing the character '-' with '/'
del movies['Unnamed: 19']                                            #Deleting redundant column 'Unnamed : 19'
movies.head(5)


# In[13]:

movies = movies[movies['Revenue'] != 0].dropna()                   #Dropping rows with Revenue values


# In[14]:

#Below,we calculate the profit which is the difference between the revenue and budget
movies['Profit']=movies['Revenue'] - movies['Budget']


# In[15]:

#We now check whether any of the columns have null values present in them
movies.isnull().sum()


# In[16]:

movies.head(5)


# #### <span style="color:Green"> Below,is a code to count the number of movies which have made a profit over the years

# In[17]:

count = []
for value in movies['Profit']:
    num = float(value)
    if num > 0:
        count.append(value)


# In[18]:

print(len(count))


# # Descriptive Analysis

# ### <span style="color:Maroon">Our first analysis,consists of extrapolating from the dataset,that which month is beneficial for a movie to release.In other words,which is the most profitable month in terms of the release of a movie 

# In[19]:

test=movies.sort_values('Profit', ascending=False)          #We sort the movies according to their earned profit
test['Month'] = pd.DatetimeIndex(test['Release_Date']).month#We know consider the month of release of each movie
import calendar
test['Month'] = test['Month'].apply(lambda x: calendar.month_abbr[x])
test['Month'].head(100)
test['Months_Frequency'] = test['Month'].head(100)
test['Months_Frequency'].value_counts()                     #Here we count the frequency of the most profitable movie released in various months
sns.set_style("whitegrid")
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
ax = sns.countplot(x ="Months_Frequency", data=test,palette="Set1")#We plot a figure,in order to visualize the analyzed data.


#  ### <span style="color:Blue"> From the above analysis we can see that the month of JUNE and may have lead to the release of most profitable movies.Also the months of december,jan and feb are not too suitable to release a movie.

# In[20]:

movies


# ### <span style="color:Maroon">Below is an analysis,to decipher which movies have been most succesfull in the last five years.The features which will be taken for this consideration are the budget,revenue and profit generated by the movies

# In[21]:

movies1=movies
movies1=movies1.nlargest(7, 'Profit')#Here we consider the top movies,based on the profit earned
movies1['budget']=movies['Budget']/1000
movies1['revenue']=movies['Revenue']/1000
movies1['profit']=movies['Profit']/1000
sns.set_style("dark")
fig, ax = plt.subplots()
fig.set_size_inches(11.7, 8.27)
ax = sns.barplot(x="profit", y="Title", data=movies1,palette="Set3")#We plot a graph of the profit earned by each movie along with its title


# In[22]:

sns.distplot(movies['Average_Vote'].fillna(movies['Average_Vote'].median()))
plt.show()


# In[23]:

from scipy import stats
from sklearn.cluster import KMeans
import seaborn as sns

df = movies

#Make a copy of DF
df_tr = df
df_tr['budget']=df_tr['Budget']/10000
df_tr['revenue']=df_tr['Revenue']/10000

#Transsform the timeOfDay to dummies
#df_tr = pd.get_dummies(df_tr, columns=['timeOfDay'])

#Standardize
clmns = ['budget', 'revenue','Popularity', 'Vote_Count']
df_tr_std = stats.zscore(df_tr[clmns])

#Cluster the data
kmeans = KMeans(n_clusters=2, random_state=0).fit(df_tr_std)
labels = kmeans.labels_

#Glue back to originaal data
df_tr['clusters'] = labels

#Add the column into our list
clmns.extend(['clusters'])

#Lets analyze the clusters
df_tr[clmns].groupby(['clusters']).mean()

#Scatter plot of Wattage and Duration
sns.lmplot('Popularity', 'Vote_Count', 
           data=df_tr, 
           fit_reg=False, 
           hue="clusters",  
           scatter_kws={"marker": "D", 
                        "s": 100})
plt.title('Popularity vs Vote_Count')
plt.xlabel('Popularity')
plt.ylabel('Vote_Count')


# In[24]:

import os
import pandas as pd
from pandas import DataFrame,Series
from sklearn import tree
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
import statsmodels.api as sm
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn import neighbors
from sklearn import linear_model
get_ipython().magic('matplotlib inline')



# ### <span style="color:Maroon"> Our below analysis,involves analyzing and visualizing the popularity of a movie with respect the votes it has received by a general spectrum of public

# In[25]:

#we first consider the numeric data of our dataset and transfer into a new dataset for simplicities of the operations to be performed.
movies2=movies._get_numeric_data()
movies


# In[26]:

#We now create bins or groups of average vote.
cut1 = pd.cut(movies2.Average_Vote, [0,2,3,4,8,10],labels=['0-2','2-4','4-6','6-8','8-10'])
movies2['Average_Vote'] =cut1


# In[27]:

#We now create graphs depicting the relationships between the average vote  and popularity.
sns.boxplot(data=movies2,x='Average_Vote',y='Popularity')
plt.ylim(0,200)
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(14.5, 5.5)
movies


# ### <span style="color:Maroon">Our next analysis,consists of analyzing which is the optimal time with respect to the duration of a movie.Certain movies can be too lengthy for a user,while some might finish in no time.We try and analyze the perfect duration of a movie,which will enable directors and producers to plan the duration of the movie,that increases their chances of getting maximum profit .

# In[28]:

cut = pd.cut(movies.Runtime,[0,20,40,60,80,100,120,140,160,180,200,220,240,260,280,300,320,340,360,380,400],labels=['0-20','20-40','40-60','60-80','80-100','100-120','120-140','160-180','180-200','200-220','220-240','240-260','260-280','280-300','300-320','320-340','340-360','360-380','380-400','400-420'])


# In[29]:

movies['Bin']= cut
movies


# In[30]:

#We create a graph depicting the optimal duration of a movie,with respect to our past dataset.
import matplotlib.pyplot as plt
fig,ax =plt.subplots()
fig.set_size_inches(18.5,10.5)
plt.ylim(0,100)
#sns.violinplot(x="bin", y="popularity", data=movies)
sns.pointplot(x='Bin',y='Popularity',data=movies)

plt.show()

From the above analysis,we can see that 180-200 is the most optimal duration to gain profits.Thus if a movie runs for this particular range duration,the more chances of it being successful,in comparison to movies of other durations 
# In[31]:

import statsmodels.formula.api as smf
lm = smf.ols(formula='Revenue ~ Runtime', data=movies).fit()

# print the coefficients
lm.params




# In[32]:

X_new = pd.DataFrame({'Runtime': [180]})
X_new.head()


# In[33]:

lm.predict(X_new)


# In[34]:

import pandas
import statsmodels.api as sm
from statsmodels.formula.api import ols

print (movies.corr())

model = ols("Profit ~ Revenue + Runtime + Popularity + Vote_Count + Average_Vote", data=movies).fit()
print (model.params)
print (model.summary())


# In[35]:



# Calculate correlations
corr = movies.corr()
sns.heatmap(corr)


# In[36]:

corr_mat=movies.corr(method='pearson')
plt.figure(figsize=(20,10))
sns.heatmap(corr_mat,vmax=1,square=True,annot=True,cmap='cubehelix')


# ### <span style="color:Maroon"> The genre or type of movie,plays a big role in the outcome of a movie.Certain movies of a certain genre are more liked by the public,and certain movies of unique and distinct genres tend to not do very well.We analyze that which genres make a movie click,and lead to greater profits.We analyze our datset and extrapolate which genres have lead to the most profitable movies.This analysis can give producers and directors an insight of what type of movies,should they be looking to make.

# In[37]:


df3=movies[movies['Profit'] > 0]
demo=df3['Genres'].value_counts()
demo


# In[38]:

#Below we depict a graph of the movies,their genre and their count with respect to only the movies that have made profit.
sns.set(style="darkgrid")
plt.figure(figsize=(17,6))
ax = sns.countplot(x="Genres", data=df3)


# ### <span style="color:Green">We can see above from our analysis,that comedy,drama and action movies tend to do relatively better,i.e more profit.Whereas movies such as war,musicals and western ones do not tend to do that well.

# ### <span style="color:Maroon">The production house of a movie contributes to a great extent in order to make a movie succesful.In our dataset we have analyzed the most popular datasets,in terms of revenue generated.This will help us analyze which banners have been successful over the years

# In[39]:

import matplotlib.pyplot as plt
import pandas as pd
count = movies['Production_Houses'].str.split().str.get(0).value_counts()
count=count.head(10)
plt.figure(figsize=(16,8))
# plot chart
#colors = ['#8B475D', '#001CF0', '#0038E2', '#0055D4', '#0071C6', '#008DB8', '#00AAAA', '#00C69C', '#00E28E', '#00FF80', ] 
colors = ["#E13F29", "#D69A80", "#D63B59", "#AE5552", "#CB5C3B", "#EB8076", "#96624E" ,"#C6CD78","#F1F08A","#8E3343"]
ax1 = plt.subplot(121, aspect='equal')
count.plot(kind='pie', ax=ax1, autopct='%1.1f%%', 
 startangle=90, shadow=False, legend = False, fontsize=14,colors=colors)


# ### <span style="color:Green"> In our analysis we have extrapolated that production houses such as paramount,universal and columbia have been the powerhouses over the years,and majority of movies released under these production houses,have lead to maximum profit.

# In[40]:

test1 = movies[['Revenue','Profit','Runtime','Popularity','Vote_Count','Average_Vote']].copy()


# In[41]:

test1=test1.sort_values('Profit', ascending=False)
test1['Month']=test['Month']


# In[42]:


test1['Revenue']=test1['Revenue']*10000
sns.kdeplot(test1.Revenue, test1.Runtime)
plt.ylim(50,150)


# Lets Apply PCA inorder to demcompose number of features.

# In[43]:

str_list = [] # empty list to contain columns with strings (words)
for colname, colvalue in movies.iteritems():
    if type(colvalue[1]) == str:
         str_list.append(colname)          
num_list = movies.columns.difference(str_list)  
X=movies[num_list]
X.shape


# Now we will Check How many components we should incorporate in order to Apply PCA 

# In[44]:


from sklearn.preprocessing import StandardScaler
X_standard = StandardScaler().fit_transform(X)


from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=11)
Y_sklearn = sklearn_pca.fit_transform(X_standard)

cum_sum = sklearn_pca.explained_variance_ratio_.cumsum()

sklearn_pca.explained_variance_ratio_[:10].sum()

cum_sum = cum_sum*100

fig, ax = plt.subplots(figsize=(8,8))
plt.bar(range(11), cum_sum, label='Cumulative _Sum_of_Explained _Varaince', color = 'b',alpha=0.5)


# We will take total 7 features to explain the model ,which accounts total 90% of our varaince. Therefore we will take only 7 components in our mode to explain it.Thus we get our profit in primarily 5 clusters

# In[45]:


from sklearn.decomposition import PCA as sklearnPCA
sklearn_pca = sklearnPCA(n_components=3)
X_red  = sklearn_pca.fit_transform(X_standard)
P=movies['Profit']
from mpl_toolkits.mplot3d import Axes3D
plt.clf()
fig = plt.figure(1, figsize=(8, 6))
ax = Axes3D(fig, elev=-150, azim=110)
ax.scatter(X_red[:, 0], X_red[:, 1], X_red[:, 2], c=P,cmap=plt.cm.Paired)
ax.set_title("First three PCA directions")
ax.set_xlabel("1st eigenvector")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("2nd eigenvector")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("3rd eigenvector")
ax.w_zaxis.set_ticklabels([])

plt.show()


# In[46]:

h = sns.jointplot(x="Profit", y="Runtime",kind='reg',size=10,xlim=[500000,110000000],ylim = [0,200],data=movies)


# In[50]:

movies.head(5)


# In[107]:

Famous=movies['Keywords'].head(20)
Famous.head(5)


# In[108]:

wordlist=pd.DataFrame()
wordlist


# In[ ]:




# In[109]:

for i in Famous:
    wsplit=i.split()
    wordlist=wordlist.append(wsplit,ignore_index=True)


# In[110]:

wordlist.head()


# In[111]:

allword=wordlist.groupby(0).size()


# In[112]:

allword.head()


# In[113]:

top20word=allword.sort_values(0,ascending=False).head(20)


# In[114]:

import matplotlib.pyplot as plt
top20word.plot(kind='bar',title="Top 20 Words")
plt.show()


# In[ ]:



