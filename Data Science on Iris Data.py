
# coding: utf-8

# In[1]:


from sklearn.neighbors import KNeighborsClassifier


# In[2]:


from sklearn import datasets as d


# In[3]:


iris=d.load_iris()


# In[4]:


y=iris.target


# In[5]:


x=iris.data


# In[6]:


knn=KNeighborsClassifier(n_neighbors=11)


# In[7]:


from sklearn.model_selection import train_test_split


# In[8]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4)


# In[9]:


knn.fit(x_train,y_train)


# In[10]:


y_pred=knn.predict(x_test)


# In[11]:


y_pred


# In[12]:


y_pred.shape


# In[13]:


from sklearn.metrics import accuracy_score,classification_report


# In[14]:


accuracy_score(y_test,y_pred)


# In[15]:


print(classification_report(y_test,y_pred))


# In[16]:


from matplotlib import pyplot as plt


# In[17]:


a=[]
for i in range (1,40):
    kn=KNeighborsClassifier(n_neighbors=i)
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.7)
    kn.fit(x_train,y_train)
    pred_i=kn.predict(x_test)
    a.append(accuracy_score(y_test,pred_i))
    plt.scatter(i,a[i-1])


# In[18]:


plt.figure(figsize=(12,6))
plt.plot(range(1,40),a,color='red',linestyle='-')
plt.plot(4,a[4],markersize="10",color="black")


# In[19]:


a.index(max(a))


# In[20]:


plt.plot(x_train,y_train,linestyle="dashed")
plt.plot(x_test,y_test)


# In[21]:


a[4]


# In[ ]:


knn

