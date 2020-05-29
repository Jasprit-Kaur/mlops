#!/usr/bin/env python
# coding: utf-8

# In[48]:


from keras.datasets import mnist


# In[49]:


dataset = mnist.load_data('mymnist.db')


# In[50]:


train , test = dataset


# In[51]:


X_train , y_train = train


# In[52]:


X_test , y_test = test


# In[53]:


X_train.shape


# In[54]:


X_test.shape


# In[55]:


X_train_1d = X_train.reshape(60000,28,28,1)
X_test_1d = X_test.reshape(10000,28,28,1)


# In[56]:


X_train = X_train_1d.astype('float32')
X_test = X_test_1d.astype('float32')


# In[57]:


from keras.utils.np_utils import to_categorical


# In[58]:


y_train = to_categorical(y_train)


# In[59]:


from keras.models import Sequential


# In[60]:


from keras.layers import Dense


# In[61]:


from keras.layers import Convolution2D


# In[62]:


from keras.layers import Flatten


# In[63]:


model = Sequential()


# In[64]:


model.add(Convolution2D(filters=2, kernel_size=(3,3), activation='relu', input_shape=(28,28,1)))


# In[65]:


parameter=1
fil=6


# In[66]:


for parameter in range(parameter):
    model.add(Convolution2D(filters=fil, kernel_size=(3,3), activation='relu'))
    fil=fil*2


# In[67]:


model.add(Flatten())


# In[68]:


model.add(Dense(units=10, activation='softmax'))


# In[69]:


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[70]:


model.fit(X_train, y_train, epochs=3)


# In[71]:


ac=model.history.history.get('accuracy')
ac=ac[0]*100
ac=int(ac)


# In[72]:


print("MODEL ACCURACY :" , ac)


# In[43]:


f = open("a.txt" , "w+")


# In[44]:


f.write(str(ac))


# In[45]:


f.close()


# In[46]:





# In[48]:





# In[49]:





# In[50]:





# In[51]:





# In[52]:





# In[53]:





# In[54]:





# In[55]:





# In[56]:





# In[57]:





# In[58]:





# In[59]:





# In[60]:





# In[61]:





# In[62]:





# In[64]:





# In[65]:





# In[66]:





# In[69]:





# In[71]:





# In[72]:





# In[ ]:




