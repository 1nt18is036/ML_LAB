#!/usr/bin/env python
# coding: utf-8

# In[18]:


from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
import statistics as std

matrix1 = [[4,1,2,2] , [1,3,9,3] , [5,7,5,1]]
print("mean:", Normalizer().fit_transform(matrix2).mean())
print("standard deviation : " , Normalizer().fit_transform(matrix2).std())
print("after standardization : \n" , Normalizer().fit_transform(matrix2))





# In[17]:


matrix2 = [[4,1,2,2] , [1,3,9,3] , [5,7,5,1]]
#scaler = StandardScaler()
#print(scaler.mean_)
print("mean:", StandardScaler().fit_transform(matrix2).mean())
print("standard deviation : " , StandardScaler().fit_transform(matrix2).std())
print("after standardization : \n" , StandardScaler().fit_transform(matrix2))


#t1.transform(matrix2)




# In[ ]:





# In[ ]:





# In[ ]:




