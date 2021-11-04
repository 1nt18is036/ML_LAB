#!/usr/bin/env python
# coding: utf-8

# In[8]:


import statistics
n = 5
list = [100,2,5,1900,200]

min_ele = min(list)
max_ele = max(list)
sum = 0
print("list after normalisation is :")
for i in range(n):
    list[i] = (list[i] - min_ele) / (max_ele - min_ele)
    sum = sum + list[i]
    print(list[i])
mean = sum // n ;
print("mean tends to ~")
print(mean)
print("standard deviation is : ")
print(statistics.stdev(list))





# In[ ]:





# In[ ]:




