#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import export_graphviz
from six import StringIO  
from IPython.display import Image  
import pydotplus
# import os

# os.environ['PATH'] = os.environ['PATH']+';'+os.environ['CONDA_PREFIX']+r"\Library\bin\graphviz"





col_names = ["hair","feathers","eggs","milk","airborne","aquator","predator","toothed","backbone","breathes","venemous","fins","legs","tails","domestic","catsize","type"]
dataset = pd.read_csv("zoo_data.csv",header=None,names=col_names)
dataset.head()
X = dataset.values[:,:16]
Y = dataset.values[:,-1]
print(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=1)
clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3)
clf = clf.fit(X_train, Y_train)
y_pred = clf.predict(X_test)
print(y_pred)
print("ACCURACY IS:" , accuracy_score(Y_test, y_pred))
print(confusion_matrix(Y_test,y_pred))
# tree.plot_tree(clf)
feature_cols= ["hair","feathers","eggs","milk","airborne","aquator","predator","toothed","backbone","breathes","venemous","fins","legs","tails","domestic","catsize"]

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['1','2','3','4','5','6','7'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('zoo.png')
Image(graph.create_png())





# In[ ]:





# In[ ]:





# In[ ]:




