---
title: Machine Learning Doc
date: sys.date()
---

# This Document Contain Everything which is required to develop or program a ML model. This is a limited to a certain level.

## importing various library

### Common to all ML algorithms
- numpy
  - from numpy.random improt randn : used for getting the same:seed() random value each time you run.
- pandas, modin.pandas
- seaborn 
- matplotlib, matplotlib.pyplot as plt
- Uses of this library should be known to you
 
 ### for Classification type model
 ```python
 from sklearn import tree
 clf=tree.DecisionTreeClassifier()
 clf=clf.fit(features,lables)
 print(clf.predict([[150,0]]))
 ```


### For a large dataset or managing 
- we use the intel library of pandas ie. modin.pandas
- for modin.panda we should have ray lib of python `pip install ray[default]`
- 
### For Linear regression model


- `from sklearn import linear_model`
- for prediction
```python
>> NOTE: This is not complete code it only have the terms used for l.rgression
coefficient=[]
intercept=[]
regress_model={}
regr=linear_model.LinearRegression()
regr.fit(train_x,train_y)
regress_model[df[i]]=regr
print("relation b/t df[i] and lable_to_check")
print("Coefficients: ",regr.coef_)
print("Intercept: ",regr.intercept_)
coefficient.append(regr.coef_)
intercept.append(regr.intercept_)
```

#### To get the r2_score 
```python
from sklearn.matrics import r2_score
test_y_=regress_model[i].predict(test_x)
print("R2_score: %.2f"% r2_score(test_y_,test_y))
```

### For LogisticRegression 

- importing
```python
import modin.pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearnex import patch_sklearn
patch_sklearn()
from sklearn.linear_model import LogisticRegression
```

#### Training and spliting the data
- code
```python
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(np_X_procs,np_Y,test_size=0.2,random_state=4)
print("Train Set: ",X_train.shape,Y_train.shape)
print("Train Set: ",X_test.shape,Y_test.shpae)
```

### Regression Multiple
#### Training model
```python
model=LogisticRegression(C=0.001,solver='liblinear',verbose=1)
model.fit(X_train,Y_train)
Y_pred=model.predict(X_test)
```

#### Predicting 
```python 
Y_pred=model.predict(X_test)
Y_pred_prob=model.predict_proba(X_test)
print(Y_pred)
print('\n')
print((Y_pred_prob))
```

#### Model Evaluation
```python
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
print("Model aachieved a classification accuracy of:",end='\t')
print(accuracy_score(Y_test,Y_pred))
dsp=ConfusionMatrixDisplay(confusion_matrix(Y_test,Y_pred),display_labels=["Yes","No"])
print('\n')
dsp.plot()
print("Model Confusion Matrix")
from sklearn.metrics import jaccard_score
print('\n')
print("Jaccard Similarity Score:", end='\t')
print(jaccard_score(Y_test,Y_pred))
```

### SVM Support Vector Machine
+ write your regular code till separating and testing your data
```python
from sklearn import svm
clf=svm.SCV(kernel='rbf',gamma='auto')
clf.fit(X_train,Y_train)

```
+ here create a obj of svm with specific kernel and gamma is mentioned
+ Now training the model with train data set
```python
yhat=clf.predict(X_test)
yhat[:5]
```
+ Now predicting test with x dataset
```python
from sklearn.metrics import classification_report, confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[2,4])
np.set_printoptions(precision=2)

print (classification_report(y_test, yhat))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix')

```


