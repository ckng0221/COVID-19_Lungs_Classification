#!/usr/bin/env python
# coding: utf-8

# # Exploratory Data Analysis (EDA)

# All the X-ray images are stored in the 'COVID', 'Lung_Opacity', 'Viral Pneumonia', and 'Normal' directories, correspond to their X-ray images of their class. 
# 
# First we carry out the Exploratory Discovery Analysis (EDA) for our dataset. 

# In[1]:


# Lung types with their corresponding directories
lungTypes = [
    'COVID', 
    'Lung_Opacity', 
    'Viral Pneumonia',
    'Normal'
]


# In[2]:


import cv2

# example image in each class
img1 = 'Images/COVID/COVID-1.png'
img2 = 'Images/Lung_Opacity/Lung_Opacity-1.png'
img3 = 'Images/Viral Pneumonia/Viral Pneumonia-1.png'
img4 = 'Images/Normal/Normal-1.png'
sample_images_png = [img1, img2, img3, img4]

# convert the images into numpy array, read in grayscale
sample_images_read = [cv2.imread(img, 0) for img in sample_images_png]


# In[3]:


# Readimg example lung X-ray image from Covid, Lung Opacity, Viral Penumonia, and Normal patient
import matplotlib.pyplot as plt
import numpy as np

# 2*2 subplots 
fig, ax = plt.subplots(2, 2, figsize=(15, 10)) 

n = 0
for row in ax:
    for img in row:
        img.imshow(sample_images_read[n], 'gray') #show individual image
        img.set_title(lungTypes[n]) # set title for image
        n += 1


# We can see there are similarities and differences between different lung types

# In[4]:


# The original pixel of the image is 299 * 299
sample_images_read[0].shape


# # Machine Learning

# ## Preprocessing

# In[5]:


from pathlib import Path

#empty lists
labels = [] 
image_samples = []

path = Path()
for lungtype in lungTypes: 
    img_dir = path / 'Images' / lungtype # image directory
    for img in img_dir.iterdir(): # read each image in the image directory
        data = cv2.imread(str(img), 0).reshape(1, -1) # convert the 2D 299*299 to 1 dimension array of 89401
        image_samples.append(data) # append the images to list
        labels.append(lungtype) # append the class labels to list


# In[6]:


# covert the list to numpy array
image_samples = np.concatenate(image_samples)
labels = np.array(labels)

print(f"Final shape of samples: {image_samples.shape}")


# In[7]:


# there are total of 21165 samples of 299 * 299 pixels image in the samples
print(f"No of image samples: {len(image_samples)}")


# ## Binary classification

# As our objective is only to perform binary classification between "COVID" class and "Others" (Non-covid) class, the non-covid labels are preprocessed.

# In[8]:


# use certain samples only
np.random.seed(10)
choices = np.random.randint(len(labels), size=4000)

X = image_samples[choices]
y = labels[choices]


# In[9]:


def convert_binaryclass(x):
    if x == 'COVID':
        return 1
    else:
        return 0
y_binary = np.array(list(map(convert_binaryclass, y)))


# In[10]:


# example of y_binary
y_binary


# In[11]:


# Test traing split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y_binary,
                                                    test_size=0.3,
                                                    random_state=10,
                                                   stratify=y_binary)


# In[12]:


# The y_labels distribution after the preprocessing
import pandas as pd

distribution = pd.DataFrame(y_train).value_counts(normalize=True)
distribution


# In[13]:


import numpy as np
from sklearn.decomposition import PCA

pca_700 = PCA(n_components=700)
pca_700.fit(X)

plt.grid()
plt.plot(np.cumsum(pca_700.explained_variance_ratio_ * 100))
plt.xlabel('Number of components')
plt.ylabel('Explained variance')
plt.show()


# In[14]:


explained = np.cumsum(pca_700.explained_variance_ratio_ * 100)
np.min(np.where(explained > 90))


# ## Transformation Pipeline

# In[15]:


# model pipeline 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA, TruncatedSVD
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()), #standard scaling
    ('pca', IncrementalPCA(n_components=60, batch_size=100, copy=False)), # PCA Dimensionality reduction
                    ])


# In[16]:


pipeline.fit(X_train)
X_train = pipeline.transform(X_train)


# In[17]:


X_train.shape


# ## Model Training

# ### Random Forest

# In[18]:


from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# In[19]:


from sklearn.ensemble import RandomForestClassifier

forest = RandomForestClassifier()
forest.fit(X_train, y_train)
y_train_pred = forest.predict(X_train) #prediciton based on X_train


# In[20]:


def score_check(ytrue, ypred, name):
    print(f"============= {name} Set Score ==========")
    print("Accuracy:", accuracy_score(ytrue, ypred))
    print("Precision:", precision_score(ytrue, ypred, pos_label=1))
    print("Recall:", recall_score(ytrue, ypred, pos_label=1))
    print("F1 Score:", f1_score(ytrue, ypred, pos_label=1))


# In[21]:


score_check(y_train, y_train_pred, 'Training')


# In[22]:


# for stratified cross validation
from sklearn.model_selection import StratifiedKFold

kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=10)


# In[23]:


# Collections of all the cross validation scores
final_scores = {}


# In[24]:


from sklearn.model_selection import cross_val_score

def cv_score_check(model, X, y, cv, model_name=None, scores_dict=None, use=False):
    scores = {}
    cross_val_accuracy = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    cross_val_precision = cross_val_score(model, X, y, cv=cv, scoring="precision")
    cross_val_recall = cross_val_score(model, X, y, cv=cv, scoring="recall")
    cross_val_f1 = cross_val_score(model, X, y, cv=cv, scoring="f1")

    print("Accuracy:", cross_val_accuracy.mean())
    print("Precision:", cross_val_precision.mean())
    print("Recall:", cross_val_recall.mean())
    print("F1 Score:", cross_val_f1.mean())
    
    if use: #for recording the scores
        scores['Accuracy'] = cross_val_accuracy.mean()
        scores['Precision'] = cross_val_precision.mean()
        scores['Recall'] = cross_val_recall.mean()
        scores['F1_Score'] = cross_val_f1.mean()
        scores_dict[model_name] = scores


# In[25]:


cv_score_check(forest, X_train, y_train, kf)


# The model appears to be overfitting when using training set.
# When using cross validation set, scoring slightly reduced.

# #### Fine tuning hyperparameters

# In[26]:


# Define hyperparameters to be used in the randomized search cross validation 

from sklearn.model_selection import RandomizedSearchCV
from pprint import pprint

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 100, num = 10)]

# Number of features to consider at every split
max_features = ['auto', 'sqrt']

# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 60, num=10)]
max_depth.append(None)

# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]

# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]

# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(random_grid)


# In[27]:


# Random search of parameters, using 3 fold cross validation, 
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator=forest,
                               param_distributions=random_grid,
                               n_iter=100,
                               cv=kf,
                               verbose=2,
                               random_state=10,
                               n_jobs = -1,
                               scoring='f1')
# Fit the random search model
rf_random.fit(X_train, y_train)


# In[28]:


# best model of the random forest model
forest_best = rf_random.best_estimator_
print(forest_best)


# In[29]:


cv_score_check(forest_best, X_train, y_train, kf, model_name='Random Forest', 
               scores_dict=final_scores, use=True)


# The fine-tuned model perform slightly better than the original model

# ### Logistic Regression

# In[30]:


from sklearn.linear_model import LogisticRegression

logistic = LogisticRegression()
logistic.fit(X_train, y_train)

y_train_pred = logistic.predict(X_train)


# In[31]:


score_check(y_train, y_train_pred, 'Training')


# In[32]:


# CV score before fine-tuning
cv_score_check(logistic, X_train, y_train, kf)


# In[33]:


from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV

solvers = ['newton-cg', 'lbfgs', 'liblinear']
penalty = ['l2']
c_values = [100, 10, 1.0, 0.1, 0.01]

# define grid search
grid = dict(solver=solvers,penalty=penalty,C=c_values)
grid_search = GridSearchCV(estimator=logistic, 
                           param_grid=grid,
                           n_jobs=-1,
                           cv=kf,
                           scoring='f1',
                           error_score=0)

logistic_grid = grid_search.fit(X_train, y_train)

# summarize results
print("Best: %f using %s" % (logistic_grid.best_score_, logistic_grid.best_params_))
means = logistic_grid.cv_results_['mean_test_score']
stds = logistic_grid.cv_results_['std_test_score']
params = logistic_grid.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[34]:


logistic_best = logistic_grid.best_estimator_
print(logistic_best)


# In[35]:


# CV score after fine-tuning
cv_score_check(logistic_best, X_train, y_train, kf, model_name='Logistic Regression', 
               scores_dict=final_scores, use=True)


# ### SVM

# In[36]:


# SVM
from sklearn.svm import SVC

svm = SVC()
svm.fit(X_train, y_train)

y_train_pred = svm.predict(X_train)
score_check(y_train, y_train_pred, 'Training')


# In[37]:


# CV score before fine-tuning
cv_score_check(svm, X_train, y_train, kf)


# In[38]:


# # SVM
from sklearn.svm import SVC

param_grid = {'C':[0.1,1, 10, 100],'gamma':[0.0001,0.001,'scale'],'kernel':['rbf']}
svc = SVC(probability=True)
svm_grid = GridSearchCV(svc, param_grid, scoring='f1', cv=kf)
svm_grid.fit(X_train, y_train)


# In[39]:


svm_best = svm_grid.best_estimator_
print(svm_best)


# In[40]:


# CV score after fine-tuning
cv_score_check(svm_best, X_train, y_train, kf, model_name='SVM', 
               scores_dict=final_scores, use=True)


# ## Cross Validation Results Compare

# In[41]:


result = pd.DataFrame(final_scores)
result


# We can see that SVM has the best accuracy, precision, recall, and F1 score in the cross validation set

# # ROC Curve

# In[42]:


from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_predict

# Random Forest
y_probas_forest = cross_val_predict(forest_best, X_train, y_train, cv=kf, method="predict_proba")
y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train, y_scores_forest)

# Logistic Regression
y_probas_logistic = cross_val_predict(logistic_best, X_train, y_train, cv=kf, method="predict_proba")
y_scores_logistic = y_probas_logistic[:, 1] # score = proba of positive class
fpr_logistic, tpr_logistic, thresholds_logistic = roc_curve(y_train, y_scores_logistic)

# SVM
y_scores_svm = svm_best.decision_function(X_train)
fpr_svm, tpr_svm, thresholds_svm = roc_curve(y_train, y_scores_svm)


def plot_roc_curve(fpr, tpr, label=None): 
    plt.plot(fpr, tpr, linewidth=1, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # dashed diagonal
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')

    
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest") # Random Forest Plot
plot_roc_curve(fpr_logistic, tpr_logistic, "Logistic Regression") # Random Forest Plot
plt.plot(fpr_svm, tpr_svm, "b:", label="SVM") #SVM Plot
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()


# ## ROC Scores

# In[43]:


# Random Forest
roc_auc_forest = roc_auc_score(y_train, y_scores_forest)

# Logistic Regression
roc_auc_logistic = roc_auc_score(y_train, y_scores_logistic)

# SVM
roc_auc_svm = roc_auc_score(y_train, y_scores_svm)

print(f"Random Forest ROC AUC: {roc_auc_forest}")
print(f"Logistic Regression ROC AUC: {roc_auc_logistic}")
print(f"SVM ROC AUC: {roc_auc_svm}")


# ## Test

# In[44]:


X_test = pipeline.transform(X_test)


# In[55]:


# Random Forest
rf_y_test_pred = forest_best.predict(X_test)
score_check(y_test, rf_y_test_pred, 'Test')


# In[56]:


# Logistic regression
logistic_y_test_pred = logistic_best.predict(X_test)
score_check(y_test, logistic_y_test_pred, 'Test')


# In[57]:


# SVM
svm_y_test_pred = svm_best.predict(X_test)
score_check(y_test, svm_y_test_pred, 'Test')


# We observed SVM model has the best performance.

# ## Confusion Matrix

# In[50]:


from sklearn.metrics import confusion_matrix
import seaborn as sns;


# In[62]:


# Random Forest

cf_matrix = confusion_matrix(y_test, rf_y_test_pred)
sns.heatmap(cf_matrix, annot=True, fmt='.3g');


# In[59]:


# Logistic Regression

cf_matrix = confusion_matrix(y_test, logistic_y_test_pred)
sns.heatmap(cf_matrix, annot=True, fmt='.3g');


# In[60]:


# SVM

cf_matrix = confusion_matrix(y_test, svm_y_test_pred)
sns.heatmap(cf_matrix, annot=True, fmt='.3g');


# We can see the true negative (971) and true positive (136) are relatively high compared to the false positive (26) and false negative (67). 

# In[ ]:




