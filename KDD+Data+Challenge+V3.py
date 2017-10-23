
# coding: utf-8

# In[2]:


# Load the main libraries and define some functions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.feature_selection import SelectPercentile, chi2, SelectKBest, RFE, RFECV, VarianceThreshold
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import scale, StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
import re
from sklearn.base import TransformerMixin
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def cv_optimize(clf, parameters, X, y, n_jobs=1, n_folds=5, score_func=None):
    if score_func:
        gs = GridSearchCV(clf, param_grid=parameters, cv=n_folds, n_jobs=n_jobs, scoring=score_func)
    else:
        gs = GridSearchCV(clf, param_grid=parameters, n_jobs=n_jobs, cv=n_folds)
    gs.fit(X, y)
    print ("Best parameters:", gs.best_params_)
    best = gs.best_estimator_
    return best

# Save classification report as pandas dataframe
def to_table(report):
    report = report.splitlines()
    res = []
    res.append(['']+report[0].split())
    for row in report[2:-2]:
       res.append(row.split())
    lr = report[-1].split()
    res.append([' '.join(lr[:3])]+lr[3:])
    return np.array(res)

def clsf_df(ytest, ypred):
    classification_scores = to_table(classification_report(ytest, ypred))
    df = pd.DataFrame(data = classification_scores[1:4], columns = classification_scores[0])
    return df

def do_classify(clf, parameters, Xtrain, ytrain, Xtest, ytest, score_func=None, n_folds=5, n_jobs=1):
   
    if parameters:
        clf = cv_optimize(clf, parameters, Xtrain, ytrain, n_jobs=n_jobs, n_folds=n_folds, score_func=score_func)
    clf=clf.fit(Xtrain, ytrain)
    training_accuracy = clf.score(Xtrain, ytrain)
    test_accuracy = clf.score(Xtest, ytest)
    print ("Accuracy on training data: %0.2f" % (training_accuracy))
    print ("Accuracy on test data:     %0.2f" % (test_accuracy))
    print ('Confusion Matrix:')
    print (confusion_matrix(ytest, clf.predict(Xtest)))
    print ('Classification Report:')
    print (clsf_df(ytest, clf.predict(Xtest)))
    #print (classification_report(ytest, clf.predict(Xtest)))
    print ("########################################################")
    return clf



def do_regress(estimator, Xtrain, ytrain, Xtest, ytest, parameters = None, score_func=None, n_folds=5, n_jobs=1):  
    if parameters:
        estimator = cv_optimize(estimator, parameters, Xtrain, ytrain, n_jobs=n_jobs, n_folds=n_folds, score_func=score_func)
    estimator=estimator.fit(Xtrain, ytrain)
    cv_results = cross_val_score(estimator, Xtrain, ytrain, cv = 5)
    ypred = estimator.predict(Xtest)
    rsquared_train = np.mean(cv_results)
    rsquared_test = estimator.score(Xtest,  ytest)
    rmse = np.sqrt(mean_squared_error(ytest, ypred))
    return rsquared_train, rsquared_test, rmse
    
def plot_coef(plot_title, estimator, X_train, y_train):
    names = X_train.columns
    coef = estimator.fit(X_train, y_train).coef_
    plt.figure(figsize=(16,7))
    _ = plt.plot(range(len(names)), coef)
    _ = plt.xticks(range(len(names)), names, rotation = 90)
    _ = plt.ylabel('Coefficients')
    _ = plt.title(plot_title)
    plt.show()

def plot_feature_importance(clf, name_list, chart_title):
    importance_list = clf.feature_importances_
    importance_list, name_list = zip(*sorted(zip(importance_list, name_list)))

    plt.figure(figsize = (10, 10))
    plt.barh(range(len(name_list)),importance_list,align='center')
    plt.yticks(range(len(name_list)),name_list)
    plt.ylabel('Features')
    plt.title('Relative importance of Each Feature in %s' %(chart_title))
    plt.show()

def make_roc(name, clf, ytest, xtest, ax=None, labe=5, proba=True, skip=0):
    initial=False
    if not ax:
        ax=plt.gca()
        initial=True
    if proba:
        fpr, tpr, thresholds=roc_curve(ytest, clf.predict_proba(xtest)[:,1])
    else:
        fpr, tpr, thresholds=roc_curve(ytest, clf.decision_function(xtest))
    roc_auc = auc(fpr, tpr)
    if skip:
        l=fpr.shape[0]
        ax.plot(fpr[0:l:skip], tpr[0:l:skip], '.-', alpha=0.3, label='ROC curve for %s (area = %0.2f)' % (name, roc_auc))
    else:
        ax.plot(fpr, tpr, '.-', alpha=0.3, label='ROC curve for %s (area = %0.2f)' % (name, roc_auc))
    label_kwargs = {}
    label_kwargs['bbox'] = dict(
        boxstyle='round,pad=0.3', alpha=0.2,
    )
    for k in range(0, fpr.shape[0],labe):
        #from https://gist.github.com/podshumok/c1d1c9394335d86255b8
        threshold = str(np.round(thresholds[k], 2))
        ax.annotate(threshold, (fpr[k], tpr[k]), **label_kwargs)
    if initial:
        ax.plot([0, 1], [0, 1], 'k--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC')
    ax.legend(loc="lower right")
    return ax

# # ** CLEAN DATA FROM HERE **

# In[ ]:


# Import cleaned dataset (called df_dummy)
df_dummy = pd.read_excel('Data/Cleaned Dataset for KDD Data Challenge.xlsx')


# # IV. Feature selection

# In[38]:


# Assign X as a DataFrame of all features, and y_donate as a series of the outcome variable (whether donate or not), 
# and y_amount as a series of the outcome variable (amount of donation)
X = df_dummy.drop(['TARGET_B', 'TARGET_D', 'Donor'], axis = 1)
y_donate = df_dummy['TARGET_B']
y_amount = df_dummy['TARGET_D']
X.shape, y_donate.shape, y_amount.shape


# ### IV-1. Normalize the dataset

# In[39]:


# Standardize the data to prepare for variance threshold feature selection
minmax_scaler = MinMaxScaler()
minmax_scaler.fit(X)
X_scaled = pd.DataFrame(minmax_scaler.transform(X), columns = X.columns)
#X_scaled.describe()


# In[40]:


# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y_donate, train_size=0.70, random_state=123)
X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y_donate, train_size=0.70, random_state=123)

print (X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print (X_train_scaled.shape, X_test_scaled.shape, y_train.shape, y_test.shape)


# ### IV-2. Feature Selection Using Variance Threshold

# In[41]:


# Remove all features whose variance are less than 0.05
var_selector = VarianceThreshold(threshold = 0.05)
var_selector.fit(X_train_scaled)
indices_selected = var_selector.get_support(indices=True)
colnames_vtselected = [X_train_scaled.columns[i] for i in indices_selected]
print(colnames_vtselected)
len(colnames_vtselected)


# ### IV-3. Further Feature Selection using RFECV

# In[42]:


# Specify the model
estimator = LogisticRegression()    # estimator for RFE, select the suitable model 

# Select variables using RFECV
rfe_selector = RFECV(estimator, step=1, cv = 5, n_jobs = -1, scoring = 'roc_auc')
#rfe_selector = RFE(estimator, step = 1)
rfe_selector.fit(X_train[colnames_vtselected], y_train)


# In[43]:


# Show the variables selected by RFECV
rfe_selected = list(zip(rfe_selector.ranking_, rfe_selector.support_, colnames_vtselected))
rfe_selected = pd.DataFrame(rfe_selected, columns = ['Ranking', 'Support', 'Feature'])
rfe_selected.head()


# In[44]:


rfe_selected_var = list(rfe_selected[rfe_selected.Ranking == 1]['Feature'])
print(rfe_selected_var)


# In[45]:


len(rfe_selected_var)


# In[46]:


# Check correlation of the selected variables
_ = plt.figure(figsize = (12, 12))
_ = sns.heatmap(X[rfe_selected_var].corr(), cmap="YlGnBu")
plt.show()


# # V. Task 1: Predict Who Will be Donors
# ### V-1. Logistic Regression

# In[47]:


clflog = LogisticRegression()
parameters = {"C": [0.0001, 0.001, 0.1, 1, 10, 100]}
clflog = do_classify(clflog, parameters, X_train[rfe_selected_var], y_train, X_test[rfe_selected_var], y_test, 
                                            n_jobs = 4, score_func = 'f1')


# ### V-2. Decision Tree

# In[48]:


clftree = tree.DecisionTreeClassifier()

parameters = {"max_depth": [1, 2, 3, 4, 5, 6, 7], 'min_samples_leaf': [1, 2, 3, 4, 5, 6]}
clftree = do_classify(clftree, parameters, X_train[rfe_selected_var], y_train, X_test[rfe_selected_var], y_test, 
                                            n_jobs = 4, score_func = 'f1')


# In[49]:


plot_feature_importance(clftree, rfe_selected_var, 'Decision Tree')


# ### V-3. Random Forest

# In[50]:


clfForest = RandomForestClassifier()

parameters = {"n_estimators": range(1, 20)}
clfForest = do_classify(clfForest, parameters, X_train[rfe_selected_var], y_train, X_test[rfe_selected_var], y_test,
                                                             n_jobs = 4, score_func='f1')


# In[51]:


plot_feature_importance(clfForest, rfe_selected_var, 'Random Forests')


# ### V-4. Naive Bayes

# In[52]:


clfnb=GaussianNB()
clfnb.fit(X_train[rfe_selected_var], y_train)
print (confusion_matrix(y_test, clfnb.predict(X_test[rfe_selected_var])))
print (clsf_df(y_test, clfnb.predict(X_test[rfe_selected_var])))


# ### V-5. Compare all classifiers

# In[53]:


# Put the total results of each classifier into a dataframe for easy comparison
clflog_total_res = clsf_df(y_test, clflog.predict(X_test[rfe_selected_var])).iloc[2]
clftree_total_res = clsf_df(y_test, clftree.predict(X_test[rfe_selected_var])).iloc[2]
clfForest_total_res = clsf_df(y_test, clfForest.predict(X_test[rfe_selected_var])).iloc[2]
clfnb_total_res = clsf_df(y_test, clfnb.predict(X_test[rfe_selected_var])).iloc[2]

all_classifiers_res = pd.concat([clflog_total_res, clftree_total_res, clfForest_total_res, clfnb_total_res], axis = 1)
all_classifiers_res.columns = ['Logistic Regression', 'Decision Tree', 'Random Forests', 'Naive Bayes']
all_classifiers_res


# In[55]:


# Make ROC curve
plt.figure(figsize=  (10, 10))
ax = make_roc("Naive Bayes",clfnb, y_test, X_test[rfe_selected_var], None, labe=200)
make_roc("Decision Tree",clftree, y_test, X_test[rfe_selected_var], ax, labe=200)
make_roc("Logistic Regression",clflog, y_test, X_test[rfe_selected_var], ax,labe=200)
#make_roc("Random Forests",clfForest, y_test, X_test[rfe_selected_var], ax, labe=20)


# # VI. Task 2: Predict Donation Amount 
# ### VI-1. Set Up the Data

# In[56]:


# Get the predicted labels for all samples (both training and test sets) according to Naive Bayes model
predicted = clfnb.predict(X[rfe_selected_var]) #Use predictions from Naive Bayes model
predicted = pd.DataFrame(predicted)
predicted.columns = ['Predicted Donor']
predicted.shape


# In[57]:


# Add y_amount (TARGET_D) back to X
X = pd.concat([X, y_amount], axis = 1)
X.head()


# In[59]:


# Reset index in X_test and predicted df, then concatenate them into X_donor df
X_reset = X.reset_index(drop = True)
predicted_reset = predicted.reset_index(drop = True)
X_reset = pd.concat([X_reset, predicted_reset], axis = 1)
#X_donor = X_donor.loc[X_donor['Predicted Donor'] ==1]
#X_reset.head(10)


# In[60]:


# Select only those who were predicted as '1':
X_donor = X_reset.loc[X_reset['Predicted Donor'] ==1]
X_donor.head()


# In[61]:


X_donor.shape #3481 people predicted to be donors according to Naive Bayes


# In[66]:


# Split above df to train and test sets to use later for regression models
X_train1, X_test1, y_train1, y_test1 = train_test_split(X_donor.drop(['TARGET_D', 'Predicted Donor'], axis = 1), 
                                                        X_donor.TARGET_D,
                                                       test_size = 0.3, random_state = 123)

print(X_train1.shape, X_test1.shape, y_train1.shape, y_test1.shape)


# ### VI-1. Feature Selection Again for Regression Task

# In[67]:


# RFE again to select the best features needed for regression task.
# Specify the model
estimator = LinearRegression()  

# Select variables using RFECV
rfe_selector1 = RFECV(estimator, step=1, cv = 5, n_jobs = -1, scoring = 'r2')
rfe_selector1.fit(X_train1[colnames_vtselected], y_train1) #Use the columns selected by Variance Threshold as in task 1

# Show the variables selected by RFECV
rfe_selected1 = list(zip(rfe_selector1.ranking_, rfe_selector1.support_, colnames_vtselected))
rfe_selected1 = pd.DataFrame(rfe_selected1, columns = ['Ranking', 'Support', 'Feature'])
rfe_selected1.head()


# In[69]:


# Select all the features where ranking = 1
rfe_selected_var1 = list(rfe_selected1[rfe_selected1.Ranking == 1]['Feature'])
print(rfe_selected_var1)


# In[71]:


# Check correlation of the selected variables
_ = plt.figure(figsize = (10, 10))
_ = sns.heatmap(X_donor[rfe_selected_var1].corr(), cmap="YlGnBu")
plt.show()


# ### VI - 3. Linear Regression Model

# In[72]:


lm = LinearRegression()
results_linear = do_regress(lm, X_train1[rfe_selected_var1], y_train1, X_test1[rfe_selected_var1], y_test1)


# ### VI-4. Ridge Regression

# In[73]:


ridge = Ridge()
param_ridge = {'alpha': [0.5, 0.2, 0.1, 0.01, 0.001, 0.0001]}
results_ridge = do_regress(ridge, X_train1[rfe_selected_var1], y_train1, X_test1[rfe_selected_var1], y_test1,
                           parameters=param_ridge)


# ### VI-5. Lasso Regression

# In[74]:


lasso = Lasso()
param_lasso = {'alpha': [0.5, 0.2, 0.1, 0.01, 0.001, 0.0001]}
results_lasso = do_regress(lasso, X_train1[rfe_selected_var1], y_train1, X_test1[rfe_selected_var1], y_test1, 
                           parameters=param_lasso)


# ### VI-6. Elastic Net

# In[75]:


l1_space = np.linspace(0, 1, 10)
param_en = {'l1_ratio': l1_space}
en = ElasticNet()
results_en = do_regress(en, X_train1[rfe_selected_var1], y_train1, X_test1[rfe_selected_var1], y_test1, parameters=param_en)


# ### VI-7. Regression Tree

# In[76]:


param_rt = {"max_depth": [1, 2, 3, 4, 5, 6, 7], 'min_samples_leaf': [4, 5, 6,7,8,9,10]}
regtree = DecisionTreeRegressor()
results_rt = do_regress(regtree, X_train1[rfe_selected_var1], y_train1, X_test1[rfe_selected_var1], y_test1,
                        parameters=param_rt)


# In[88]:


regtree = regtree.fit(X_train1[rfe_selected_var1], y_train1)

importance_list = regtree.feature_importances_
name_list = rfe_selected_var1
importance_list, name_list = zip(*sorted(zip(importance_list, name_list)))

plt.figure(figsize = (7, 7))
plt.barh(range(len(name_list)),importance_list,align='center')
plt.yticks(range(len(name_list)),name_list)
plt.xlabel('Relative Importance in the Random Forest')
plt.ylabel('Features')
plt.title('Relative importance of Each Feature - Regression Tree')
plt.show()


# ### VI-8. Compare Results of All Models

# In[89]:


# Create a dataframe to compare results of all models
results_all = list(zip(results_linear, results_ridge, results_lasso, results_en, results_rt))
results_all = pd.DataFrame(results_all, columns = ['Linear Regression', 'Ridge', 'Lasso', 'Elastic Net', 'Regression Tree'],
                          index = ['R-squared (training)', 'R-squared (test)', 'Root Mean Squared Error'])
results_all


# In[82]:


# Plot coefficients for Linear Regression
lm = LinearRegression()
_ = plot_coef('Linear Regression', lm, X_train1[rfe_selected_var1], y_train1)
plt.show()


# In[83]:


# Plot coefficients for RIdge Regression
ridge = Ridge(alpha = 0.5)
_ = plot_coef('Ridge Regression', ridge, X_train1[rfe_selected_var1], y_train1)
plt.show()


# In[84]:


# Plot coefficients for Lasso Regression
lasso = Lasso(alpha = 0.01)
_ = plot_coef('Lasso Regression', lasso, X_train1[rfe_selected_var1], y_train1)
plt.show()


# In[85]:


# Plot coefficients for Elastic Net
en = ElasticNet(l1_ratio= 0)
_ = plot_coef('Elastic Net', en, X_train1[rfe_selected_var1], y_train1)
plt.show()

