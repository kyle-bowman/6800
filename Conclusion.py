# -*- coding: utf-8 -*-
"""
Created on Sun Mar 29 04:44:05 2020

@author: Owner
"""

### Path to data
PATH=r'C:\Users\Owner\Desktop\6800\machine-learning'

### Standards
import pandas as pd
import numpy as np
import copy
import joblib
import matplotlib.pyplot as plt

### Graph for Decision Tree
from six import StringIO
from pydotplus import graph_from_dot_data 
### Selection Methodology
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold
### Accuracy Measure, this is what we are looking to maximize
from sklearn.metrics import make_scorer,accuracy_score,roc_curve,auc
### Decision Tree
from sklearn.tree import DecisionTreeClassifier,export_graphviz
### Random Forest
from sklearn.ensemble import RandomForestClassifier
### KNN
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier,NeighborhoodComponentsAnalysis
### Naive Bayes
from sklearn.naive_bayes import MultinomialNB,CategoricalNB,BernoulliNB,GaussianNB,ComplementNB
### Support Vector Machine
from sklearn.svm import SVC
### Neural Network
from sklearn.neural_network import MLPClassifier
### Linear Models
from sklearn.linear_model import LogisticRegression,LinearRegression

#%%
##############################################################
#------------------------------------------------------------- 
# Data Cleaning
#------------------------------------------------------------- 
##############################################################

### Assign the data:

masses_data = pd.read_csv(PATH+'\MLCourse\MLCourse'+ '\\' + 'mammographic_masses.data.txt', 
                          na_values=['?'], 
                          names = ['BI-RADS', 'Age', 'Shape', 'Margin', 'Density', 'Severity'])
masses_data.describe()

masses_data_impute=masses_data.copy(deep=True)

masses_data_impute['missing_age']=np.where(masses_data_impute['Age'].isnull(),1,0)
masses_data_impute['missing_shape']=np.where(masses_data_impute['Shape'].isnull(),1,0)
masses_data_impute['missing_margin']=np.where(masses_data_impute['Margin'].isnull(),1,0)
masses_data_impute['missing_density']=np.where(masses_data_impute['Density'].isnull(),1,0)
masses_data_impute['total_missing']=masses_data_impute[['missing_age','missing_shape','missing_margin','missing_density']].sum(axis=1)
masses_data_impute=masses_data_impute.fillna(value=0,axis=1)
masses_data_impute=masses_data_impute[masses_data_impute['total_missing'] <= 1]

### Normalize BI-RADS for conclusion...
masses_data_impute['BI-RADS']=np.where(masses_data_impute['BI-RADS']==0,1,masses_data_impute['BI-RADS'])
masses_data_impute['BI-RADS']=np.where(masses_data_impute['BI-RADS'] >= 5,5,masses_data_impute['BI-RADS'])
masses_data_impute['Binary BI-RADS']=np.where(masses_data_impute['BI-RADS'] >= 4,1,0)

### Features for the modeling exercise...
X=masses_data_impute[['Age','Shape','Margin','Density','missing_age','missing_shape','missing_margin','missing_density']].values
Y=masses_data_impute['Severity'].values
### for labels, we will store these in a list
X_names=['Age', 'Shape', 'Margin', 'Density','missing_age','missing_shape','missing_margin','missing_density']

np.random.seed(1234) ### set seed for repetition

### Split the data into training and testing, 6/40 split
(X_train,X_test,Y_train,Y_test)=train_test_split(X,Y,train_size=0.60,test_size=0.4,random_state=1) 

scoring_data=masses_data_impute.copy(deep=True)
scoring_data.reset_index(inplace=True,drop=True)

#%%
##############################################################
#------------------------------------------------------------- 
# Decision Tree for Classification
#------------------------------------------------------------- 
##############################################################

DT_gs=GridSearchCV(DecisionTreeClassifier(random_state=1),
                   param_grid={'criterion':['gini','entropy'],
                               'max_depth':list(range(1,20)),
                               'min_samples_split':list(range(2,20)),
                               'min_samples_leaf':list(range(1,20)),
                               #'min_impurity_decrease':np.arange(0.00,1.0,0.01),
                               #'max_features':['auto','sqrt','log2',None],
                               'splitter':['random','best']},
                           
                   cv=5,
                   scoring=make_scorer(accuracy_score),
                   n_jobs=-1)

DT_gs.fit(X,Y)
joblib.dump(DT_gs,PATH+'\\models\\DT.sav')

#%%
##############################################################
#------------------------------------------------------------- 
# Random Forest for Classification
#------------------------------------------------------------- 
##############################################################

RF_gs=GridSearchCV(RandomForestClassifier(random_state=1),
                   param_grid={'criterion':['gini','entropy'],
                               'max_depth':list(range(1,20)),
                               'min_samples_split':list(range(2,20)),
                               'min_samples_leaf':list(range(1,20)),
                               'oob_score':['True','False']},
                   cv=5,
                   scoring=make_scorer(accuracy_score),
                   n_jobs=-1)

RF_gs.fit(X,Y)
joblib.dump(RF_gs,PATH+'\\models\\RF.sav')

#%%
##############################################################
#------------------------------------------------------------- 
# Nearest Neighbors
#------------------------------------------------------------- 
##############################################################

### Standard KNN
KNN_gs=GridSearchCV(KNeighborsClassifier(),
                    param_grid={'algorithm':['auto','ball_tree','kd_tree','brute'],
                               'weights':['uniform','distance'],
                               'n_neighbors':list(range(1,40)),
                               'leaf_size':list(range(10,100,10))
                               },
                    cv=5,
                    scoring=make_scorer(accuracy_score),
                    n_jobs=-1)

### KNN with Principal Component Analysis
KNN_PCA_gs=GridSearchCV(Pipeline(steps=[('pca',NeighborhoodComponentsAnalysis(random_state=1)),
                                        ('KNN',KNeighborsClassifier())]),
                        param_grid={'KNN__algorithm':['auto','ball_tree','kd_tree','brute'],
                                    'KNN__weights':['uniform','distance'],
                                    'KNN__n_neighbors':list(range(1,40)),
                                    'KNN__leaf_size':list(range(10,100,10))
                                    },
                        cv=5,
                        scoring=make_scorer(accuracy_score),
                        n_jobs=-1)

KNN_gs.fit(X,Y)
joblib.dump(KNN_gs,PATH+'\\models\\KNN.sav')

KNN_PCA_gs.fit(X,Y)
joblib.dump(KNN_PCA_gs,PATH+'\\models\\KNN_PCA.sav')

#%%

##############################################################
#------------------------------------------------------------- 
# Niave Bayes
#------------------------------------------------------------- 
##############################################################

### Catagorrical, Multinomial, Complement can be ran using same parameters
### Gausian will be independent, Bernoulli will be independent.

#NB_CAT_gs=GridSearchCV(CategoricalNB(),param_grid={'alpha':[.45,.5]},cv=5)
#                       param_grid={'alpha':np.arange(0.01,1.0,0.01)},cv=5)                       
#                       ,scoring=make_scorer(accuracy_score),n_jobs=-1)
#NB_CAT_gs=CategoricalNB(alpha=0.01)
NB_GAUS_gs=GridSearchCV(GaussianNB(),param_grid={'var_smoothing':np.arange(0.01,1.01,0.01)},cv=5,scoring=make_scorer(accuracy_score),n_jobs=-1)
NB_MN_gs=GridSearchCV(MultinomialNB(),param_grid={'alpha':np.arange(0.01,1.01,0.01)},cv=5,scoring=make_scorer(accuracy_score),n_jobs=-1)
NB_COM_gs=GridSearchCV(ComplementNB(),param_grid={'alpha':np.arange(0.01,1.01,0.01)},cv=5,scoring=make_scorer(accuracy_score),n_jobs=-1)
NB_BER_gs=GridSearchCV(BernoulliNB(),param_grid={'alpha':np.arange(0.01,1.01,0.01),'binarize':list(range(1,10))},
                       cv=5,scoring=make_scorer(accuracy_score),n_jobs=-1)

#NB_CAT_gs.fit(X,Y).score(X,Y)
#NB_CAT_cv_results=NB_CAT_gs.cv_results_
#NB_CAT_best=NB_CAT_gs.best_estimator_
#NB_CAT_best_predict=pd.Series(NB_CAT_best.predict(X),name='NB_CAT_Predict')
#NB_CAT_best_predict_prob=pd.Series(NB_CAT_best.predict_proba(X)[:,1],name='NB_CAT_Probability')

NB_GAUS_gs.fit(X,Y)
joblib.dump(NB_GAUS_gs,PATH+'\\models\\NB_GAUS.sav')

NB_MN_gs.fit(X,Y)
joblib.dump(NB_MN_gs,PATH+'\\models\\NB_MN.sav')

NB_COM_gs.fit(X,Y)
joblib.dump(NB_COM_gs,PATH+'\\models\\NB_COM.sav')

NB_BER_gs.fit(X,Y)
joblib.dump(NB_BER_gs,PATH+'\\models\\NB_BER.sav')

#%%

##############################################################
#------------------------------------------------------------- 
# SVM
#------------------------------------------------------------- 
##############################################################

SVM_LINEAR_gs=GridSearchCV(SVC(random_state=1,kernel='linear',probability=True),param_grid={'C':list(range(1,100,5))},
                           cv=5,scoring=make_scorer(accuracy_score),n_jobs=-1)
SVM_RBF_gs=GridSearchCV(SVC(random_state=1,kernel='rbf',probability=True),param_grid={'C':list(range(1,100,5)),'gamma':['scale','auto']},
                        cv=5,scoring=make_scorer(accuracy_score),n_jobs=-1)
#SVM_POLY_gs=GridSearchCV(SVC(kernel='poly',random_state=1,probability=True),param_grid={'C':list(range(1,100,5)),
#                                                                                        'degree':[2,4,5,6,7],
#                                                                                        'gamma':['scale','auto']},
#                         cv=5,scoring=make_scorer(accuracy_score),n_jobs=-1)

SVM_LINEAR_gs.fit(X,Y)
joblib.dump(SVM_LINEAR_gs,PATH+'\\models\\SVM_Linear.sav')

SVM_RBF_gs.fit(X,Y)
joblib.dump(SVM_RBF_gs,PATH+'\\models\\SVM_RBF.sav')

#SVM_POLY_gs.fit(X,Y)
#SVM_PLOY_cv_results=SVM_POLY_gs.cv_results_
#SVM_POLY_best=SVM_POLY_gs.best_estimator_
#SVM_POLY_best_predict=pd.Series(SVM_POLY_best.predict(X),name='SVM_Poly_Predict')
#SVM_POLY_best_predict_prob=pd.Series(SVM_POLY_best.predict_proba(X)[:,1],name='SVM_Poly_Probability')

#%%

##############################################################
#------------------------------------------------------------- 
# Nueral Network
#------------------------------------------------------------- 
##############################################################

NN_LBFGS_gs=GridSearchCV(MLPClassifier(random_state=1,activation='logistic',solver='lbfgs'),
                         param_grid={'alpha':np.arange(0.0001,0.5,0.0005)},
                         cv=5,
                         scoring=make_scorer(accuracy_score),
                         n_jobs=-1
                         )
NN_SGD_gs=GridSearchCV(MLPClassifier(random_state=1,activation='logistic',solver='sgd',warm_start=True),
                       param_grid={'alpha':np.arange(0.0001,0.5,0.0005),
                                   'learning_rate':['constant','invscaling','adaptive'],
                                   'momentum':np.arange(0.1,1,0.1)
                                   },
                       cv=5,
                       scoring=make_scorer(accuracy_score),
                       n_jobs=-1
                       )

NN_LBFGS_gs.fit(X,Y)
joblib.dump(NN_LBFGS_gs,PATH+'\\models\\NN_LBFGS.sav')

NN_SGD_gs.fit(X,Y)
joblib.dump(NN_SGD_gs,PATH+'\\models\\NN_SGD.sav')

#%%

##############################################################
#------------------------------------------------------------- 
# Logistic Regression
#------------------------------------------------------------- 
##############################################################

LOGIT_gs=GridSearchCV(LogisticRegression(random_state=1),
                      param_grid={'C':np.arange(0.001,2,0.001),
                                  'fit_intercept':[True,False]},
                      cv=5,
                      scoring=make_scorer(accuracy_score),
                      n_jobs=-1)

LOGIT_gs.fit(X,Y)
joblib.dump(LOGIT_gs,PATH+'\\models\\LOGIT.sav')

#%%

##############################################################
#------------------------------------------------------------- 
# Linear Probability
#------------------------------------------------------------- 
##############################################################

def OLS_accuracy(y_true,y_predict):
    return accuracy_score(y_true,[round(y) for y in y_predict])

OLS_gs=GridSearchCV(LinearRegression(),
                    param_grid={'normalize':[True,False],
                                'fit_intercept':[True,False]},
                    cv=5,
                    scoring=make_scorer(OLS_accuracy,greater_is_better=True),
                    n_jobs=-1)
OLS_gs.fit(X,Y)
joblib.dump(OLS_gs,PATH+'\\models\\OLS.sav')

#%%

##############################################################
#------------------------------------------------------------- 
# Load Results from disk...
#------------------------------------------------------------- 
##############################################################

### Defined how K-Folds works
kfold_data=pd.DataFrame({'Outcome':[0,0,0,1,1,1],'Predictor':[0.1,0.2,0.3,0.4,0.5,0.6]},columns=['Outcome','Predictor'])
k_fold=StratifiedKFold(n_splits=3)
#k_fold.get_n_splits(kfold_data['Predictor'],kfold_data['Outcome'])
for train,test in k_fold.split(kfold_data['Predictor'],kfold_data['Outcome']):
    print('Train: ',train,' Test:',test)
   
### Decision Tree
DT_gs=joblib.load(PATH+'\\models\\DT.sav')
DT_cv_results=DT_gs.cv_results_
DT_best=DT_gs.best_estimator_
DT_best_predict=pd.Series(DT_best.predict(X),name='DT_Predict')
DT_best_predict_prob=pd.Series(DT_best.predict_proba(X)[:,1],name='DT_Probability')
dir(DT_best.get_params)
print(DT_best.get_params)

### Random Forest
RF_gs=joblib.load(PATH+'\\models\\RF.sav')
RF_cv_results=RF_gs.cv_results_
RF_best=RF_gs.best_estimator_
RF_best_predict=pd.Series(RF_best.predict(X),name='RF_Predict')
RF_best_predict_prob=pd.Series(RF_best.predict_proba(X)[:,1],name='RF_Probability')

### Nearest Neighbor
KNN_gs=joblib.load(PATH+'\\models\\KNN.sav')
KNN_cv_results=KNN_gs.cv_results_
KNN_best=KNN_gs.best_estimator_
KNN_best_predict=pd.Series(KNN_best.predict(X),name='KNN_Predict')
KNN_best_predict_prob=pd.Series(KNN_best.predict_proba(X)[:,1],name='KNN_Probability')

KNN_PCA_gs=joblib.load(PATH+'\\models\\KNN_PCA.sav')
KNN_PCA_cv_results=KNN_PCA_gs.cv_results_
KNN_PCA_best=KNN_PCA_gs.best_estimator_
KNN_PCA_best_predict=pd.Series(KNN_PCA_best.predict(X),name='KNN_PCA_Predict')
KNN_PCA_best_predict_prob=pd.Series(KNN_PCA_best.predict_proba(X)[:,1],name='KNN_PCA_Probability')

### Naive Bayes
NB_GAUS_gs=joblib.load(PATH+'\\models\\NB_GAUS.sav')
NB_GAUS_cv_results=NB_GAUS_gs.cv_results_
NB_GAUS_best=NB_GAUS_gs.best_estimator_
NB_GAUS_best_predict=pd.Series(NB_GAUS_best.predict(X),name='NB_GAUS_Predict')
NB_GAUS_best_predict_prob=pd.Series(NB_GAUS_best.predict_proba(X)[:,1],name='NB_GAUS_Probability')

NB_MN_gs=joblib.load(PATH+'\\models\\NB_MN.sav')
NB_MN_cv_results=NB_MN_gs.cv_results_
NB_MN_best=NB_MN_gs.best_estimator_
NB_MN_best_predict=pd.Series(NB_MN_best.predict(X),name='NB_MN_Predict')
NB_MN_best_predict_prob=pd.Series(NB_MN_best.predict_proba(X)[:,1],name='NB_MN_Probability')

NB_COM_gs=joblib.load(PATH+'\\models\\NB_COM.sav')
NB_COM_cv_results=NB_COM_gs.cv_results_
NB_COM_best=NB_COM_gs.best_estimator_
NB_COM_best_predict=pd.Series(NB_MN_best.predict(X),name='NB_COM_Predict')
NB_COM_best_predict_prob=pd.Series(NB_COM_best.predict_proba(X)[:,1],name='NB_COM_Probability')

NB_BER_gs=joblib.load(PATH+'\\models\\NB_BER.sav')
NB_BER_cv_results=NB_BER_gs.cv_results_
NB_BER_best=NB_BER_gs.best_estimator_
NB_BER_best_predict=pd.Series(NB_BER_best.predict(X),name='NB_BER_Predict')
NB_BER_best_predict_prob=pd.Series(NB_BER_best.predict_proba(X)[:,1],name='NB_BER_Probability')

### SVM
SVM_LINEAR_gs=joblib.load(PATH+'\\models\\SVM_Linear.sav')
SVM_LINEAR_cv_results=SVM_LINEAR_gs.cv_results_
SVM_LINEAR_best=SVM_LINEAR_gs.best_estimator_
SVM_LINEAR_best_predict=pd.Series(SVM_LINEAR_best.predict(X),name='SVM_Linear_Predict')
SVM_LINEAR_best_predict_prob=pd.Series(SVM_LINEAR_best.predict_proba(X)[:,1],name='SVM_Linear_Probability')

SVM_RBF_gs=joblib.load(PATH+'\\models\\SVM_RBF.sav')
SVM_RBF_cv_results=SVM_RBF_gs.cv_results_
SVM_RBF_best=SVM_RBF_gs.best_estimator_
SVM_RBF_best_predict=pd.Series(SVM_RBF_best.predict(X),name='SVM_RBF_Predict')
SVM_RBF_best_predict_prob=pd.Series(SVM_RBF_best.predict_proba(X)[:,1],name='SVM_RBF_Probability')

### Neural Network
NN_LBFGS_gs=joblib.load(PATH+'\\models\\NN_LBFGS.sav')
NN_LBFGS_cv_results=NN_LBFGS_gs.cv_results_
NN_LBFGS_best=NN_LBFGS_gs.best_estimator_
NN_LBFGS_best_predict=pd.Series(NN_LBFGS_best.predict(X),name='NN_LBFGS_Predict')
NN_LBFGS_best_predict_prob=pd.Series(NN_LBFGS_best.predict_proba(X)[:,1],name='NN_LBFGS_Probability')

NN_SGD_gs=joblib.load(PATH+'\\models\\NN_SGD.sav')
NN_SGD_cv_results=NN_SGD_gs.cv_results_
NN_SGD_best=NN_SGD_gs.best_estimator_
NN_SGD_best_predict=pd.Series(NN_SGD_best.predict(X),name='NN_SGD_Predict')
NN_SGD_best_predict_prob=pd.Series(NN_SGD_best.predict_proba(X)[:,1],name='NN_SGD_Probability')

### Logistic Regression
LOGIT_gs=joblib.load(PATH+'\\models\\LOGIT.sav')
LOGIT_cv_results=LOGIT_gs.cv_results_
LOGIT_best=LOGIT_gs.best_estimator_
LOGIT_best_predict=pd.Series(LOGIT_best.predict(X),name='LOGIT_Predict')
LOGIT_best_predict_prob=pd.Series(LOGIT_best.predict_proba(X)[:,1],name='LOGIT_Probability')

### Linear Probability
OLS_gs=joblib.load(PATH+'\\models\\OLS.sav')
OLS_cv_results=OLS_gs.cv_results_
OLS_best=OLS_gs.best_estimator_
OLS_best_predict=pd.Series([round(y) for y in OLS_best.predict(X)],name='OLS_Predict')
OLS_best_predict_prob=pd.Series(OLS_best.predict(X),name='OLS_Probability')

dir(OLS_best)
OLS_best.intercept_
[('Intercept',OLS_best.intercept_)]+list(zip(X_names,OLS_best.coef_))

print('Decision Tree, best model: ',str(DT_best.score(X,Y)))
print('Random Forest, best model: ',str(RF_best.score(X,Y)))
print('KNN, best model: ',str(KNN_best.score(X,Y)))
print('KNN (PCA), best model: ',str(KNN_PCA_best.score(X,Y)))
#print('NB Categorical, best model: ',str(NB_CAT_best.score(X,Y)))
print('NB Gaussian, best model: ',str(NB_GAUS_best.score(X,Y)))
print('NB Multinomial, best model: ',str(NB_MN_best.score(X,Y)))
print('NB Complement, best model: ',str(NB_COM_best.score(X,Y)))
print('NB Bernoulli, best model: ',str(NB_BER_best.score(X,Y)))
print('SVM Linear, best model: ',str(SVM_LINEAR_best.score(X,Y)))
print('SVM RBF, best model: ',str(SVM_RBF_best.score(X,Y)))
#print('SVM POLY, best model: ',str(SVM_POLY_best.score(X,Y)))
print('NN LBFGS, best model: ',str(NN_LBFGS_best.score(X,Y)))
print('NN SGD, best model: ',str(NN_SGD_best.score(X,Y)))
print('Logit, best model: ',str(LOGIT_best.score(X,Y)))
print('OLS, best model: ',str(accuracy_score(Y,[round(y) for y in OLS_best.predict(X)])))

scoring_data=pd.concat([scoring_data,DT_best_predict,DT_best_predict_prob,
                        RF_best_predict,RF_best_predict_prob,
                        KNN_best_predict,KNN_best_predict_prob,
                        KNN_PCA_best_predict,KNN_PCA_best_predict_prob,
                        #NB_CAT_best_predict,NB_CAT_best_predict_prob,
                        NB_GAUS_best_predict,NB_GAUS_best_predict_prob,
                        NB_MN_best_predict,NB_MN_best_predict_prob,
                        NB_COM_best_predict,NB_COM_best_predict_prob,
                        NB_BER_best_predict,NB_BER_best_predict_prob,
                        SVM_LINEAR_best_predict,SVM_LINEAR_best_predict_prob,
                        SVM_RBF_best_predict,SVM_RBF_best_predict_prob,
                        #SVM_POLY_best_predict,SVM_POLY_best_predict_prob
                        NN_LBFGS_best_predict,NN_LBFGS_best_predict_prob,
                        NN_SGD_best_predict,NN_SGD_best_predict_prob,
                        LOGIT_best_predict,LOGIT_best_predict_prob,
                        OLS_best_predict,OLS_best_predict_prob
                        ],axis=1)


### Bar Plot of model fit...

model_times=[('Decision Tree',sum(DT_cv_results['mean_fit_time'])),
             ('Naive Bayes',sum(NB_GAUS_cv_results['mean_fit_time'])+sum(NB_MN_cv_results['mean_fit_time'])+ \
                           sum(NB_COM_cv_results['mean_fit_time']),+sum(NB_BER_cv_results['mean_fit_time'])),
             ('Random Forest',sum(RF_cv_results['mean_fit_time'])),
             ('KNN',sum(KNN_cv_results['mean_fit_time'])+sum(KNN_PCA_cv_results['mean_fit_time'])),
             ('SVM',sum(SVM_LINEAR_cv_results['mean_fit_time'])+sum(SVM_RBF_cv_results['mean_fit_time'])),
             ('Neural Network',sum(NN_LBFGS_cv_results['mean_fit_time'])+sum(NN_SGD_cv_results['mean_fit_time'])),
             ('Logistic Regression',sum(LOGIT_cv_results['mean_fit_time'])),
             ('Linear Probability',sum(OLS_cv_results['mean_fit_time']))
             ]

ax=plt.barh(list(zip(*model_times))[0],list(zip(*model_times))[1])
time_values=[round(x.get_width(),2) for x in ax.patches]
for index,value in enumerate(time_values):
    plt.text(value,index,str(value))
plt.xticks(range(0,31000,5000))
plt.title('Total Time in Seconds used for GridSearchCV')

plt.savefig(PATH + '\\outputs\\Model_Fit_Times.png',bbox_inches='tight')
plt.close('all')   

### Bar Plot of best model fit...
best_model_times=[('Decision Tree',DT_cv_results['mean_fit_time'][DT_gs.best_index_]),
                  ('Naive Bayes (Gaussian)',NB_GAUS_cv_results['mean_fit_time'][NB_GAUS_gs.best_index_]),
                  ('Naive Bayes (Multinomial)',NB_MN_cv_results['mean_fit_time'][NB_MN_gs.best_index_]),
                  ('Naive Bayes (Complement)',NB_COM_cv_results['mean_fit_time'][NB_COM_gs.best_index_]),
                  ('Naive Bayes (Bernoulli)',NB_BER_cv_results['mean_fit_time'][NB_BER_gs.best_index_]),
                  ('Random Forest',RF_cv_results['mean_fit_time'][RF_gs.best_index_]),
                  ('KNN',KNN_cv_results['mean_fit_time'][KNN_gs.best_index_]),
                  ('KNN (PCA)',KNN_PCA_cv_results['mean_fit_time'][KNN_PCA_gs.best_index_]),
                  ('SVM (Linear)',SVM_LINEAR_cv_results['mean_fit_time'][SVM_LINEAR_gs.best_index_]),
                  ('SVM (RBF)',SVM_RBF_cv_results['mean_fit_time'][SVM_RBF_gs.best_index_]),                  
                  ('Neural Network (LBFGS)',NN_LBFGS_cv_results['mean_fit_time'][NN_LBFGS_gs.best_index_]),
                  ('Neural Network (SGD)',NN_SGD_cv_results['mean_fit_time'][NN_SGD_gs.best_index_]),
                  ('Logistic Regression',LOGIT_cv_results['mean_fit_time'][LOGIT_gs.best_index_]),
                  ('Linear Probability',OLS_cv_results['mean_fit_time'][OLS_gs.best_index_])
             ]
ax=plt.barh(list(zip(*best_model_times))[0],list(zip(*best_model_times))[1])
time_values=[round(x.get_width(),2) for x in ax.patches]
for index,value in enumerate(time_values):
    plt.text(value,index,str(value))
plt.xticks(range(0,31,5))
plt.title('Total Time in Seconds used for Best Model')

plt.savefig(PATH + '\\outputs\\Model_Fit_Times_Best.png',bbox_inches='tight')
plt.close('all')  

### ROC Plot
ROC_dict={'fpr':[],'tpr':[],'auc':[]}
model_list=[(DT_best_predict_prob,'Decision Tree','darkcyan'),
            (RF_best_predict_prob,'Random Forest','darkorange'),
            (KNN_best_predict_prob,'KNN','darksalmon'),
            (KNN_PCA_best_predict_prob,'KNN (PCA)','darkviolet'),
            (NB_GAUS_best_predict_prob,'NB (Gaussian)','red'),
            (NB_MN_best_predict_prob,'NB Multinomial','darkcyan'),
            (NB_COM_best_predict_prob,'NB Compliment','gold'),
            (NB_BER_best_predict_prob,'NB (Bernoulli)','deeppink'),
            (SVM_LINEAR_best_predict_prob,'SVM Linear','gray'),
            (SVM_RBF_best_predict_prob,'SVM RBF','khaki'),
            (NN_LBFGS_best_predict_prob,'Neural Network (LBFGS)','green'),
            (NN_SGD_best_predict_prob,'Neural Network (SGD)','blue'),
            (LOGIT_best_predict_prob,'Logit','hotpink'),
            (OLS_best_predict_prob,'Linear Probability','aqua')
            ]

for e,i in enumerate(model_list,start=0):
    _fpr,_tpr,_=roc_curve(Y,i[0])
    _roc_auc=auc(_fpr,_tpr)
    ROC_dict['fpr'].append(_fpr)
    ROC_dict['tpr'].append(_tpr)
    ROC_dict['auc'].append(_roc_auc)
    plt.plot(ROC_dict['fpr'][e],ROC_dict['tpr'][e],label=i[1]+' (area=%0.2f)' % ROC_dict['auc'][e],color=i[2])
    del _fpr,_tpr,_roc_auc

plt.plot([0,1],[0,1],color='navy',lw=2,linestyle='--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristics')
plt.legend(loc='center',bbox_to_anchor=(0.5,-0.4),ncol=2)

plt.savefig(PATH + '\\outputs\\ROC.png',bbox_inches='tight')
plt.close('all') 

##############################################################
#------------------------------------------------------------- 
# Conclusion...
#------------------------------------------------------------- 
##############################################################

print(pd.crosstab(scoring_data['Severity'],scoring_data['BI-RADS']))

### 2 models choosen for simplilcity of interpretation...

example_obs=pd.DataFrame(scoring_data.iloc[57]).T
### Decision Tree
dot_data=StringIO()
export_graphviz(DT_best,out_file=dot_data,feature_names=X_names,class_names=['Benign','Maalignant',])
DT_graph=graph_from_dot_data(dot_data.getvalue())
DT_graph.write_png(PATH+'\\outputs\\Conclusion_Decision_Tree.png')

### OLS Example Score for conclusion
ols_coef={k:v for k,v in [('Intercept',OLS_best.intercept_)]+list(zip(X_names,OLS_best.coef_))}
ols_score=ols_coef['Intercept']+ \
    ols_coef['Age']*example_obs['Age']+ \
    ols_coef['Shape']*example_obs['Shape']+ \
    ols_coef['Margin']*example_obs['Margin']+ \
    ols_coef['Density']*example_obs['Density']

### Logistic Example Score for conclusion
logit_coef={k:v for k,v in [('Intercept',LOGIT_best.intercept_[0])]+list(zip(X_names,LOGIT_best.coef_[0]))}
logit_score=1/(1+np.exp(-(logit_coef['Age']*example_obs['Age']+ \
                          logit_coef['Shape']*example_obs['Shape']+ \
                          logit_coef['Margin']*example_obs['Margin']+ \
                          logit_coef['Density']*example_obs['Density']+ \
                          logit_coef['Intercept'])))


print(pd.crosstab(scoring_data['Severity'],scoring_data['DT_Predict']))
print(pd.qcut(scoring_data['DT_Probability'],q=[0.0,0.2,0.4,0.6,0.8,1.0],labels=[1,2,3,4,5]))

#print(pd.crosstab(scoring_data['Severity'],pd.qcut(scoring_data['DT_Probability'],q=[0.0,0.2,0.4,0.6,0.8,1.0],labels=[1,2,3,4,5])))
print(pd.crosstab(scoring_data['Severity'],pd.cut(scoring_data['DT_Probability'],bins=[0.0,0.2,0.4,0.6,0.8,1.0],labels=[1,2,3,4,5],include_lowest=True)))

#print(pd.crosstab(scoring_data['Severity'],pd.qcut(scoring_data['LOGIT_Probability'],q=[0.0,0.2,0.4,0.6,0.8,1.0],labels=[1,2,3,4,5])))
print(pd.crosstab(scoring_data['Severity'],pd.cut(scoring_data['LOGIT_Probability'],bins=[0.0,0.2,0.4,0.6,0.8,1.0],labels=[1,2,3,4,5],include_lowest=True)))

#print(pd.crosstab(scoring_data['Severity'],pd.qcut(scoring_data['OLS_Probability'],q=[0.0,0.2,0.4,0.6,0.8,1.0],labels=[1,2,3,4,5])))
print(pd.crosstab(scoring_data['Severity'],pd.cut(scoring_data['OLS_Probability'],bins=[-1.5,0.2,0.4,0.6,0.8,1.5],labels=[1,2,3,4,5],include_lowest=True)))
#print(pd.crosstab(scoring_data['Severity'],pd.cut(scoring_data['RF_Probability'],q=[0.0,0.2,0.4,0.6,0.8,1.0],labels=[1,2,3,4,5])))
print(pd.crosstab(scoring_data['Severity'],pd.cut(scoring_data['RF_Probability'],bins=[0.0,0.2,0.4,0.6,0.8,1.0],labels=[1,2,3,4,5],include_lowest=True)))
print(pd.crosstab(scoring_data['Severity'],pd.cut(scoring_data['KNN_Probability'],bins=[0.0,0.2,0.4,0.6,0.8,1.0],labels=[1,2,3,4,5],include_lowest=True)))

print(pd.crosstab(scoring_data['Severity'],scoring_data['DT_Predict']))
print(pd.crosstab(scoring_data['Severity'],scoring_data['LOGIT_Predict']))
print(pd.crosstab(scoring_data['Severity'],scoring_data['OLS_Predict']))
print(pd.crosstab(scoring_data['Severity'],scoring_data['Binary BI-RADS']))

from sklearn.inspection import plot_partial_dependence, partial_dependence
meow=plot_partial_dependence(LOGIT_best,X,Y)

from statsmodels.api import Logit,OLS
from statsmodels.tools.tools import add_constant
OLS(scoring_data['Severity'],
    add_constant(scoring_data[['Age','Shape','Margin','Density','missing_age','missing_shape','missing_margin','missing_density']])).fit().summary()
Logit(scoring_data['Severity'],
      add_constant(scoring_data[['Age','Shape','Margin','Density','missing_age','missing_shape','missing_margin','missing_density']])).fit().summary()

Logit(scoring_data['Severity'],
      add_constant(scoring_data[['Age','Shape','Margin','Density','missing_age','missing_shape','missing_margin','missing_density']])).fit().get_margeff().summary()



