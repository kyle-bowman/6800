
#%%

### Path to data
PATH=r'C:\Users\Owner\Desktop\6800\machine-learning'

### Standards
import pandas as pd
import numpy as np
from collections import OrderedDict
import gc
### modeling
from sklearn import preprocessing
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.externals.six import StringIO  
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import neighbors
from sklearn.naive_bayes import MultinomialNB,CategoricalNB,BernoulliNB,GaussianNB,ComplementNB
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn import neighbors
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.exceptions import ConvergenceWarning
### Reporting
from pydotplus import graph_from_dot_data 
from IPython.display import Image 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
### Deep Learning
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer,accuracy_score

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

### Report out total descriptive statistics
simple=masses_data.describe()
missing=masses_data.isnull().sum().to_frame(name='missing').T
simple=pd.concat([simple,missing]).to_html()
text_file=open(PATH+'\\outputs\\'+"Simple_Stats.html",'w')
text_file.write(simple)
text_file.close()

### Report out total descriptive statiscs by severity
severity_simple=masses_data.groupby('Severity').describe()
severity_missing=masses_data.drop('Severity',1).isna().groupby(masses_data.Severity,sort=True).sum()
text_file=open(PATH+'\\outputs\\'+"Severity_Simple_Stats.html",'w')
#text_file.write(severity_simple.to_html())
text_file.write('<h3>BI-RADS</h3>')
text_file.write(severity_simple['BI-RADS'].to_html())
text_file.write('<h3>Age</h3>')
text_file.write(severity_simple['Age'].to_html())
text_file.write('<h3>Shape</h3>')
text_file.write(severity_simple['Shape'].to_html())
text_file.write('<h3>Margin</h3>')
text_file.write(severity_simple['Margin'].to_html())
text_file.write('<h3>Density</h3>')
text_file.write(severity_simple['Density'].to_html())
text_file.write('<h3>Missing</h3>')
text_file.write(severity_missing.to_html())
text_file.close()

### Replace missing values with 0 and create a column to flag the event was imputed
masses_data_impute=masses_data.fillna(value=0,axis=0)
### Get minimum value
masses_data_impute['missing']=masses_data_impute[['Age','Shape','Margin','Density']].min(axis=1)
### Assign missing flag
masses_data_impute['missing']=np.where(masses_data_impute['missing']==0,1,0)
### Function to count the number of missing values for the predictors
def zero_value(row):
    missing_num=0
    if row['Age']==0: 
        missing_num+=1
    if row['Shape']==0:
        missing_num+=1
    if row['Margin']==0:
        missing_num+=1
    if row['Density']==0:
        missing_num+=1
        
    return missing_num
### Assigns a variable that contains the number of missing attributes for the row
masses_data_impute['missing_num']=masses_data_impute[['Age','Shape','Margin','Density']].apply(lambda row: zero_value(row),axis=1)
### Store drops in a dataframe
dropped_data=masses_data_impute[masses_data_impute['missing_num']>1]
masses_data_impute=masses_data_impute[masses_data_impute['missing_num']<=1]


#%%
##############################################################
#------------------------------------------------------------- 
# Decision Tree for Classification
#------------------------------------------------------------- 
##############################################################

### Prepare the data

X = masses_data_impute[['Age','Shape','Margin','Density']].values
Y = masses_data_impute['Severity'].values
### for labels, we will store these in a list
X_names = ['Age', 'Shape', 'Margin', 'Density']

def DecisionTreeModel(x,y,names,ccp=None,split_type='gini',png_tree=None):
    
    np.random.seed(1234) ### set seed for repetition

    ### Split the data into training and testing, 6/40 split
    (training_inputs,
     testing_inputs,
     training_classes,
     testing_classes) = train_test_split(X, Y, train_size=0.60,test_size=0.4, random_state=1)    
    
    ### Assign the model    
    clf=DecisionTreeClassifier(random_state=1,criterion=split_type)
    ### Fit the model:
    # Train the classifier on the training set
    clf.fit(training_inputs, training_classes)
    
    ### We dont want to remove the complexity, trees will be very large
    if ccp=='Y':
        
        ### Assign the cost complexity on the training data to prune the tree
        ccp_path=clf.cost_complexity_pruning_path(training_inputs, training_classes)
        ### Now, using the path, run the decision tree again for the list off effective alphas
        ### Initiate a list of alphas to append for plotting
        clfs=[]
        for ccp_alpha in ccp_path.ccp_alphas[:-1]:
            ### Fit the model iterating through the ccps
            clf=DecisionTreeClassifier(random_state=1,criterion=split_type,ccp_alpha=ccp_alpha).fit(training_inputs,training_classes)
            ### Append the model to the list
            clfs.append(clf)            
                
        ### Initiate the graph
        fig,ax=plt.subplots()
        ### Plot the alphas/impurities, removing the last observation, as its redundent.
        ax.plot(ccp_path.ccp_alphas[:-1],ccp_path.impurities[:-1],drawstyle='steps-post')
        ax.set_xlabel('Effective Alpha')
        ax.set_ylabel('Total Impurity of Leaves')
        ax.set_title('Total Impurity of Leaves vs Effective Alpha on Training Data')
        fig.savefig(PATH + '\\outputs\\' + png_tree + '_Impurities.png')
        fig.clf()
        plt.close(fig)        
        

        ### Plot the Nodes of the Tree
        fig,ax=plt.subplots()        
        node_counts=[clf.tree_.node_count for clf in clfs]
        ax.plot(ccp_path.ccp_alphas[:-1],node_counts,marker='o',drawstyle='steps-post')
        ax.set_xlabel('Alpha')
        ax.set_ylabel('Number of Tree Nodes')
        ax.set_title('Number of Tree Nodes vs Alpha')
        fig.savefig(PATH + '\\outputs\\' + png_tree + '_Nodes.png')
        fig.clf()
        plt.close(fig)                
        
        ### Plot the depth of the tree        
        depth=[clf.tree_.max_depth for clf in clfs]        
        ax.plot(ccp_path.ccp_alphas[:-1],depth,marker='o',drawstyle='steps-post')
        ax.set_xlabel('Alpha')
        ax.set_ylabel('Depth of Decision Tree')
        ax.set_title('Depth of Decision Tree vs Alpha')        
        fig.savefig(PATH + '\\outputs\\' + png_tree + '_Depth.png')
        fig.clf()
        plt.close(fig)        
        
        ### Now, we deviate from the CCP's max value, and change the criterion to the best CCP value on accuracy of the testing data
        train_scores = [clf.score(training_inputs, training_classes) for clf in clfs]
        test_scores = [clf.score(testing_inputs, testing_classes) for clf in clfs]
        ### Find the position of the best accuracy
        result=np.where(np.array(test_scores)==np.array(test_scores).max())[0][0]
        #print(result[0][0])
        ccp_alphas=ccp_path.ccp_alphas[:-1]
        best_alpha=ccp_alphas[result]
        ### Refit, the training data using the CCP value which contains the best accuracy
        clf=DecisionTreeClassifier(random_state=1,criterion=split_type,ccp_alpha=best_alpha).fit(training_inputs,training_classes)
        
        fig, ax = plt.subplots()
        ax.set_xlabel("Alpha")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy vs Aplha for training and testing data")
        #clfs = clfs[:-1]
        ax.plot(ccp_path.ccp_alphas[:-1], train_scores, marker='o', label="train",
                drawstyle="steps-post")
        ax.plot(ccp_path.ccp_alphas[:-1], test_scores, marker='o', label="test",
                drawstyle="steps-post")
        ax.legend()
        fig.savefig(PATH + '\\outputs\\' + png_tree + '_Accuracy.png')
        fig.clf()
        plt.close(fig)
        
        # Reporting, Decision Tree
        dot_data = StringIO()  
        tree.export_graphviz(clf, out_file=dot_data,  
                             feature_names=X_names)  
        graph = graph_from_dot_data(dot_data.getvalue())  
        graph.write_png(PATH + '\\outputs\\' + png_tree + '_Pruned.png')        
    else:
        # Reporting, Decision Tree
        dot_data = StringIO()  
        tree.export_graphviz(clf, out_file=dot_data,  
                             feature_names=X_names)  
        graph = graph_from_dot_data(dot_data.getvalue())  
        graph.write_png(PATH + '\\outputs\\' + png_tree + '.png')
    
    y_score=clf.score(testing_inputs,testing_classes)
    y_feature=dict(zip(names,clf.feature_importances_))
        
    return y_score,y_feature

gini_score,gini_features=DecisionTreeModel(X,Y,X_names,png_tree='Gini_Tree')
entropy_score,entropy_features=DecisionTreeModel(X,Y,X_names,split_type='entropy',png_tree='Entropy_Tree')
### CCP Version for Pruning.
gini_score_purned,gini_features_pruned=DecisionTreeModel(X,Y,X_names,ccp='Y',png_tree='Gini_Tree')
entropy_score_pruned,entropy_features_pruned=DecisionTreeModel(X,Y,X_names,ccp='Y',split_type='entropy',png_tree='Entropy_Tree')

#------------------------------------------------------------- 
# Manual Pruning
#------------------------------------------------------------- 

# Manually control the tree
(training_inputs,
 testing_inputs,
 training_classes,
 testing_classes) = train_test_split(X, Y, train_size=0.60,test_size=0.4, random_state=1)  

clf=DecisionTreeClassifier(random_state=1,max_depth=4,min_samples_split=15,criterion='entropy').fit(training_inputs, training_classes)
manual_score=clf.score(testing_inputs,testing_classes)
manual_features=dict(zip(X_names,clf.feature_importances_))

# Reporting, Decision Tree
dot_data = StringIO()  
tree.export_graphviz(clf, out_file=dot_data,  
                     feature_names=X_names)  
graph = graph_from_dot_data(dot_data.getvalue())  
graph.write_png(PATH + '\\outputs\\' + 'Manual_Tree' + '.png')

print('Decision Tree Gargabe, ',str(gc.collect()))
print('Decision Tree, uncollectable garbage ',str(gc.garbage))

#%%

##############################################################
#------------------------------------------------------------- 
# Random Forest
#------------------------------------------------------------- 
##############################################################

#------------------------------------------------------------- 
# Prepare the features
#------------------------------------------------------------- 

X = masses_data_impute[['Age','Shape','Margin','Density']].values
Y = masses_data_impute['Severity'].values
### for labels, we will store these in a list
X_names = ['Age', 'Shape', 'Margin', 'Density']

(training_inputs,
 testing_inputs,
 training_classes,
 testing_classes) = train_test_split(X, Y, train_size=0.60,test_size=0.4, random_state=1)  

# Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.

def oob_forest(min_estimators=5,max_estimators=500):
    
    ### For out of bag error rates, generate 3 ways of reduction, SqRt, Log2 and None for error rate comparisons
    enseble_clfs=[
        ('None',RandomForestClassifier(warm_start=True,oob_score=True,max_features=None,random_state=1,criterion='entropy')),   
        ('(Restricted) None',RandomForestClassifier(warm_start=True,oob_score=True,max_features=None,random_state=1,max_depth=4,min_samples_split=15,criterion='entropy')) 
        ]    
    
    error_rate = OrderedDict((label, []) for label, _ in enseble_clfs)

    for label,clf in enseble_clfs:
        for i in range(min_estimators,max_estimators+1):
            clf.set_params(n_estimators=i)
            clf.fit(training_inputs,training_classes)
            score=clf.score(testing_inputs,testing_classes)
            
            oob_error=1-clf.oob_score_
            #imp=clf.feature_importances_
            imp=dict(zip(X_names,clf.feature_importances_))
            
            error_rate[label].append((i,oob_error,score,imp))
   
    plt.clf()
    plt.close('all')   

    for label, clf_err in error_rate.items():
        xs, ys, s, x = zip(*clf_err)
        plt.plot(xs, ys, label=label)
        
    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("# Trees")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    plt.title('Error Rate')
    plt.savefig(PATH + '\\outputs\\Random_Forest_Error.png')
   
    plt.clf()   
    plt.close('all')    
    for label, clf_err in error_rate.items():
        xs, ys, s, x = zip(*clf_err)
        plt.plot(xs,s,label=label)
        
    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("# Trees")
    plt.ylabel("OOB error rate")
    plt.legend(loc="lower right")
    plt.title('Accuracy')
    plt.savefig(PATH + '\\outputs\\Random_Forest_Accruacy.png')
    plt.clf()    
    plt.close('all')
    
    return error_rate
    
RF_info=oob_forest()

#------------------------------------------------------------- 
# Extract best performing trees
#------------------------------------------------------------- 

RF_None_best=np.array([i[2] for i in RF_info['None']])
print('None Average: ',RF_None_best.mean())
RF_None_forest=[i for i in RF_info['None'] if i[2]==RF_None_best.max()]
        
RF_Restricted_best=np.array([i[2] for i in RF_info['(Restricted) None']])
print('Restricted Average: ',RF_Restricted_best.mean())
RF_Restricted_forest=[i for i in RF_info['(Restricted) None'] if i[2]==RF_Restricted_best.max()]

for i in RF_info.items():
    
    RF_None_Importance={'Age':[],'Shape':[],'Margin':[],'Density':[]}
    for j in RF_info['None']:
        RF_None_Importance['Age'].append(j[3]['Age'])
        RF_None_Importance['Shape'].append(j[3]['Shape'])
        RF_None_Importance['Margin'].append(j[3]['Margin'])
        RF_None_Importance['Density'].append(j[3]['Density'])

    RF_Restricted_Importance={'Age':[],'Shape':[],'Margin':[],'Density':[]}
    for j in RF_info['(Restricted) None']:
        RF_Restricted_Importance['Age'].append(j[3]['Age'])
        RF_Restricted_Importance['Shape'].append(j[3]['Shape'])
        RF_Restricted_Importance['Margin'].append(j[3]['Margin'])
        RF_Restricted_Importance['Density'].append(j[3]['Density'])

print('None (Age): '+str(np.array(RF_None_Importance['Age']).mean()))
print('None (Shape): '+str(np.array(RF_None_Importance['Shape']).mean()))
print('None (Margin): '+str(np.array(RF_None_Importance['Margin']).mean()))
print('None (Density): '+str(np.array(RF_None_Importance['Density']).mean()))
  
print('Restricted (Age): '+str(np.array(RF_Restricted_Importance['Age']).mean()))
print('Restricted (Shape): '+str(np.array(RF_Restricted_Importance['Shape']).mean()))
print('Restricted (Margin): '+str(np.array(RF_Restricted_Importance['Margin']).mean()))
print('Restricted (Density): '+str(np.array(RF_Restricted_Importance['Density']).mean()))

#------------------------------------------------------------- 
# Extract best performing trees
#------------------------------------------------------------- 

RF=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=1,max_depth=1,min_samples_split=15)
RF.fit(training_inputs,training_classes)
print(RF.score(testing_inputs,testing_classes))
RF_features=dict(zip(X_names,RF.feature_importances_))

print('Random Forest Gargabe, ',str(gc.collect()))
print('Random Forest, uncollectable garbage ',str(gc.garbage))

#%%

##############################################################
#------------------------------------------------------------- 
# Nearest Neighbors
#------------------------------------------------------------- 
##############################################################

# https://stackabuse.com/k-nearest-neighbors-algorithm-in-python-and-scikit-learn/

#------------------------------------------------------------- 
# Prepare the features
#------------------------------------------------------------- 

X = masses_data_impute[['Age','Shape','Margin','Density']].values
Y = masses_data_impute['Severity'].values
### for labels, we will store these in a list
X_names = ['Age', 'Shape', 'Margin', 'Density']

(training_inputs,
 testing_inputs,
 training_classes,
 testing_classes) = train_test_split(X, Y, train_size=0.60,test_size=0.4, random_state=1)  

scaler=StandardScaler()
scaler.fit(training_inputs)

training_scaled=scaler.transform(training_inputs)
testing_scaled=scaler.transform(testing_inputs)

def KNN_model():
    KNN_dict={'Uniform Non-PCA':[],'Uniform PCA':[],'Distance Non-PCA':[],'Distance PCA':[]}
    nca=neighbors.NeighborhoodComponentsAnalysis(random_state=1)
    for i in range(1,40):
        K_n=neighbors.KNeighborsClassifier(n_neighbors=i,weights='uniform')
        K_n.fit(training_scaled,training_classes)
        non_pca=(i,K_n.score(testing_scaled,testing_classes))
        KNN_dict['Uniform Non-PCA'].append(non_pca)
        
        nca_pipe=Pipeline([('nca',nca),('K_n',K_n)])
        nca_pipe.fit(training_scaled,training_classes)
        pca=(i,nca_pipe.score(testing_scaled,testing_classes))
        KNN_dict['Uniform PCA'].append(pca)
        
        K_n=neighbors.KNeighborsClassifier(n_neighbors=i,weights='distance')
        K_n.fit(training_scaled,training_classes)
        non_pca=(i,K_n.score(testing_scaled,testing_classes))
        KNN_dict['Distance Non-PCA'].append(non_pca)
        
        nca_pipe=Pipeline([('nca',nca),('K_n',K_n)])
        nca_pipe.fit(training_scaled,training_classes)
        pca=(i,nca_pipe.score(testing_scaled,testing_classes))
        KNN_dict['Distance PCA'].append(pca)        
        
    return KNN_dict

KNN_results=KNN_model()

plt.plot([x[1] for x in KNN_results['Uniform PCA']],label='Uniform PCA')        
plt.plot([x[1] for x in KNN_results['Uniform Non-PCA']],label='Uniform Non-PCA')       
plt.plot([x[1] for x in KNN_results['Distance PCA']],label='Distance PCA')        
plt.plot([x[1] for x in KNN_results['Distance Non-PCA']],label='Distance Non-PCA')        
 
plt.xlabel("# of Neighbors")
plt.ylabel("Accuracy")
plt.legend(loc="lower right")
plt.title('KNN Accuracy')
        
plt.savefig(PATH + '\\outputs\\KNN_Accuracy.png')
plt.close('all')   


plt.plot([1-x[1] for x in KNN_results['Uniform PCA']],label='Uniform PCA')        
plt.plot([1-x[1] for x in KNN_results['Uniform Non-PCA']],label='Uniform Non-PCA')       
plt.plot([1-x[1] for x in KNN_results['Distance PCA']],label='Distance PCA')        
plt.plot([1-x[1] for x in KNN_results['Distance Non-PCA']],label='Distance Non-PCA')        
 
plt.xlabel("# of Neighbors")
plt.ylabel("Error")
plt.legend(loc="upper right")
plt.title('KNN Error')

plt.savefig(PATH + '\\outputs\\KNN_Error.png')
plt.close('all')   

K_n=neighbors.KNeighborsClassifier(n_neighbors=15)
nca=neighbors.NeighborhoodComponentsAnalysis(random_state=1)
K_n.fit(training_scaled,training_classes)
pred_testing=K_n.predict(testing_scaled)
K_n.score(testing_scaled,testing_classes)

print('Distance Non-PCA Results: ',
      str([(x,i) for x,i in KNN_results['Distance Non-PCA'] if np.array([i for x,i in KNN_results['Distance Non-PCA']]).max()==i][0]))
print('Distance PCA Results: ',
      str([(x,i) for x,i in KNN_results['Distance PCA'] if np.array([i for x,i in KNN_results['Distance PCA']]).max()==i][0]))
print('Uniform Non-PCA Results: ',
      str([(x,i) for x,i in KNN_results['Uniform Non-PCA'] if np.array([i for x,i in KNN_results['Uniform Non-PCA']]).max()==i][0]))
print('Uniform PCA Results: ',
      str([(x,i) for x,i in KNN_results['Uniform PCA'] if np.array([i for x,i in KNN_results['Uniform PCA']]).max()==i][0]))
print('Manual: ',str(K_n.score(testing_scaled,testing_classes)))
print(confusion_matrix(testing_classes, pred_testing))

#%%

##############################################################
#------------------------------------------------------------- 
# Niave Bayes
#------------------------------------------------------------- 
##############################################################

X = masses_data_impute[['Age','Shape','Margin','Density']].values
Y = masses_data_impute['Severity'].values
### for labels, we will store these in a list
X_names = ['Age', 'Shape', 'Margin', 'Density']

(training_inputs,
 testing_inputs,
 training_classes,
 testing_classes) = train_test_split(X, Y, train_size=0.60,test_size=0.4, random_state=1)  

   
def NB_model():

    ### List of models:
    NB_info={'Categorical NB':[],
             'Gaussian NB':[],
             'Multinomial NB':[],
             'Bernoulli NB':[],
             'Bernoulli NB (Binarize=3)':[],
             'Complement NB':[]
             }
        
    for i in range(0,100+1,1):
        
        ### Categorical NB Model
        cat_nb=CategoricalNB(alpha=max(1e-10,i/100))
        cat_nb.fit(training_inputs,training_classes)
        cat_score=cat_nb.score(testing_inputs,testing_classes)
        NB_info['Categorical NB'].append((i,cat_score))
        
        ### Gaussian NB Model
        gaus_nb=GaussianNB(var_smoothing=max(1e-10,i/100))
        gaus_nb.fit(training_inputs,training_classes)
        gaus_score=gaus_nb.score(testing_inputs,testing_classes)
        NB_info['Gaussian NB'].append((i,gaus_score))        
        
        ### Multinomial NB Model
        mn_nb=MultinomialNB(alpha=max(1e-10,i/100))
        mn_nb.fit(training_inputs,training_classes)
        mn_score=mn_nb.score(testing_inputs,testing_classes)
        NB_info['Multinomial NB'].append((i,mn_score))
    
        ### Complement NB Model
        com_nb=ComplementNB(alpha=max(1e-10,i/100))
        com_nb.fit(training_inputs,training_classes)
        com_score=com_nb.score(testing_inputs,testing_classes)
        NB_info['Complement NB'].append((i,com_score))
        
        ### Bernoulli NB Model
        bern_nb=BernoulliNB(alpha=max(1e-10,i/100))
        bern_nb.fit(training_inputs,training_classes)
        bern_score=bern_nb.score(testing_inputs,testing_classes)
        NB_info['Bernoulli NB'].append((i,bern_score))     
        
        ### Bernoulli NB Model, Bin=5
        bern_nb_5=BernoulliNB(alpha=max(1e-10,i/100),binarize=3)
        bern_nb_5.fit(training_inputs,training_classes)
        bern_score_5=bern_nb_5.score(testing_inputs,testing_classes)
        NB_info['Bernoulli NB (Binarize=3)'].append((i,bern_score_5))        

    return NB_info

NB_info=NB_model()

print('Categorical NB: ',
      str([(x,i) for x,i in NB_info['Categorical NB'] if np.array([i for x,i in NB_info['Categorical NB']]).max()==i][0]))
print('Gaussian NB: ',
      str([(x,i) for x,i in NB_info['Gaussian NB'] if np.array([i for x,i in NB_info['Gaussian NB']]).max()==i][0]))
print('Multinomial NB: ',
      str([(x,i) for x,i in NB_info['Multinomial NB'] if np.array([i for x,i in NB_info['Multinomial NB']]).max()==i][0]))
print('Complement NB: ',
      str([(x,i) for x,i in NB_info['Complement NB'] if np.array([i for x,i in NB_info['Complement NB']]).max()==i][0]))
print('Bernoulli NB: ',
      str([(x,i) for x,i in NB_info['Bernoulli NB'] if np.array([i for x,i in NB_info['Bernoulli NB']]).max()==i][0]))
print('Bernoulli NB (Binarize=3): ',
      str([(x,i) for x,i in NB_info['Bernoulli NB (Binarize=3)'] if np.array([i for x,i in NB_info['Bernoulli NB (Binarize=3)']]).max()==i][0]))


        
plt.plot([x[1] for x in NB_info['Categorical NB']],label='Categorical NB')        
plt.plot([x[1] for x in NB_info['Gaussian NB']],label='Gaussian NB')       
plt.plot([x[1] for x in NB_info['Multinomial NB']],label='Multinomial NB')        
plt.plot([x[1] for x in NB_info['Complement NB']],label='Complement NB')        
plt.plot([x[1] for x in NB_info['Bernoulli NB']],label='Bernoulli NB')        
plt.plot([x[1] for x in NB_info['Bernoulli NB (Binarize=3)']],label='Bernoulli NB (Binarize=3)')        

plt.xlabel("Alpha Level")
plt.ylabel("Accuracy")
plt.legend(loc="center",bbox_to_anchor=(0.5,-0.3),ncol=2)
plt.title('Niave Bayes Accuracy')
        
plt.savefig(PATH + '\\outputs\\Niave_Bayes_Accruacy.png')
plt.close('all')   

plt.plot([1-x[1] for x in NB_info['Categorical NB']],label='Categorical NB')        
plt.plot([1-x[1] for x in NB_info['Gaussian NB']],label='Gaussian NB')       
plt.plot([1-x[1] for x in NB_info['Multinomial NB']],label='Multinomial NB')        
plt.plot([1-x[1] for x in NB_info['Complement NB']],label='Complement NB')        
plt.plot([1-x[1] for x in NB_info['Bernoulli NB']],label='Bernoulli NB')        
plt.plot([1-x[1] for x in NB_info['Bernoulli NB (Binarize=3)']],label='Bernoulli NB (Binarize=3)')        

plt.xlabel("Alpha Level")
plt.ylabel("Error Rate")
plt.legend(loc="center",bbox_to_anchor=(0.5,-0.3),ncol=2)
plt.title('Niave Bayes Error Rate')
        
plt.savefig(PATH + '\\outputs\\Niave_Bayes_Error.png')
plt.close('all')   


#%%

##############################################################
#------------------------------------------------------------- 
# SVM
#------------------------------------------------------------- 
##############################################################

X = masses_data_impute[['Age','Shape','Margin','Density']].values
Y = masses_data_impute['Severity'].values
### for labels, we will store these in a list
X_names = ['Age', 'Shape', 'Margin', 'Density']

(training_inputs,
 testing_inputs,
 training_classes,
 testing_classes) = train_test_split(X, Y, train_size=0.60,test_size=0.4, random_state=1)  

scaler=StandardScaler()
scaler.fit(training_inputs)

training_scaled=scaler.transform(training_inputs)
testing_scaled=scaler.transform(testing_inputs)

def SVM_search():
    
    SVM_info={'linear':[],'poly_3':[],'poly_2':[],'poly_4':[],'rbf':[],
              'linear_scaled':[],'poly_3_scaled':[],'poly_2_scaled':[],'poly_4_scaled':[],'rbf_scaled':[]}
    
    for k in SVM_info.keys():
        for j in list(range(1,101,5)):
            
            ### split for degree on poly...
            if k.split('_')[0] == 'poly':
                svc=svm.SVC(kernel=k.split('_')[0],C=j,random_state=1,degree=int(k.split('_')[1]))
            elif k.split('_')[-1] == 'scaled':
                svc=svm.SVC(kernel=k.split('_')[0],C=j,random_state=1)
            else:
                svc=svm.SVC(kernel=k,C=j,random_state=1)
            
            if k.split('_')[-1] == 'scaled':
                svc.fit(training_scaled,training_classes)
                SVM_info[k].append((j,svc.score(testing_scaled,testing_classes)))
            else:
                svc.fit(training_inputs,training_classes)
                SVM_info[k].append((j,svc.score(testing_inputs,testing_classes)))
            
            del svc
            
    return SVM_info

SVM_result=SVM_search()

print('Linear SVM: ',
      str([(x,i) for x,i in SVM_result['linear'] if np.array([i for x,i in SVM_result['linear']]).max()==i][0]))
print('Poly(3) SVM: ',
      str([(x,i) for x,i in SVM_result['poly_3'] if np.array([i for x,i in SVM_result['poly_3']]).max()==i][0]))
print('Poly(4) SVM: ',
      str([(x,i) for x,i in SVM_result['poly_4'] if np.array([i for x,i in SVM_result['poly_4']]).max()==i][0]))
print('RBF SVM: ',
      str([(x,i) for x,i in SVM_result['rbf'] if np.array([i for x,i in SVM_result['rbf']]).max()==i][0]))
print('Linear (Scaled) SVM: ',
      str([(x,i) for x,i in SVM_result['linear_scaled'] if np.array([i for x,i in SVM_result['linear_scaled']]).max()==i][0]))
print('Poly(3) Scaled SVM: ',
      str([(x,i) for x,i in SVM_result['poly_3_scaled'] if np.array([i for x,i in SVM_result['poly_3_scaled']]).max()==i][0]))
print('Poly(4) Scaled SVM: ',
      str([(x,i) for x,i in SVM_result['poly_4_scaled'] if np.array([i for x,i in SVM_result['poly_4_scaled']]).max()==i][0]))
print('RBF Scaled SVM: ',
      str([(x,i) for x,i in SVM_result['rbf_scaled'] if np.array([i for x,i in SVM_result['rbf_scaled']]).max()==i][0]))

plt.plot([x[0] for x in SVM_result['linear']],[x[1] for x in SVM_result['linear']],label='Kernel: Linear')        
#plt.plot([x[0] for x in SVM_result['poly_3']],[x[1] for x in SVM_result['poly_3']],label='Kernel: Poly(3)')        
plt.plot([x[0] for x in SVM_result['poly_4']],[x[1] for x in SVM_result['poly_4']],label='Kernel: Poly(4)')        
plt.plot([x[0] for x in SVM_result['rbf']],[x[1] for x in SVM_result['rbf']],label='Kernel: RBF')
#plt.plot([x[0] for x in SVM_result['linear_scaled']],[x[1] for x in SVM_result['linear_scaled']],label='Kernel: Scaled Linear')        
#plt.plot([x[0] for x in SVM_result['poly_3_scaled']],[x[1] for x in SVM_result['poly_3_scaled']],label='Kernel: Scaled Poly(3)')        
#plt.plot([x[0] for x in SVM_result['poly_4_scaled']],[x[1] for x in SVM_result['poly_4_scaled']],label='Kernel: Scaled Poly(4)')        
#plt.plot([x[0] for x in SVM_result['rbf_scaled']],[x[1] for x in SVM_result['rbf_scaled']],label='Kernel: Scaled RBF')         

plt.xlabel("C")
plt.ylabel("Accuracy")
plt.legend(loc="center",bbox_to_anchor=(0.5,-0.3),ncol=2)
plt.title('SVM Accuracy')
        
plt.savefig(PATH + '\\outputs\\SVM_Accruacy.png',bbox_inches='tight')
plt.close('all')   

plt.plot([x[0] for x in SVM_result['linear']],[1-x[1] for x in SVM_result['linear']],label='Kernel: Linear')        
#plt.plot([1-x[0] for x in SVM_result['poly_3']],[x[1] for x in SVM_result['poly_3']],label='Kernel: Poly(3)')        
plt.plot([x[0] for x in SVM_result['poly_4']],[1-x[1] for x in SVM_result['poly_4']],label='Kernel: Poly(4)')        
plt.plot([x[0] for x in SVM_result['rbf']],[1-x[1] for x in SVM_result['rbf']],label='Kernel: RBF')
#plt.plot([x[0] for x in SVM_result['linear_scaled']],[x[1] for x in SVM_result['linear_scaled']],label='Kernel: Scaled Linear')        
#plt.plot([x[0] for x in SVM_result['poly_3_scaled']],[x[1] for x in SVM_result['poly_3_scaled']],label='Kernel: Scaled Poly(3)')        
#plt.plot([x[0] for x in SVM_result['poly_4_scaled']],[x[1] for x in SVM_result['poly_4_scaled']],label='Kernel: Scaled Poly(4)')        
#plt.plot([x[0] for x in SVM_result['rbf_scaled']],[x[1] for x in SVM_result['rbf_scaled']],label='Kernel: Scaled RBF')         

plt.xlabel("C")
plt.ylabel("Error")
plt.legend(loc="center",bbox_to_anchor=(0.5,-0.3),ncol=2)
plt.title('SVM Error')
        
plt.savefig(PATH + '\\outputs\\SVM_Error.png',bbox_inches='tight')
plt.close('all')   

### Attempt at explanation...

clf_linear=svm.SVC(kernel='linear')
clf_linear.fit(np.array(masses_data_impute[['Margin','Shape']].values),masses_data_impute['Severity'])
plt.scatter(clf_linear.support_vectors_[:,0],clf_linear.support_vectors_[:,1],s=80,facecolors='none',zorder=10,edgecolors='k')
plt.scatter(masses_data_impute['Margin'],masses_data_impute['Shape'],zorder=10,cmap=plt.cm.Paired,edgecolors='k')

XX_linear,YY_linear=np.mgrid[0:5:200j,0:4:200j]
Z_linear=clf_linear.decision_function(np.c_[XX_linear.ravel(),YY_linear.ravel()])
Z_linear=Z_linear.reshape(XX_linear.shape)

plt.pcolormesh(XX_linear,YY_linear,Z_linear > 0,cmap=plt.cm.Paired)
plt.contour(XX_linear,YY_linear,Z_linear,colors=['k','k','k'],linestyles=['--','-','--'],levels=[-0.5,0,0.5])
plt.xlabel('Margin')
plt.ylabel('Shape')
plt.title('SVC: Linear')
plt.savefig(PATH + '\\outputs\\SVC_Linear_Explanation.png',bbox_inches='tight')


clf_rbf=svm.SVC(kernel='rbf')
clf_rbf.fit(np.array(masses_data_impute[['Margin','Shape']].values),masses_data_impute['Severity'])
plt.scatter(clf_rbf.support_vectors_[:,0],clf_rbf.support_vectors_[:,1],s=80,facecolors='none',zorder=10,edgecolors='k')
plt.scatter(masses_data_impute['Margin'],masses_data_impute['Shape'],zorder=10,cmap=plt.cm.Paired,edgecolors='k')

XX_rbf,YY_rbf=np.mgrid[0:5:200j,0:4:200j]
Z_rbf=clf_rbf.decision_function(np.c_[XX_rbf.ravel(),YY_rbf.ravel()])
Z_rbf=Z_rbf.reshape(XX_rbf.shape)

plt.pcolormesh(XX_rbf,YY_rbf,Z_rbf > 0,cmap=plt.cm.Paired)
plt.contour(XX_rbf,YY_rbf,Z_rbf,colors=['k','k','k'],linestyles=['--','-','--'],levels=[-0.5,0,0.5])
plt.xlabel('Margin')
plt.ylabel('Shape')
plt.title('SVC: RBF (3rd Polynomial)')
plt.savefig(PATH + '\\outputs\\SVC_RBF_Explanation.png',bbox_inches='tight')

#%%

##############################################################
#------------------------------------------------------------- 
# Nueral Network
#------------------------------------------------------------- 
##############################################################

X = masses_data_impute[['Age','Shape','Margin','Density']].values
Y = masses_data_impute['Severity'].values
### for labels, we will store these in a list
X_names = ['Age', 'Shape', 'Margin', 'Density']

(training_inputs,
 testing_inputs,
 training_classes,
 testing_classes) = train_test_split(X, Y, train_size=0.60,test_size=0.4, random_state=1)  

def NN_info():
    
    NN_dict={'NN Hidden Layers(1)':[],
             'NN Hidden Layers(2)':[],
             'NN Hidden Layers(3)':[],
             'NN Hidden Layers(4)':[],
             'NN Hidden Layers(5)':[],
             'NN Hidden Layers(6)':[],
             'NN Hidden Layers(7)':[],
             'NN Hidden Layers(8)':[],
             'NN Hidden Layers(9)':[],
             'NN Hidden Layers(10)':[]

             }
    
    for _i in [1,2,3,4,5,6,7,8,9,10]:
        
        for a in np.arange(0.0001,0.05,0.001):
            
            network=MLPClassifier(random_state=1,max_iter=100,activation='logistic',solver='lbfgs',hidden_layer_sizes=(_i,),alpha=a)
            network.fit(training_inputs,training_classes)
            NN_dict['NN Hidden Layers('+str(_i)+')'].append((a,network.score(testing_inputs,testing_classes)))
            
    return NN_dict

NN_info=NN_info()

for k,v in NN_info.items():
    print(k,str([(x,i) for x,i in NN_info[k] if np.array([i for x,i in NN_info[k]]).max()==i][0]))
    
for k,v in NN_info.items():
    plt.plot([x[0] for x in NN_info[k]],[x[1] for x in NN_info[k]],label=k)
   
plt.xlabel("alpha")
plt.ylabel("Accuracy")
plt.legend(loc="center",bbox_to_anchor=(0.5,-0.4),ncol=2)
plt.title('Neural Network')
        
plt.savefig(PATH + '\\outputs\\Neural_Network_Accuracy.png',bbox_inches='tight')
plt.close('all')   

for k,v in NN_info.items():
    plt.plot([x[0] for x in NN_info[k]],[1-x[1] for x in NN_info[k]],label=k)
   
plt.xlabel("alpha")
plt.ylabel("Error")
plt.legend(loc="center",bbox_to_anchor=(0.5,-0.4),ncol=2)
plt.title('Neural Network')
        
plt.savefig(PATH + '\\outputs\\Neural_Network_Error.png',bbox_inches='tight')
plt.close('all')   

#%%

##############################################################
#------------------------------------------------------------- 
# Logistic Regression
#------------------------------------------------------------- 
##############################################################

X = masses_data_impute[['Age','Shape','Margin','Density']].values
Y = masses_data_impute['Severity'].values
### for labels, we will store these in a list
X_names = ['Age', 'Shape', 'Margin', 'Density']

(training_inputs,
 testing_inputs,
 training_classes,
 testing_classes) = train_test_split(X, Y, train_size=0.60,test_size=0.4, random_state=1)  

def Logit_Model():
    
    logit_dict={'Logit Intercept':[],'Logit No-Intercept':[]}
    
    for intercept in [True,False]:
        for C in np.arange(0.01,1.01,0.01):
            
            logit=LogisticRegression(C=C,fit_intercept=intercept,random_state=1)
            logit.fit(training_inputs,training_classes)
            if intercept == True:
                logit_dict['Logit Intercept'].append((C,logit.score(testing_inputs,testing_classes)))
            else:
                logit_dict['Logit No-Intercept'].append((C,logit.score(testing_inputs,testing_classes)))
            
    return logit_dict

Logistic_info=Logit_Model()

print('Logistic Regssion (Intercept): ',str([(x,i) for x,i in Logistic_info['Logit Intercept'] if np.array([i for x,i in Logistic_info['Logit Intercept'] ]).max()==i][0]))
print('Logistic Regssion (No Intercept): ',str([(x,i) for x,i in Logistic_info['Logit No-Intercept'] if np.array([i for x,i in Logistic_info['Logit No-Intercept'] ]).max()==i][0]))

plt.plot([x[0] for x in Logistic_info['Logit Intercept']],[x[1] for x in Logistic_info['Logit Intercept']],label='Logistic Regssion (Intercept)')
plt.plot([x[0] for x in Logistic_info['Logit No-Intercept']],[x[1] for x in Logistic_info['Logit No-Intercept']],label='Logistic Regssion (No Intercept)')

plt.xlabel("C")
plt.ylabel("Accuracy")
plt.legend(loc="center",bbox_to_anchor=(0.5,-0.3),ncol=2)
plt.title('Logistic Regression')

plt.savefig(PATH + '\\outputs\\Logistic_Accuracy.png',bbox_inches='tight')
plt.close('all')

plt.plot([x[0] for x in Logistic_info['Logit Intercept']],[1-x[1] for x in Logistic_info['Logit Intercept']],label='Logistic Regssion (Intercept)')
plt.plot([x[0] for x in Logistic_info['Logit No-Intercept']],[1-x[1] for x in Logistic_info['Logit No-Intercept']],label='Logistic Regssion (No Intercept)')

plt.xlabel("C")
plt.ylabel("Error")
plt.legend(loc="center",bbox_to_anchor=(0.5,-0.3),ncol=2)
plt.title('Logistic Regression')

plt.savefig(PATH + '\\outputs\\Logistic_Error.png',bbox_inches='tight')
plt.close('all')   

#%%

##############################################################
#------------------------------------------------------------- 
# Linear Probability
#------------------------------------------------------------- 
##############################################################

X = masses_data_impute[['Age','Shape','Margin','Density']].values
Y = masses_data_impute['Severity'].values
### for labels, we will store these in a list
X_names = ['Age', 'Shape', 'Margin', 'Density']

(training_inputs,
 testing_inputs,
 training_classes,
 testing_classes) = train_test_split(X, Y, train_size=0.60,test_size=0.4, random_state=1)  

def OLS_accuracy(y_true,y_predict):
    return accuracy_score(y_true,[round(y) for y in y_predict])

OLS_intercept=LinearRegression(fit_intercept=True,normalize=True)
#OLS_intercept=LinearRegression(fit_intercept=True)
OLS_intercept.fit(training_inputs,training_classes)
print('OLS (Intercept)',str(OLS_accuracy(testing_classes,[round(x) for x in OLS_intercept.predict(testing_inputs)])))

OLS_nointercept=LinearRegression(fit_intercept=False)
OLS_nointercept.fit(training_inputs,training_classes)
OLS_nointercept.score(testing_inputs,testing_classes) 
print('OLS (No Intercept)',str(OLS_accuracy(testing_classes,[round(x) for x in OLS_nointercept.predict(testing_inputs)])))

#%%
#Extremely helpful for the final information, regarding the plotting of times and the learning curves:
#https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py


