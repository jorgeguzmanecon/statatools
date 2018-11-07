
###########################################################################
# This script is very simple and allows a user to run a random forest but #
# does not perform any fancy modeling on the forest                       #
###########################################################################
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pandas as pd
import argparse
import pdb
import numpy as np
from sklearn.preprocessing import Imputer
import sklearn.metrics as skm
import random
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def estimate_percentile_distribution(y_true, y_pred):
    d = pd.DataFrame(y_true, y_pred)
    d.sort_values(y_pred, inplace=True)
    d.reset_index()
    d['percentile'] = np.floor(d.index*20/len(d.index))
    
    res = d.groupby('percentile').y_true.sum()
    sum_true = d.y_true.sum()
    res['share'] = res['y_true']/sum_true
    return res



def n_fold_cross_validation(X,y, train_index,output_file, folds=10):
    sortorder = pd.Series(np.random.rand(len(train_index),1)[:,0])
    d = {'train_index': train_index, 'sortorder': sortorder}
    folds = pd.DataFrame(d)
    folds.sort_values('sortorder', inplace = True)
    folds.reset_index()
    folds['fold'] = np.floor(folds.index*10/len(folds.index))

    results = None

    for fold_num in  range(10):
        train_folds_index = folds.train_index & (folds.fold != fold_num)
        test_folds_index  = folds.train_index & (folds.fold == fold_num)
        
        preds = randomforest(X, y, train_folds_index, report= False)
        (y_true, y_pred) = (y[test_folds_index],  preds[test_folds_index])

        perc_dist = estimate_percentile_distribution(y_true, y_pred)
        perc_dist['fold'] = fold_num

        #Add to all Results
        if results is None:
            results = perc_dist
        else:
            results = results.append(perc_dist)
    
    # Store results in file
    results.to_stata(output_file, write_index = False)
        

#################################
# Method to run a random forest #
#################################
 

def randomforest(X,y, train_index, report=True):
    global roconly

    Xt = X.loc[train_index,:]
    ## Impute data: Is this a good idea?
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp = imp.fit(X)
    X_imp = imp.transform(X)
    X_imp_train = imp.transform(X.loc[train_index,:])
    y_train = y[train_index]

    #Only clean data
    (Xc, yc) = (X_imp_train[~np.isnan(y_train)] , y_train[~np.isnan(y_train)])

    # Random Forest Classifier
    clt = RandomForestClassifier(min_samples_split=2, random_state=99, n_estimators = 30, verbose = 1, max_depth=5)
    clt = clt.fit(Xc,yc)
    preds = clt.predict_proba(X_imp)[:,1]

    if report is True:
        #Print the Features Importance
        importances = clt.feature_importances_
        std = np.std([tree.feature_importances_ for tree in clt.estimators_], axis=0)
        indices = np.argsort(importances)[::-1]
        
        if not roconly: print("\n\t RANDOM FOREST FEATURE RANKING")

        for f in range(X.shape[1]):
            if not roconly: print("\t%d. feature %s (%f)" % (f + 1, str(args.varlist[1:][indices[f]]), importances[indices[f]]))
        
        if not roconly: print "\nRANDOM FOREST COMPLETE" 

    return preds




def logit( X,y, train_index):
    Xt = X.loc[train_index,:]
    ## Impute data: Is this a good idea?
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp = imp.fit(X)
    X_imp = imp.transform(X)
    X_imp_train = imp.transform(X.loc[train_index,:])
    y_train = y[train_index]

    #Only clean data
    (Xc, yc) = (X_imp_train[~np.isnan(y_train)] , y_train[~np.isnan(y_train)])


    # Random Forest Classifier
    lr = LogisticRegression()
    clt = lr.fit(Xc,yc)
    preds = clt.predict_proba(X_imp)[:,1]

    #Print the Features Importance
    # importances = clt._
    # std = np.std([tree.feature_importances_ for tree in clt.estimators_], axis=0)
    # indices = np.argsort(importances)[::-1]

    # print("\n\tLOGIT FEATURE RANKING")

    # for f in range(X.shape[1]):
    #     print("\t%d. feature %s (%f)" % (f + 1, str(args.varlist[1:][indices[f]]), importances[indices[f]]))
 
    print "\tLOGIT MODEL COMPLETE" 
    return preds




## A simple method so that errors fail safe
def get_roc_score(y, y_pred):
    try:
        return skm.roc_auc_score(y, y_pred)
    except ValueError:
        print "Error: The ROC Score could not be estimated.  Possibly because y_pred is a multi value parameter"
        return 0

 



#
# This method predicts all the ROC scores to score the model
# as well as save an ROC graph
#
def get_model_scores(y_pred, y, prediction_index = None, train_index  = None, title="", store_roc = None):
    scores = {}
    print "\t {0}".format(title)

    yg = (~np.isnan(y)) & (~np.isnan(y_pred))
 
    roc_full_sample = get_roc_score(y[yg],y_pred[yg])
    roc_prediction = None
    roc_train = None
    print "\t\t ROC Score (Full Sample): {0}".format(roc_full_sample)

    if prediction_index is not None:
        roc_prediction = get_roc_score(y[prediction_index][yg],y_pred[prediction_index][yg]) 
        print "\t\t ROC Score (Prediction Sample): {0}".format(roc_prediction)
    

    if train_index is not None:
        roc_train = get_roc_score(y[train_index][yg],y_pred[train_index][yg])
        print "\t\t ROC Score (Training Sample): {0}".format(roc_train )

        ## If there is a path provided in store_curve, then it is saved in JPG to that path
        if store_roc is not None:
            print "Storing Curve to: {0}".format(store_roc)
            fpr, tpr, threshold = skm.roc_curve(y[train_index][yg], y_pred[train_index][yg])
            
            fig = plt.figure()
            lw = 2
            roc_auc = skm.auc(fpr, tpr, reorder=True)
            plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (score = %0.2f)' % roc_auc)
            plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.005])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('')            
            plt.legend(loc="lower right")
            fig.savefig(store_roc)
            

    return pd.DataFrame({'roc_full':[roc_full_sample], 'roc_prediction':[roc_prediction], 'roc_train': [roc_train]})

## End get_model_scores 



#Configure the command line parameters
argparser = argparse.ArgumentParser()
argparser.add_argument("statadta",nargs=1)
argparser.add_argument("--roconly", help="Displays only the ROC scores, and nothing else",action='store_true')
argparser.add_argument("--store_roc", help="If a path is provided, it stores the ROC graph on this path", nargs=1)
argparser.add_argument("-g","--gen",help="the name of the variable predicted value", default="pred_proba")
argparser.add_argument("-n","--n_estimators",help="the number of trees to create")
argparser.add_argument('--logit', dest='logit', action='store_true')
argparser.add_argument("varlist",nargs='+')
argparser.add_argument("--train_data",help="A variable name that has a value of 1 only for the observations that should be included in the model")
argparser.add_argument("--predict_data",help="A variable name that has a value of 1 only for the observations that should be included in the prediction")
argparser.set_defaults(logit=False)
argparser.add_argument("--tenfold", help="Runs a 10 fold cross validation and stores it in fileaname", nargs=1)
args = argparser.parse_args()




roconly = False
if hasattr(args, 'roconly') and args.roconly:
    roconly = True


#Print the command as it was typed. Good for debugging.

if not roconly: print " \n\n ******* Starting Random Forest in Python ********* "
if not roconly: print "\tI.COMMAND \n\n-->$\tpython {0}".format(" ".join(sys.argv[:]))
#print "\t print_model = {0}".format(print_model)

# Load Stata File
data = pd.read_stata(args.statadta[0])

#Split into prediction and test
print dir(args)
if hasattr(args, 'train_data') and args.train_data is not None:
    if not roconly: print "Train Data: {0}".format(args.train_data)

    choice_col = data[str(args.train_data)]
    train_index = data[choice_col == 1].index
else:
    train_index = data.index

if (args.predict_data) is not None:
    choice_col = data[str(args.predict_data)]
    predict_index = data[choice_col == 1].index
else:
    predict_index = data.index

#setup X and y
y = data[args.varlist[0]]

Xvars = args.varlist[1:]
#remove the duplicate items (e.g. if listed in both $xGrowth and $xMove
Xvars = list(set(Xvars))
X = data[Xvars]



#run commands
if not roconly: print "\tII.MODELS \n\t{0}".format(" ".join(sys.argv[:]))
preds = randomforest(X,y, train_index)
pred_var = str(args.gen)
data[pred_var] = np.nan
#data[pred_var][predict_index] = preds[predict_index]
data[pred_var] = preds

if args.logit:
    preds = logit(X, y, train_index)
    pred_var =  "{0}_logit".format(str(args.gen))
    data[pred_var] = np.nan
    #data[pred_var][predict_index] = preds[predict_index]
    data[pred_var] = preds



#Prediction Results
print "\n\n\t III. PREDICTION STATISTICS"

store_roc = None
if hasattr(args, 'store_roc') and args.store_roc is not None:
    store_roc = args.store_roc[0]

model_scores = get_model_scores(data[str(args.gen)], y, prediction_index = predict_index, train_index  = train_index, title="Random Forest", store_roc = store_roc)


if args.logit:
    print_model_scores(data["{0}_logit".format(str(args.gen))], y, prediction_index = predict_index, train_index  = train_index, title="Logit")


#Ten Fold Cross Validation
#if args.tenfold is not None &  len(args.tenfold[0]) > 0 :
#    print "\n\n\t IV. TEN FOLD CROSS VALIDATION"
#    n_fold_cross_validation(X, y, train_index, output_file = args.tenfold[0])

# Output Data
out_dta_file = args.statadta[0]
out_roc_file = out_dta_file.replace(".dta","") + "_rocscores.dta"
data.to_stata( out_dta_file , write_index=False, encoding='ascii')
model_scores.to_stata(out_roc_file , write_index=False, encoding='ascii')

if not roconly: print "\n\n\tOutput stored in {0}".format(args.statadta[0])

if not roconly: print "\n ******* End of Python Script *********\n\n"
# 
