#### utils_eval
# Scripts originally copied from jupyter notebook in python-practice repo. See example use there. 

from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, roc_curve, accuracy_score, precision_score, recall_score
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

#### example calls
# BinClassEval(lgr1, X_train, y_train)

# models = {'simple log reg': lgr1, 'reg log reg': grid_search}
# evals = CompareBinClassModels(models, X_train, y_train)

from sklearn.model_selection import TimeSeriesSplit

def cross_val_predict_tscv(model, X, y, method = 'predict', n_splits = 4):
    """
    Using TimeSeriesCV means that you dont get a prediction for the first traiing fold.
    Hence sklearn's cross_val_predict does not work.
    We therefore need a custom function to calculate prediction probabilites using CV.
    
    Input
    =====
    cv, int,  number of splits in TimeSeriesSplit
    
    
    Returns
    ======
    proba_preds, numpy array, containing probability predictions
    label_preds, numpy array, containing label predictions
    y_labels, numpy array, containing associated true labels
    
    #### NOTE: need to include decision_function in case predict_probab does not exist. 
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
#     print(tscv)
    X = X.to_numpy() # change input to numpy from pandas
    
    proba_preds = []
    label_preds = []
    
    #### loop through each split and calculate scores
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model.fit(X_train, y_train) # fit model to this fold of train data
        proba_preds_split = model.predict_proba(X_test)[:,1].tolist() # make predictions to this fold of test(validation) data
        label_preds_split = model.predict(X_test).tolist()
        proba_preds = proba_preds + proba_preds_split # append results to list
        label_preds = label_preds + label_preds_split # append results to list
        
        y_labels = find_corresponding_y_train_values(proba_preds, y) # get list of 0,1 labels that correspond with the tscv preds
    
    return proba_preds, label_preds, y_labels

def find_corresponding_y_train_values(proba_predictions, y_train_real):
    """
    Using TimeSeriesCV means that you dont get a prediction for the first traiing fold.
    This function cuts down the actual y_train array to match those which have been predicted using cross_val_predict_proba_tscv.
    """
    number_preds = len(proba_predictions)
    y_train_tscv_subset = y_train_real.values[-number_preds:]
    return y_train_tscv_subset


from sklearn.metrics import brier_score_loss, average_precision_score, precision_recall_curve, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

class ModelCVEvaluation():
    """
    Class to evaluate fully a model using CV procedure.
    Model fitted and then evaluated on each fold.
    Outputs saved in class include: 
    - each train fold:  average_precision, briers score
    - each validation fold: average_precision, briers score

    - plotting of each eval fold available: inc, number of points, 
    - 
    """
    def __init__(self, model, model_name, X, y, cv_splits):
        """
        """
        # assign init values
        self.model = model
        self.model_name = model_name
        self.X = X.to_numpy()
        self.y = y.to_numpy()
        self.cv_splits = cv_splits
        self.cv = TimeSeriesSplit(n_splits=cv_splits)
        
        # create empty results attributes
        self.train_probab_preds_byfold = pd.DataFrame(index=np.arange(0,len(X))) # dfs contain values in columns fold1, etc
        self.valid_probab_preds_byfold = pd.DataFrame(index=np.arange(0,len(X))) # with values indexed as in CV fold index.
        self.valid_probab_preds_all = []
        self.valid_y_labels_byfold = pd.DataFrame(index=np.arange(0,len(X)))
        self.train_y_labels_byfold = pd.DataFrame(index=np.arange(0,len(X)))
        
        # call methods to conduct evaluation
        self.evaluate()
        
        return
    
    def evaluate(self):
        """
        Main class to call all evaluation functionality in order.
        - loops through each CV fold and calls relevant methods.
        """
        self.get_predictions()
        self.get_valid_scores()
        self.plot_PR_curve(self.valid_y_true_all ,self.valid_probab_preds_all, 'PR curves for all CV folds')
        return
    
    def get_predictions(self):
        """ 
        Calls loop over CV and records train + validation probabilities. 
        Indexes for each fold are gneerated using the TimeSeriesCV object from sklearn. 
        
        Values for each CV fold are recorded in a dataframe, the col name is in format 'fold1', 'fold2' etc. 
        The indexing matches the original data index. 
        self.train_probab_preds_byfold, df, probabilities which were calculated when fitting on training fold, each fold saved in col 'fold1'..
        self.valid_probab_preds_byfold, df, probs which were calculated when predicting on validation fold, each fold saved in col'fold1' .       
        
        The true labels are then recorded in the same df format:
        self.valid_y_labels_byfold, df, 
        self.train_y_labels_byfold, df, 
        
        All outputs are assigned as class attributes.
        
        """
        
        fold_counter = 1
        train_probab_preds_byfold = pd.DataFrame()
        valid_probab_preds_byfold = pd.DataFrame()
        
        for train_index, valid_index in self.cv.split(self.X):
            X_train, X_valid = self.X[train_index], self.X[valid_index]
            y_train, y_valid = self.y[train_index], self.y[valid_index]
            fold_name = 'fold' + str(fold_counter)
            
            #### fit model to this fold of train data
            self.model.fit(X_train, y_train)

            #### get validation probabilities - NOTE: need to account for when decison function method availalbe
            valid_proba_preds_split = self.model.predict_proba(X_valid)[:,1].tolist() # make predictions to this fold of valid data
            # assign the results for this fold to a df with the name fold1 etc, 
            # values are placed into the same index as CV.
            self.valid_probab_preds_byfold.loc[valid_index, fold_name] = valid_proba_preds_split
            
            train_proba_preds_split = self.model.predict_proba(X_train)[:,1].tolist() # make predictions to this fold of valid data
            self.train_probab_preds_byfold.loc[train_index, fold_name] = train_proba_preds_split #[0:5]
            
            #### assign y labels
            self.valid_y_labels_byfold.loc[valid_index, fold_name] = y_valid
            self.train_y_labels_byfold.loc[train_index, fold_name] = y_train

            # increment fold number for naming in df for next loop
            fold_counter += 1
            
        return
    
    def get_valid_scores(self):
        """
        Calculates score for validation sets, for all validation points and for each CVfold individually.
        
        Takes the _byfold validation dataframes and concatonates into single array (this only works for validation as there are no repeated values, unlike train data.)
        These contain all validation values from each of the CV folds.
        self.valid_y_probab_pred_all, 1D array, probability predictions for all folds, from models which have been trained seperately on each fold of trainging data. 
        self.valid_y_true_all, 1D array, correpsonding true labels for the predictions in array above.
        
        Performance scores for all CV validation samples calced and assigned to: 
        self.valid_average_precision_score_all, self.valid_brier_score_loss_all
        
        Performance scores for each CV fold are calculated seperately and assigned to:
        self.CV_average_precision_scores, list of floats, 
        self.CV_briers_score_losses,  list of floats, 
        These are in same order as the folds.
        
        """
        #### assign all valid probability to one list
        valid_probab_preds_all = self.valid_probab_preds_byfold.stack().values #note this only works by chance! stack sorts by index
        self.valid_probab_preds_all = valid_probab_preds_all
        #### find the y_labels for full validation set in one list
        valid_y_true_all = self.valid_y_labels_byfold.stack().values
        self.valid_y_true_all = valid_y_true_all
        
        #### calculate scores for all CV validation model data.
        self.valid_average_precision_score_all = average_precision_score(valid_y_true_all, valid_probab_preds_all)
        self.valid_brier_score_loss_all = brier_score_loss(valid_y_true_all, valid_probab_preds_all)
        
        #### calc scores for each CV fold
        apscores = []
        bscores = []
        for fold_no in np.arange(1, self.cv_splits + 1):
            col_name = 'fold' + str(fold_no)
            y_true = self.valid_y_labels_byfold[col_name].dropna()
            y_prob_pred = self.valid_probab_preds_byfold[col_name].dropna()
            apscores.append(average_precision_score(y_true, y_prob_pred))
            bscores.append(brier_score_loss(y_true, y_prob_pred))
        
        self.CV_average_precision_scores = apscores
        self.CV_briers_score_losses = bscores
        
        self.CV_scores = pd.DataFrame(data={'average_precision': apscores, 'briers_score_loss':bscores})
        print('CV results')
        print('='*30)
        # print(self.CV_scores)
        print(self.CV_scores.describe())
        
        
        return
    
    def get_train_scores(self):
        "Calcs scores for training sets for each fold. NOTE: is this valid as each fold not independent."
        return
        
    def save_as_pickle(path):
        "Saves evaluation class as a pickle object."
        return
    
    def plot_PR_curve(self, y_true, y_prob_pred, title):
        "Plots AUC and Precision-Recall plot as one figures."
        
        precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob_pred)
        
        fig,ax = plt.subplots(1,2,figsize=(9,4))
        fig.suptitle('Cross-Validation ' + self.model_name + title)
        # prec-recall plot
        ax[0].plot(recalls[:-1], precisions[:-1],'g-') #,label='Reca')
        ax[0].set_ylabel('Precision')
        ax[0].set_xlabel('Recall')
        ax[0].legend(frameon=True,loc='center right')
        ax[0].set_ylim([0,1.1])
        ax[0].grid()

        # decision function plot
        ax[1].plot(thresholds, precisions[:-1], "b--", label="Precision")
        ax[1].plot(thresholds, recalls[:-1], "g-", label="Recall")
        ax[1].set_ylabel("Score")
        ax[1].set_xlabel("Decision Threshold")
#         ax[2].set_title("Threshold: prec-{0:0.3},recall-{1:0.3}".format(self.precision, self.recall))
        ax[1].legend(loc='best')
        ax[1].grid()
        ax[1].set_xlim([0,1])
        return
    
    def plot_PR_curve_each_valid_set(self):
        for fold in np.arange(1,self.cv_splits +1):
            col_name = 'fold' + str(fold)
            y_labels = self.valid_y_labels_byfold[col_name].dropna().values
            y_proba_pred = self.valid_probab_preds_byfold[col_name].dropna().values
            self.plot_PR_curve(y_labels, y_proba_pred, col_name)
        return
    
        return

    