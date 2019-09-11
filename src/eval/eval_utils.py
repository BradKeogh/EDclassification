#### utils_eval
# Scripts originally copied from jupyter notebook in python-practice repo. See example use there. 

from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, roc_curve, accuracy_score, precision_score, recall_score
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import pickle

#### example calls
# BinClassEval(lgr1, X_train, y_train)

# models = {'simple log reg': lgr1, 'reg log reg': grid_search}
# evals = CompareBinClassModels(models, X_train, y_train)

from sklearn.model_selection import TimeSeriesSplit

class BinLabelEval():
    """ 
    Makes all evalutation you need to assess binary classifcation models.
    Suggested use for final evaluation once models have been decided. e.g. running on test set. 
    Built for sklearn models: some functionality requires model to have either a predict proba or decision function.
    """
    
    
    def __init__(self, y_pred, y_true):
        """
        Input
        -----
        Model, sklearn model, (that has had .fit() method called aready).
        X, df/numpy array, containing features
        y, df/numpy array, containing binary target
        """
        #### assign inputs
        self.y_pred = y_pred
        self.y_true = y_true
            
        #### run evaluation
        self.AUC()
        self.F1()
        self.accuracy()
        self.precision()
        self.recall()
        # self.f1_manual()

        return
    
    def AUC(self):
        "Prints and returns AUC score."
        AUC = roc_auc_score(self.y_pred, self.y_true).round(3)
        self.AUC = AUC
        print('AUC: ', AUC) # NOTE: sklearn doc says use prob, if not use decision function.
        return
    
    def F1(self):
        "Prints and returns F1 score."
        F1 = f1_score(self.y_pred, self.y_true).round(3)
        self.F1 = F1
        print('F1 score: ', F1)
        return

    def precision(self):
        "Prints precision score."
        precision = precision_score(self.y_pred, self.y_true).round(3)
        self.precision = precision
        print('precision score: ', precision)
        return
    
    def recall(self):
        "Prints recall score."
        recall = recall_score(self.y_pred, self.y_true).round(3)
        self.recall = recall
        print('recall scare: ', recall)
        return

    def f1_manual(self):
        " calcs and prints f1 score from precision and recall scores. (Rather than using f1_score metric)."
        precision = self.precision
        recall = self.recall
        F1 = (2 * (precision * recall) / (precision + recall)).round(3)
        self.f1_manual = F1
        print('F1 manual calc: ', F1)
        return
    
    def accuracy(self):
        "Prints and returns accuracy score."
        accuracy = accuracy_score(self.y_pred, self.y_true).round(3)
        self.accuracy = accuracy
        print('accuracy: ', accuracy)
        return
    
    def confusion_matrix(self):
        "Prints confusion matrix."
        return

        return

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

        # retrain model on all available data so ready for test predictions
        self.retrain_model()
        
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

    def retrain_model(self):
        "retrain model on all available data so ready for test predictions."
        self.model.fit(self.X, self.y)
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
            
            train_proba_preds_split = self.model.predict_proba(X_train)[:,1].tolist() # make predictions to this fold of train data
            self.train_probab_preds_byfold.loc[train_index, fold_name] = train_proba_preds_split # assign fold to dataframe as new col
            
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
        #### assign all validation probability predictions to one list
        valid_probab_preds_all = self.valid_probab_preds_byfold.stack().values #note this only works by chance! stack sorts by index
        self.valid_probab_preds_all = valid_probab_preds_all
        #### find the y_labels for full validation set in one list
        valid_y_true_all = self.valid_y_labels_byfold.stack().values
        self.valid_y_true_all = valid_y_true_all
        
        #### calculate scores for all CV validation model data.
        self.valid_average_precision_score_all = average_precision_score(valid_y_true_all, valid_probab_preds_all)
        self.valid_brier_score_loss_all = brier_score_loss(valid_y_true_all, valid_probab_preds_all)
        
        #### calc scores for each CV fold
        valid_apscores = []
        valid_bscores = []
        train_apscores = []
        train_bscores = []

        for fold_no in np.arange(1, self.cv_splits + 1):
            col_name = 'fold' + str(fold_no)
            # validation scores
            y_true = self.valid_y_labels_byfold[col_name].dropna()
            y_prob_pred = self.valid_probab_preds_byfold[col_name].dropna()
            valid_apscores.append(average_precision_score(y_true, y_prob_pred))
            valid_bscores.append(brier_score_loss(y_true, y_prob_pred))
            # training scores
            y_true = self.train_y_labels_byfold[col_name].dropna()
            y_prob_pred = self.train_probab_preds_byfold[col_name].dropna()
            train_apscores.append(average_precision_score(y_true, y_prob_pred))
            train_bscores.append(brier_score_loss(y_true, y_prob_pred))
        
        # dont think these attributes are used anywhere else as more convienient in df form below
        self.validCV_average_precision_scores = valid_apscores
        self.validCV_briers_score_losses = valid_bscores

        self.trainCV_average_precision_scores = train_apscores
        self.trainCV_briers_score_losses = train_bscores
        
        self.validCV_scores = pd.DataFrame(data={'average_precision': valid_apscores, 'briers_score_loss': valid_bscores})
        self.trainCV_scores = pd.DataFrame(data={'average_precision': train_apscores, 'briers_score_loss': train_bscores})

        print('INPUT DATA')
        print('='*10)
        self.print_data_facts()
        print('='*30)
        print('CV RESULTS')
        self.print_scores()
        print('='*30)

        return

    def print_scores(self, metrics = ['average_precision', 'briers_score_loss']):
        "Print results from train and validation sets. results_df in format with columns "
        for metric in metrics:
            print('='*10)
            print(metric)
            tmean = "{0:.3f}".format(self.trainCV_scores[metric].mean())
            tstd = "{0:.3f}".format(self.trainCV_scores[metric].std())
            print("TRAIN MEAN (std): ", tmean, "(", tstd ,")")
            vmean = "{0:.3f}".format(self.validCV_scores[metric].mean())
            vstd = "{0:.3f}".format(self.validCV_scores[metric].std())
            print("VALID MEAN (std): ", vmean, "(", vstd ,")")
            diff_means = "{0:.3f}".format(self.trainCV_scores[metric].mean() - self.validCV_scores[metric].mean())
            print("DIFF BETWEEN MEAN: ", diff_means)


        return
    
    def get_train_scores(self):
        "Calcs scores for training sets for each fold. NOTE: is this valid as each fold not independent."
        return
        
    def save_as_pickle(path, filename):
        "Saves evaluation class as a pickle object. NOTE: Not currently working."
        outfile = open(filename,'wb')
        pickle.dump(self, outfile) # 
        outfile.close()
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
        "Loop through folds and plot a PR curve for each fold."

        def print_label_info(array):
            "given array of labels print size and proportion of labels = 1."
            ones = array.sum()
            class_fraction = ones/len(array)
            print("VALIDATION SAMPLES: " + str(len(array)) )
            print("CLASS 1 PROPORTION: " + '{0:.2f}'.format(class_fraction))
            return

        for fold in np.arange(1,self.cv_splits +1):
            col_name = 'fold' + str(fold)
            y_labels = self.valid_y_labels_byfold[col_name].dropna().values
            y_proba_pred = self.valid_probab_preds_byfold[col_name].dropna().values
            self.plot_PR_curve(y_labels, y_proba_pred, col_name)
            print('-'*20)
            print(col_name)
            print_label_info(y_labels)
        return

    def print_data_facts(self):
        "Calc and print class imbalance from a numpy array."
        ones = self.y.sum()
        class_fraction = ones/len(self.y)
        print("FEATURES:" + str(self.X.shape[1]) )
        print("TRAINING SAMPLES: " + str(self.X.shape[0]) )
        print("CLASS 1 PROPORTION: " + '{0:.2f}'.format(class_fraction))
        return
    
        return





