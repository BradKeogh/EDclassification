#### utils_eval
# Scripts originally copied from jupyter notebook in python-practice repo. See example use there. 

from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, roc_curve, accuracy_score, precision_score, recall_score
from matplotlib import pyplot as plt
import pandas as pd

#### example calls
# BinClassEval(lgr1, X_train, y_train)

# models = {'simple log reg': lgr1, 'reg log reg': grid_search}
# evals = CompareBinClassModels(models, X_train, y_train)


class BinClassEval():
    """ 
    Makes all evalutation you need to assess binary classifcation models.
    Suggested use for final evaluation once models have been decided. e.g. running on test set. 
    Built for sklearn models: some functionality requires model to have either a predict proba or decision function.
    """
    
    
    def __init__(self, model, X, y, plot_title='Model Evaluation', plot = False):
        """
        Input
        -----
        Model, sklearn model, (that has had .fit() method called aready).
        X, df/numpy array, containing features
        y, df/numpy array, containing binary target
        """
        #### assign inputs
        self.model = model
        self.X = X
        self.y = y
        self.plot_title = plot_title
        
        self.proba_pred_avail = False
        
        #### calc prob and decision functions (where possible). 
        ### NOTE: these are assigned to the same proba_preds attribute. If predict_prob avail then this takes precident.
        if hasattr(model, 'decision_function'):
            predDF = model.decision_function(X) # warning, some model dont have DF
            self.proba_preds = predDF
            self.DF = predDF
            print('Model has decision_function.')
            
        if hasattr(model, 'predict_proba'):
            proba_preds = model.predict_proba(X)[:,1]
            self.proba_preds = proba_preds
            self.proba_preds_avail = True
            print('Model has predict_proba.')
            
        self.label_preds = self.model.predict(X)
            
        #### run evaluation
        self.AUC()
        self.F1()
        self.accuracy()
        self.precision()
        self.recall()
        # self.f1_manual()
        
        if plot == True:
            self.plot_AUC_PR()
            
        return
    
    def AUC(self):
        "Prints and returns AUC score."
        AUC = roc_auc_score(self.y, self.proba_preds).round(3)
        self.AUC = AUC
        print('AUC: ', AUC) # NOTE: sklearn doc says use prob, if not use decision function.
        return
    
    def F1(self):
        "Prints and returns F1 score."
        F1 = f1_score(self.y, self.label_preds).round(3)
        self.F1 = F1
        print('F1 score: ', F1)
        return

    def precision(self):
        "Prints precision score."
        precision = precision_score(self.y, self.label_preds).round(3)
        self.precision = precision
        print('precision score: ', precision)
        return
    
    def recall(self):
        "Prints recall score."
        recall = recall_score(self.y, self.label_preds).round(3)
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
        accuracy = accuracy_score(self.y, self.label_preds).round(3)
        self.accuracy = accuracy
        print('accuracy: ', accuracy)
        return
    
    def confusion_matrix(self):
        "Prints confusion matrix."
        return
    
    def plot_AUC_PR(self):
        "Plots AUC and Precision-Recall plot as one figures."
        
        precisions, recalls, thresholds = precision_recall_curve(self.y, self.proba_preds)

        fpr,tpr,thresholds_ROC = roc_curve(self.y, self.proba_preds)
        
        fig,ax = plt.subplots(1,3,figsize=(13,4))
        fig.suptitle(self.plot_title)
        # prec-recall plot
        ax[0].plot(recalls[:-1], precisions[:-1],'g-') #,label='Reca')
        ax[0].set_ylabel('Precision')
        ax[0].set_xlabel('Recall')
        ax[0].legend(frameon=True,loc='center right')
        ax[0].set_ylim([0,1.1])
        ax[0].grid()
        # AUC plot
        ax[1].plot(fpr, tpr)
        ax[1].plot([0,1],[0,1],'k--') # 45def line
        ax[1].set_xlabel('F positive rate')
        ax[1].set_ylabel('T positve rate')
        ax[1].grid()
        # decision function plot
        ax[2].plot(thresholds, precisions[:-1], "b--", label="Precision")
        ax[2].plot(thresholds, recalls[:-1], "g-", label="Recall")
        ax[2].set_ylabel("Score")
        ax[2].set_xlabel("Decision Threshold")
        ax[2].set_title("Threshold: prec-{0:0.3},recall-{1:0.3}".format(self.precision, self.recall))
        ax[2].legend(loc='best')
        ax[2].grid()
        ax[2].set_xlim([0,1])
        return


class BinClassEvalCV():
    """ 
    Makes all evalutation you need to assess binary classifcation models.
    Suggested use for final evaluation once models have been decided. e.g. running on test set. 
    Built for sklearn models: some functionality requires model to have either a predict proba or decision function.
    """
    
    
    def __init__(self, model, X, y, plot_title='Model Evaluation', plot = False):
        """
        Input
        -----
        Model, sklearn model, (that has had .fit() method called aready).
        X, df/numpy array, containing features
        y, df/numpy array, containing binary target
        """
        #### assign inputs
        self.model = model
        self.X = X
        self.y = y
        self.plot_title = plot_title
        
        self.proba_pred_avail = False
        
        #### calc prob and decision functions (where possible). 
        ### NOTE: these are assigned to the same proba_preds attribute. If predict_prob avail then this takes precident.
        if hasattr(model, 'decision_function'):
            predDF = model.decision_function(X) # warning, some model dont have DF
            self.proba_preds = predDF
            self.DF = predDF
            print('Model has decision_function.')
            
        if hasattr(model, 'predict_proba'):
            proba_preds = model.predict_proba(X)[:,1]
            self.proba_preds = proba_preds
            self.proba_preds_avail = True
            print('Model has predict_proba.')
            
        self.label_preds = self.model.predict(X)
            
        #### run evaluation
        self.AUC()
        self.F1()
        self.accuracy()
        self.precision()
        self.recall()
        # self.f1_manual()
        
        if plot == True:
            self.plot_AUC_PR()
            
        return
    
    def AUC(self):
        "Prints and returns AUC score."
        AUC = roc_auc_score(self.y, self.proba_preds).round(3)
        self.AUC = AUC
        print('AUC: ', AUC) # NOTE: sklearn doc says use prob, if not use decision function.
        return
    
    def F1(self):
        "Prints and returns F1 score."
        F1 = f1_score(self.y, self.label_preds).round(3)
        self.F1 = F1
        print('F1 score: ', F1)
        return

    def precision(self):
        "Prints precision score."
        precision = precision_score(self.y, self.label_preds).round(3)
        self.precision = precision
        print('precision score: ', precision)
        return
    
    def recall(self):
        "Prints recall score."
        recall = recall_score(self.y, self.label_preds).round(3)
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
        accuracy = accuracy_score(self.y, self.label_preds).round(3)
        self.accuracy = accuracy
        print('accuracy: ', accuracy)
        return
    
    def confusion_matrix(self):
        "Prints confusion matrix."
        return
    
    def plot_AUC_PR(self):
        "Plots AUC and Precision-Recall plot as one figures."
        
        precisions, recalls, thresholds = precision_recall_curve(self.y, self.proba_preds)

        fpr,tpr,thresholds_ROC = roc_curve(self.y, self.proba_preds)
        
        fig,ax = plt.subplots(1,3,figsize=(13,4))
        fig.suptitle(self.plot_title)
        # prec-recall plot
        ax[0].plot(recalls[:-1], precisions[:-1],'g-') #,label='Reca')
        ax[0].set_ylabel('Precision')
        ax[0].set_xlabel('Recall')
        ax[0].legend(frameon=True,loc='center right')
        ax[0].set_ylim([0,1.1])
        ax[0].grid()
        # AUC plot
        ax[1].plot(fpr, tpr)
        ax[1].plot([0,1],[0,1],'k--') # 45def line
        ax[1].set_xlabel('F positive rate')
        ax[1].set_ylabel('T positve rate')
        ax[1].grid()
        # decision function plot
        ax[2].plot(thresholds, precisions[:-1], "b--", label="Precision")
        ax[2].plot(thresholds, recalls[:-1], "g-", label="Recall")
        ax[2].set_ylabel("Score")
        ax[2].set_xlabel("Decision Threshold")
        ax[2].set_title("Threshold: prec-{0:0.3},recall-{1:0.3}".format(self.precision, self.recall))
        ax[2].legend(loc='best')
        ax[2].grid()
        ax[2].set_xlim([0,1])
        return


from sklearn.model_selection import cross_val_score

class CompareBinClassModels():
    """
    Takes list of models and evaluates each on same set of data.
    
    This may take some time to run as currently retrains each model on each fold(4).
    """
    def __init__(self, models, X, y):
        """
        Input
        ------
        models, dict, name (string) and sklearn model pairs (that have been .fit()),
        X, df/np array , with features,
        y, df/np array , with target (binary),
        
        """
        #### assign inputs as attributes
        self.models = models
        self.X = X
        self.y = y
        
        #### run evals
        self.run_evaluations()
        
        return
    
    def run_evaluations(self):
        #### instantiate each model evaluation object
        names = []
        means = []
        stds = []
        print('start run_evals')
        
        for modelname in models:
            model = models[modelname]
#             print('in loop', model)
            
            scores = get_cross_validation_scores(model, self.X, self.y, 'f1')
            names.append(modelname)
            means.append(np.mean(scores))
            
            stds.append(np.std(scores))
            
#             print(scores)
            
#             m_eval_obj = BinClassEval(model, X, y, plot=False) # create model eval object
            # save F1 score
#             modelevals[modelname] = m_eval_obj # save evaluation object
            
        #### make model evaluations, scores etc into df
        cols = ['model_name', 'mean', 'std']
        df = pd.DataFrame([names, means, stds], index=cols).T
        self.modelevals = df
        
        return
    
    def compare_models(self):
        pass
    
def get_cross_validation_scores(model, X, y, scoring='f1'):
    "Calc cross validation scores."
    scores = cross_val_score(model, X, y, cv=4, scoring=scoring)
    return(scores)
    

def save_model_to_log(path_to_log, dataversion, model, gridsearch, notes = None, df_row=None):
    """
    Open log df, add new row with model + training info. Save df to pickle again.
    
    Input
    =====
    path_to_log, str, string to location of pikcle file.
    dataversion, str, name of data version.
    model, sklearn model object, which has been fit.  
    gridsearch, sklearn gridsearch object, which has been fit.
    notes, str, (optional) text describing model.
    df_row, int, (optional) integer row to replace in dataframe. Suggested if re-reporting an updated model which has already been logged.
    
    """
    model_log = pd.read_pickle(path_to_log)
    #### get gridsearch results out of gridsearch object.
    gridsearch_results = pd.DataFrame(gridsearch.cv_results_)
    mean_train = gridsearch_results.query('rank_test_score == 1')['mean_train_score'].values.round(3)
    mean_valid = gridsearch_results.query('rank_test_score == 1')['mean_test_score'].values.round(3)
    
    model_log = model_log.append({'dataV': dataversion,'model':model,'gridsearch':gridsearch_results,'mean_train': mean_train,
                                  'mean_valid':mean_valid ,'notes': notes}, ignore_index=True)
    model_log.to_pickle(path_to_log)
    return(print('Model logged.'))
