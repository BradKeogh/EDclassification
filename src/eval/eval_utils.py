#### utils_eval
# Scripts originally copied from jupyter notebook in python-practice repo. See example use there. 

from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, roc_curve
from matplotlib import pyplot as plt

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
        
        fig,ax = plt.subplots(1,2,figsize=(9,4))
        fig.suptitle(self.plot_title)
        # prec-recall plot
        ax[0].plot(recalls[:-1],precisions[:-1],'g-') #,label='Reca')
        ax[0].set_ylabel('Precision')
        ax[0].set_xlabel('Recall')
        ax[0].legend(frameon=True,loc='center right')
        ax[0].set_ylim([0,1.1])
        # AUC plot
        ax[1].plot(fpr,tpr)
        ax[1].plot([0,1],[0,1],'k--') # 45def line
        ax[1].set_xlabel('F positive rate')
        ax[1].set_ylabel('T positve rate')
        
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
    
    