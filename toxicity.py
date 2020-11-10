import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from imblearn.under_sampling import AllKNN
from imblearn.under_sampling import NearMiss
import ds_functions as ds
from sklearn.feature_selection import SelectKBest, chi2

data: pd.DataFrame = pd.read_csv('../CD databases/qsar_oral_toxicity.csv')
y: np.ndarray = data.pop('classification').values
X: np.ndarray = data.values
labels = pd.unique(y)

trnX, tstX, trnY, tstY = train_test_split(X, y, train_size=0.7, stratify=y)


allknn = AllKNN()
nm = NearMiss()
smt = SMOTE()
ada = ADASYN(random_state=42)

lst = [allknn, nm, smt, ada]
gb = GradientBoostingClassifier(n_estimators=50, max_depth=10, learning_rate=0.5)

for samp in lst:
    trnX, trnY = samp.fit_resample(trnX, trnY)

    n_estimators = [50]
    max_depths = [10]
    learning_rate = [.5]
    best = ('', 0, 0)
    last_best = 0
    best_tree = None
    
    for k in range(len(max_depths)):
        d = max_depths[k]
        values = {}
        for lr in learning_rate:
            yvalues = []
            for n in n_estimators:
                gb.fit(trnX, trnY)
                prdY = gb.predict(tstX)
                yvalues.append(metrics.accuracy_score(tstY, prdY))
                if yvalues[-1] > last_best:
                    best = (d, lr, n)
                    last_best = yvalues[-1]
                    best_tree = gb
            values[lr] = yvalues
    
    prd_trn = best_tree.predict(trnX)
    prd_tst = best_tree.predict(tstX)
    ds.plot_evaluation_results(pd.unique(y), trnY, prd_trn, tstY, prd_tst)
    
'''
 UNDERSAMPLING
allknn = AllKNN()
trnX, trnY = allknn.fit_resample(trnX, trnY)

nm = NearMiss()
trnX, trnY = nm.fit_resample(trnX, trnY)

 OVERSAMPLING 
smt = SMOTE()
trnX, trnY = smt.fit_resample(trnX, trnY)

ada = ADASYN(random_state=42)
trnX, trnY = smt.fit_sample(trnX, trnY)

'''
 
best_vars = SelectKBest(chi2, k=100).fit_transform(X, y)
trnX, tstX, trnY, tstY = train_test_split(best_vars, y, train_size=0.7, stratify=y)

trnX, trnY = smt.fit_resample(trnX, trnY)

n_estimators = [50]
max_depths = [10]
learning_rate = [.5]
best = ('', 0, 0)
last_best = 0
best_tree = None

for k in range(len(max_depths)):
    d = max_depths[k]
    values = {}
    for lr in learning_rate:
        yvalues = []
        for n in n_estimators:
            gb.fit(trnX, trnY)
            prdY = gb.predict(tstX)
            yvalues.append(metrics.accuracy_score(tstY, prdY))
            if yvalues[-1] > last_best:
                best = (d, lr, n)
                last_best = yvalues[-1]
                best_tree = gb
        values[lr] = yvalues

prd_trn = best_tree.predict(trnX)
prd_tst = best_tree.predict(tstX)
ds.plot_evaluation_results(pd.unique(y), trnY, prd_trn, tstY, prd_tst)

