import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

def show_plot(dataset):
    x = np.arange(2)
    y = [dataset.shape[0], dataset.shape[1]]
    plt.bar(x, y, align='center', alpha=0.5)
    plt.xticks(x, ('nr records', 'nr variables'))
    plt.title('Nr of records vs nr of variables')
    plt.show()

register_matplotlib_converters()
dataset1 = pd.read_csv('../CD databases/qsar_oral_toxicity.csv')
dataset2 = pd.read_csv('../CD databases/heart_failure_clinical_records_dataset.csv')

show_plot(dataset1)
show_plot(dataset2)

mv = {}
for var in dataset2:
    mv[var] = dataset2[var].isna().sum()
mv.values()