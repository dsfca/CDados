import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
data = pd.read_csv('../CD databases/qsar_oral_toxicity.csv')
data1 = pd.read_csv('../CD databases/heart_failure_clinical_records_dataset.csv')

columns = data.select_dtypes(include='number').columns
rows, cols = len(columns)-1, len(columns)-1
#rows,cols= 10, 10

#columns= columns[485:495]


plt.figure()
fig, axs = plt.subplots(rows, cols, figsize=(cols*4, rows*4), squeeze=False)
for i in range(len(columns)):
    var1 = columns[i]
    for j in range(i+1, len(columns)):
        var2 = columns[j]
        axs[i, j-1].set_title("%s x %s"%(var1,var2))
        axs[i, j-1].set_xlabel(var1)
        axs[i, j-1].set_ylabel(var2)
        axs[i, j-1].scatter(data[var1], data[var2])
plt.show()

import seaborn as sns
fig = plt.figure(figsize=[12, 12])
corr_mtx = data.corr()
sns.heatmap(corr_mtx, xticklabels=corr_mtx.columns, yticklabels=corr_mtx.columns, annot=True, cmap='Blues')
plt.title('Correlation analysis')
plt.show()

def check_corr():
    cont = 0
    for i in range(len(corr_mtx)):
        for j in range(len(corr_mtx)):
            if corr_mtx.iat[i,j] < -0.75 or corr_mtx.iat[i,j] > 0.75 and corr_mtx.iat[i,j] != 1:
                #print(corr_mtx[i,j])
                #print(i)
                #print(j)
                #return
                cont = cont + 1
    print(cont)
    
#1- retirar variaveis com baixa correlacao. threshold ira ser tipo 90%. ver que variaveis
#estao nesse threshold e mante las. 

#2- representar com diferentes cores o outcome: negativo ou positivo para o toxicity, ie

#3- Alterar a coluna do outcome para int