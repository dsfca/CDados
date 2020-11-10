import pandas as pd
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
tox_data = pd.read_csv('../CD databases/qsar_oral_toxicity.csv')

corr_mtx = [tox_data.corr()]

high_vars = []
#le cada uma das corr_mtx e conta o numero de dados acima de um threshold
for data in corr_mtx:
    for i in range(len(data)):
        for j in range(len(data)):
            if data.iat[i,j] < -0.75 or data.iat[i,j] > 0.75 and data.iat[i,j] != 1:
                high_vars.append(j) if j not in high_vars else high_vars

high_vars_str = [str(i) for i in high_vars]
c=0
for column in high_vars_str:
    del tox_data[column]
    c = c+1
print(c)
tox_data.to_csv(r'C:\Users\jocam\OneDrive\Documentos\Data Science IST\CD databases\toxicity_reduced.csv', index = False)

#retirar variaveis com baixa correlacao. threshold ira ser tipo 90%. ver que variaveis
#estao nesse threshold e mante las. 
