import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

heart_dataset = pd.read_csv('../CD databases/heart_failure_clinical_records_dataset.csv')

head_list = heart_dataset.columns.tolist()

x = list(range(0,299))

for column_name in head_list[:-1]:
    y = heart_dataset[column_name].tolist()
    plt.scatter(x, y)
    plt.title(column_name)
    plt.show()
