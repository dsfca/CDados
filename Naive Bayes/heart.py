import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('../CD databases/heart_failure_clinical_records_dataset.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 12].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

ratio = (cm[0][0] + cm[1][1]) / len(X_test)