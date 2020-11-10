import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE

neighborsAmount = 3

dataset = pd.read_csv('../CD databases/qsar_oral_toxicity.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1024].values

# Encoding categorical data
ct = ColumnTransformer(transformers= [('encoder', OneHotEncoder(), [0])], remainder='passthrough')
y = LabelEncoder().fit_transform(y)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)

''' OVERSAMPLING '''
smt = SMOTE(random_state=0)
X_train, y_train = smt.fit_sample(X_train, y_train)

classifier = KNeighborsClassifier(n_neighbors = neighborsAmount, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

ratio = (cm[0][0] + cm[1][1]) / len(X_test)