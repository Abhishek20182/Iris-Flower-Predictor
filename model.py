#Python Libraries 
import pandas as pd
from sklearn.svm import LinearSVC
import pickle

iris_df = pd.read_csv('archive.zip')
print(iris_df.head())
X = iris_df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = iris_df['species']

lsvc = LinearSVC(max_iter=1000)
lsvc.fit(X, y)

pickle.dump(lsvc, 'model.pkl')
