import pandas as pd

Data = pd.read_csv('data.csv')

X = Data["Email Text"].values
y = Data["Email Type"].values

from sklearn.model_selection import train_test_split
X_train,x_test,y_train,y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# define the Classifier
classifier = Pipeline([("tfidf",TfidfVectorizer() ),("classifier",RandomForestClassifier(n_estimators=10))])

classifier.fit(X_train,y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import classification_report,accuracy_score,confusion_matrix

print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))

import joblib
joblib.dump(classifier,'eTc.sav')