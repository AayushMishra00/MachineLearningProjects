from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

CSV_FILE = "/Users/aayushm/Documents/PythonProjects/Computer Vision/SignLanguageDetector/data_collection.csv"
#Loading the data
df = pd.read_csv(CSV_FILE)
#Separating features and output
x = df.drop("label", axis = 1)
y = df["label"]

#splitting the dataset 80-20 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#training the model
rf = RandomForestClassifier(n_estimators=100)
rf.fit(x_train, y_train)

y_pred = rf.predict(x_test)
print(classification_report(y_test,y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, xticklabels=rf.classes_, yticklabels=rf.classes_)
plt.show()

with open("sign_model.pkl", "wb") as f:
    pickle.dump(rf, f)

print("Model saved!")



