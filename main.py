import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv("parkinsons.csv")

x = df[['DFA', 'HNR', 'RPDE', 'PPE']]
y = df[['status']]

sns.pairplot(df, vars=['DFA', 'HNR', 'RPDE', 'PPE'], hue='status')
plt.suptitle('Pair Plot of Numerical Features by status', y=1.02)
plt.show()

scaler = MinMaxScaler()
x_scale = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scale, y, test_size=0.2)

dt = SVC(kernel='linear')
dt.fit(x_train, y_train)

y_predict = dt.predict(x_test)
accuracy = accuracy_score(y_test, y_predict)

joblib.dump(dt, 'parkinsons.joblib')
