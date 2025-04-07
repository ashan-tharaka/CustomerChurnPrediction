import pandas as pd
from sklearn.preprocessing import LabelEncoder
import seaborn as  sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pickle


df=pd.read_csv('TelcoCustomerChurn.csv')
print(df.head())

df.drop("customerID",axis=1,inplace=True)

df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')
print(df.head())
print(df.isnull().sum())

df.dropna(inplace=True)

#encode categorical features
le=LabelEncoder()
df['Churn']=le.fit_transform(df['Churn'])
df=pd.get_dummies(df,drop_first=True)
print(df.head())

sns.countplot(x='Churn',data=df)
plt.title('Churn Distribution')
plt.show()

plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

X = df.drop('Churn', axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

#save the model

with open("churn_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save feature names
with open("model_features.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)
