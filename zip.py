import zipfile
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, mean_squared_error, r2_score
import joblib
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
from collections import Counter


def unzip_file(zip_file, output_dir):
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"Extracted {zip_file} to {output_dir}")
    except Exception as e:
        print(f"Failed to extract {zip_file}: {e}")

os.makedirs("extracted_files", exist_ok=True)

unzip_file("sales-transactions-dataset.zip", "extracted_files")
unzip_file("classification-for-sandstone-and-shale.zip", "extracted_files")

print(os.listdir("extracted_files"))

test_data = pd.read_excel("extracted_files/Test.xlsx")
train_data = pd.read_excel("extracted_files/Train.xlsx")
data1 = pd.read_csv("extracted_files/Data1.txt", delimiter="\t")

print(test_data.columns)
print("Test Data Head:")
print(test_data.head())
print("\nTrain Data Head:")
print(train_data.head())
print("Data1.txt Head:")
print(data1.head())

test_data_200 = test_data.head(200)
train_data_200 = train_data.head(200)
data1_200= data1.head(200)

test_data_200.to_csv("For_Prediction_Test.csv", index=False)
train_data_200.to_csv("For_Prediction_Train.csv", index=False)
data1_200.to_csv("For_Prediction_Data1.csv", index=False)

print("First 200 rows saved as For_Prediction_Test.csv, For_Prediction_Train.csv and For_Prediction_Data1.csv")

print(test_data.isnull().sum())
print(train_data.isnull().sum())
print(data1.isnull().sum())

test_data = test_data.fillna(0)
train_data = train_data.fillna(0)
data1 = data1.fillna(0)

if 'Unnamed: 2' in data1.columns:
    data1 = data1.drop(columns=['Unnamed: 2'])

data1['effporosity'] = pd.to_numeric(data1['effporosity'], errors='coerce')
data1['effporosity'].fillna(0, inplace=True) 

print(test_data.dtypes)
print(train_data.dtypes)
print(data1.dtypes)

test_data.hist(figsize=(10, 8))
plt.show()

train_data.hist(figsize=(10, 8))
plt.show()

data1.hist(figsize=(10, 8))
plt.show()

sns.boxplot(data=test_data)
plt.show()

sns.boxplot(data=train_data)
plt.show()

sns.boxplot(data=data1)
plt.show()

print(test_data.columns)

test_data1 = test_data.head(10)

for col in ['ReportID', 'SalesPersonID', 'ProductID']:
    le = LabelEncoder()
    test_data1[col] = le.fit_transform(test_data1[col])

X = test_data1.drop(columns=['TotalSalesValue'])
y = test_data1['TotalSalesValue'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training data shape:", X_train.shape)
print("Test data shape:", X_test.shape)

reg_model = LinearRegression()
reg_model.fit(X_train, y_train)

sales_predictions = reg_model.predict(X_test)
mse = mean_squared_error(y_test, sales_predictions)
r2 = r2_score(y_test, sales_predictions)

print("Mean Squared Error (MSE):", mse)
print("R-squared Score (R²):", r2)

model = RandomForestClassifier(random_state=1)
model.fit(X_train, y_train)

print("Class distribution before resampling:", Counter(y_train))

#smote = SMOTE(random_state=1, k_neighbors=0)
#X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
#model.fit(X_train_resampled, y_train_resampled)

ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

print("Class distribution after resampling:", Counter(y_train_resampled))
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")

joblib.dump(model, 'trained_model.pkl')
print("Model saved as 'trained_model.pkl'")

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

feature_importances = model.feature_importances_
importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)
sns.barplot(x='Importance', y='Feature', data=importance_df)
plt.title("Feature Importances")
plt.show()

test_data_200_processed = test_data.head(200).copy()
for col in ['ReportID', 'SalesPersonID', 'ProductID']:
    test_data_200_processed[col] = le.fit_transform(test_data_200_processed[col])

X_new = test_data_200_processed.drop(columns=['TotalSalesValue'])
predictions = model.predict(X_new)
print("Predictions on new data:", predictions)
