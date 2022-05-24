import pandas as pd
import numpy as np
import xlrd
from sklearn.preprocessing import LabelEncoder

#data = xlrd.open_workbook("C:/Data/DATA.xlrd")
data = pd.read_excel("C:/Data/DATA.xlsx")

data.isnull().sum()
data.dropna()
data.columns
data = data.drop(["Mode_Of_Transport"], axis = 1)
data = data.drop(["Patient_ID"], axis = 1)
data = data.drop(["Test_Booking_Date"], axis = 1)
data = data.drop(["Sample_Collection_Date"], axis = 1)
data = data.drop(["Patient_Gender"], axis = 1)
data.columns


# Converting into binary
lb = LabelEncoder()
#data["Patient_Gender"] = lb.fit_transform(data["Patient_Gender"])
data["Test_Name"] = lb.fit_transform(data["Test_Name"])
data["Sample"] = lb.fit_transform(data["Sample"])
data["Way_Of_Storage_Of_Sample"] = lb.fit_transform(data["Way_Of_Storage_Of_Sample"])
#data["Test_Booking_Date"] = lb.fit_transform(data["Test_Booking_Date"])
data["Traffic_Conditions"] = lb.fit_transform(data["Traffic_Conditions"])
data["Cut-off Schedule"] = lb.fit_transform(data["Cut-off Schedule"])
#data["Test_Booking_Date"] = lb.fit_transform(data["Sample_Collection_Date"])
data["Reached_On_Time"] = lb.fit_transform(data["Reached_On_Time"])


#data["default"]=lb.fit_transform(data["default"])

data['Reached_On_Time'].unique()
data['Reached_On_Time'].value_counts()
colnames = list(data.columns)

predictors = colnames[:15]
target = colnames[15]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(data, test_size = 0.28)

from sklearn.tree import DecisionTreeClassifier as DT

#help(DT)
model = DT(criterion = 'entropy',max_depth=4)
model.fit(train[predictors], train[target])

 
# Prediction on Test Data
preds = model.predict(test[predictors])
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predictions'])

np.mean(preds == test[target]) # Test Data Accuracy 

# Prediction on Train Data
preds = model.predict(train[predictors])
pd.crosstab(train[target], preds, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(preds == train[target]) # Train Data Accuracy

# saving the model
# importing pickle
import pickle
pickle.dump(model, open('model.pkl', 'wb'))

# load the model from disk
model = pickle.load(open('model.pkl', 'rb'))

# checking for the results
list_value = pd.DataFrame(data.iloc[0:1,:15])
list_value

print(model.predict(list_value))
