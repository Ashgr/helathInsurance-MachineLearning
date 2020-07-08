import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import pickle

data = pd.read_csv("insurance.csv",sep = ",")
data = data[["age","sex","bmi","children","smoker","region","charges"]]

"""
Change the Strings Data to Numeric Data
    Male = 0 Female = 1
    unSmoker = 0 Smoker = 1
    northeast = 1
    southeast = 2
    southwest = 3
    northwest = 4
"""

data.loc[(data['sex'] == 'male'), ['sex'] ] =  0
data.loc[(data['sex'] == 'female'), ['sex'] ] =  1
data.loc[(data['smoker'] == 'yes'), ['smoker'] ] =  1
data.loc[(data['smoker'] == 'no'), ['smoker'] ] =  0
data.loc[(data['region'] == 'northeast'), ['region'] ] =  1
data.loc[(data['region'] == 'southeast'), ['region'] ] =  2
data.loc[(data['region'] == 'southwest'), ['region'] ] =  3
data.loc[(data['region'] == 'northwest'), ['region'] ] =  4

# The attribute we want to predict
predict = "charges"


X = np.array(data.drop([predict],1))
Y = np.array(data[predict])

X_Train , X_Test , Y_Train , Y_Test = sklearn.model_selection.train_test_split(X,Y, test_size= 0.2)

best_accuracy = 0
for i in range(10):
    Linear = linear_model.LinearRegression()
    Linear.fit(X_Train, Y_Train)
    acc = Linear.score(X_Test,Y_Test)
    if acc>best_accuracy:
        best_accuracy = acc
        with open("healthModel.pickle", "wb") as f:
            pickle.dump(Linear, f)

pickle_in = open("healthModel.pickle", "rb")
Linear = pickle.load(pickle_in)
accuracy = Linear.score(X_Test,Y_Test)
predictions = Linear.predict(X_Test)

print("Model accuracy: ",accuracy)
for prd in range(len(predictions)):
    print(predictions[prd] , X_Test[prd] , Y_Test[prd])
    """   Predicted Data ,  Other attributes , The original data   """