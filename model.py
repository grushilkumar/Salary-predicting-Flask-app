import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor 
from word2number import w2n

df = pd.read_csv('data.csv')
df['experience'].fillna('zero', inplace=True)
df.experience = df.experience.apply(w2n.word_to_num)
# Call the Get_dummies function
X = df.drop('salary', axis=1)
y = df.salary
X_train, X_test, y_train, y_test = train_test_split (X, y, test_size=0.33, random_state=0) 

model = DecisionTreeRegressor(max_depth = 4, random_state = 42)
model.fit(X_train, y_train)
pred_tree = model.predict(X_test)

# save the model to disk
filename = 'first.sav'
pickle.dump(model, open(filename, 'wb'))
