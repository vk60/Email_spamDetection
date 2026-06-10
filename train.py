import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import pickle

# __LOAD DATA ______

df=pd.read_csv('spam.csv',encoding='latin-1')
print(df.head())

# ___Features ANd Labels______
X=df['message']
y=df['label']

# Text_Vectorization______
vectorizer=TfidfVectorizer()
X_vectorizer=vectorizer.fit_transform(X)

# ---Train_test_split-------
x_train,x_test,y_train,y_test=train_test_split(X_vectorizer,y,test_size=0.2,random_state=42)

# ---Model------------------

model=LogisticRegression()
# -----Train model----------
model.fit(x_train,y_train)
# ----Predection------------
predection=model.predict(x_test)

# --Accuracy-----
accuracy=accuracy_score(y_test,predection)
print(f'Accuracy:{accuracy}')
# ----save model----------
pickle.dump(model,open('spam_model.pkl','wb'))

pickle.dump(vectorizer,open('vectorizer.pkl','wb'))

print("Model and vectorizer saved sucessfully")