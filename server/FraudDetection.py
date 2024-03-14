from flask import Flask, request, jsonify
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

readCsv=r'F:\Artificial Intelligence\Machine Learning\3_ML Playlist\Machine Learning Projects\1_Deployement\Fraud Detection\server\train.csv'
df=pd.read_csv(readCsv)
df.drop(columns=['id'],inplace=True)
df['title'].fillna('The Dark Agenda Behind Globalism And Open Borders',inplace=True)
df['author'].fillna('Pam Key',inplace=True)
df.dropna(subset=['text'],inplace=True)
year=df['title'].value_counts()
a=year[year<3]
df['title']=df['title'].apply(lambda x: 'The Dark Agenda Behind Globalism And Open Borders' if x in a else x )
df['title']=df['title'].str.split(' ').str.slice(0,4).str.join(' ')
year=df['author'].value_counts()
a=year[year<5]
df['author']=df['author'].apply(lambda x: 'Pam Key' if x in a else x )
year=df['text'].value_counts()
a=year[year<4]
df['text']=df['text'].apply(lambda x: 'The Dark Agenda Behind Globalism And Open Borders' if x in a else x )
x=df.drop(columns=['label'])
y=df[['label']]
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=42)
OHE=OneHotEncoder(sparse=False)
xtrain=OHE.fit_transform(xtrain)
xtest=OHE.transform(xtest)
LR=SVC(kernel='linear')
LR.fit(xtrain,ytrain)
print(' i have been called')
def Predict_Price(xtest):
    xtest=OHE.transform(xtest)
    ypred=LR.predict(xtest)
    return ypred 

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    response = Predict_Price([[data['title'],data['author'],data['text']]])
    response = np.array(response)
    response_list = response.tolist()
    return jsonify({'response': response_list})
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)        