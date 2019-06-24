from flask import Flask
from flask import jsonify,request
import pickle
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import Mapping as mp
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)


tfidf = TfidfVectorizer()
@app.route("/predict",methods=['GET'])
def predict():
	df= pd.read_csv("Job titles and industries.csv")
	# Features and Labels
	df['label'] = df['industry'].map({'IT': 0, 'Marketing': 1,'Education':2,'Accountancy':3})
	x = df['job title']
	y = df['label']
	
	# Extract Feature With CountVectorizer
	tfidf = TfidfVectorizer()
	x = tfidf.fit_transform(df["job title"]).toarray() # Fit the Data

	from sklearn.model_selection import train_test_split
	x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)


	model=RandomForestClassifier(n_estimators=100)
	model.fit(x_train,y_train)

	jop_title=request.args.get('jop_title')
	data=mp.run(jop_title)
	vect=tfidf.transform([data]).toarray()
	result = model.predict(vect)

	if result[0]==0:
		return "The Result of The Prediction: <b> IT"
	elif result[0]==1:
		return "The Result of The Prediction: <b> Marketing"
	elif result[0]==2:
		return "The Result of The Prediction: <b> Education"
	elif result[0]==3:
		return "The Result of The Prediction: <b> Accountancy"




	
	
	
if __name__ == '__main__':
    app.run(port = 9000, debug = True)	