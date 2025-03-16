from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_absolute_error
import os

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def preprocess_data(df, target_column):
    df = df.dropna()
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    if y.dtype == 'O':
        y = LabelEncoder().fit_transform(y)
    
    X = pd.get_dummies(X)  # Convert categorical variables
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        target_column = request.form['target_column']
        
        if file and target_column:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            df = pd.read_csv(filepath)
            X_train, X_test, y_train, y_test = preprocess_data(df, target_column)
            
            models = {
                'Linear Regression': LinearRegression(),
                'Support Vector Regression': SVR(),
                'Decision Tree': DecisionTreeClassifier(),
                'K-Nearest Neighbors': KNeighborsClassifier(),
                'Random Forest': RandomForestClassifier(),
                'AdaBoost': AdaBoostClassifier(),
                'K-Means': KMeans(n_clusters=3)
            }
            
            results = {}
            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
                if isinstance(model, (LinearRegression, SVR)):
                    score = mean_absolute_error(y_test, y_pred)
                else:
                    score = accuracy_score(y_test, np.round(y_pred))
                
                results[name] = round(score, 4)
            
            return render_template('results.html', results=results)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
