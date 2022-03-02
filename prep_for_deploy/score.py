
import os 
import json
import joblib
from pandas import json_normalize
import pandas as pd

def init():
    global model
    # Replace filename if needed.
    model_path = os.path.join(os.getenv('AZUREML_MODEL_DIR'), 'titanic_model.pkl')
    # Deserialize the model file back into a sklearn model.
    model = joblib.load(model_path)

def run(data):
    dict= json.loads(data)
    df = json_normalize(dict['data']) 
    df['loc']= df['cabin'].apply(lambda x: x[0] if pd.notnull(x) else 'X')
    df['hasFamily'] = (df['sibsp'] > 0) | (df['parch'] > 0)
    #note we remove the survived column
    cols_to_keep = ['pclass','sex','age','embarked','loc','hasFamily']
    df = df[cols_to_keep]
    
    print(df.isnull().sum())
    y_pred = model.predict(df)
    print(type(y_pred))
    
    #return json.dumps(y_pred)
    result = {"result": y_pred.tolist()}
    return result
