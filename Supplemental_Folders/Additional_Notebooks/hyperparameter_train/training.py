
import os
import sys
import argparse
import joblib
import pandas as pd
import numpy as np

from azureml.core import Run, Dataset, Workspace, Experiment

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score,roc_curve

# Calculate model performance metrics
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt

from azureml.core import Model
from azureml.core.resource_configuration import ResourceConfiguration

def getRuntimeArgs():
    parser = argparse.ArgumentParser()
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str)
    parser.add_argument("--penalty_term", type=str, dest = 'penalty_term', default = 'l2', help = 'penalty term')
    parser.add_argument('--C', type=float, dest='C', default=0.1, help='learning rate')
    args = parser.parse_args()
    return args

def buildpreprocessorpipeline(X_raw):
    categorical_features = X_raw.select_dtypes(include=['object']).columns
    numeric_features = X_raw.select_dtypes(include=['float','int64']).columns

    categorical_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value="missing")),
                                              ('onehotencoder', OneHotEncoder(categories='auto', sparse=False, handle_unknown='ignore'))])
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('numeric', numeric_transformer, numeric_features),
            ('categorical', categorical_transformer, categorical_features)
        ], remainder="drop")
    
    return preprocessor

def model_train(LABEL, df, run, penalty_term, C_value):  
    y_raw = df[LABEL]
    X_raw = df.drop([LABEL], axis=1)
    
     # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=0.3, random_state=0)
    
    lg = LogisticRegression(penalty=penalty_term, C=C_value, solver='liblinear')
    preprocessor = buildpreprocessorpipeline(X_train)
    
    #estimator instance
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                               ('regressor', lg)])

    model = clf.fit(X_train, y_train)
    
    
    # calculate AUC
    y_scores = model.predict_proba(X_test)
    auc = roc_auc_score(y_test,y_scores[:,1])
    print('AUC: ' + str(auc))
    run.log('AUC', np.float(auc))

    
    # calculate test accuracy
    y_hat = model.predict(X_test)
    acc = np.average(y_hat == y_test)
    print('Accuracy:', acc)
    run.log('Accuracy', np.float(acc))

    # plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])
    fig = plt.figure(figsize=(6, 4))
    # Plot the diagonal 50% line
    plt.plot([0, 1], [0, 1], 'k--')
    # Plot the FPR and TPR achieved by our model
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    run.log_image(name = "ROC", plot = fig)
    plt.show()

    # plot confusion matrix
    # Generate confusion matrix
    cmatrix = confusion_matrix(y_test, y_hat)
    cmatrix_json = {
        "schema_type": "confusion_matrix",
           "schema_version": "v1",
           "data": {
               "class_labels": ["0", "1"],
               "matrix": [
                   [int(x) for x in cmatrix[0]],
                   [int(x) for x in cmatrix[1]]
               ]
           }
    }
    
    run.log_confusion_matrix('ConfusionMatrix_Test', cmatrix_json)

    return model, auc, acc
    # Save the trained model
    
    
def main():
    # Create an Azure ML experiment in your workspace
    args = getRuntimeArgs()
    
    run = Run.get_context()
    run.log('penalty_term', args.penalty_term)
    run.log('C', np.float(args.C))
    dataset_dir = './dataset/'
    os.makedirs(dataset_dir, exist_ok=True)
    ws = run.experiment.workspace
    print(ws)
    

    print("Loading Data...")
    #dataset = Dataset.get_by_id(ws, id=args.input_data)
    dataset = run.input_datasets['titanic']
    # Load a TabularDataset & save into pandas DataFrame
    df = dataset.to_pandas_dataframe()
    
    print(df.head(5))
 
    model, auc, acc = model_train('Survived', df, run, args.penalty_term, args.C)
    
    os.makedirs('outputs', exist_ok=True)
    
    
    model_file = os.path.join('outputs', 'titanic_model.pkl')
    joblib.dump(value=model, filename=model_file)
    
    # Register the model
    print('Registering model...')
    Model.register(workspace=run.experiment.workspace,
               model_path = model_file,
               model_name = 'titanic_model',
               tags={'Training context':'Compute'},
               properties={'AUC': np.float(auc), 'Accuracy': np.float(acc)})

    run.complete()

if __name__ == "__main__":
    main()
