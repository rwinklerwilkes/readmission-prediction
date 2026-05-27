from sklearn import model_selection as model_sel
from sklearn import metrics
import utils as u
import xgboost as xgb
import feature_engineering as f
import os
import joblib
import json

def split_X_and_y(feature_df):
    Xf, yf = f.get_features_to_use()
    X = feature_df[Xf]
    X = convert_to_categories(X)
    
    y = feature_df[yf]
    return X,y

def train_test_split(X,y):
    X_train, X_test, y_train, y_test = model_sel.train_test_split(X, y, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test

def convert_to_categories(X):
    X = X.copy()
    category_types = ['race','gender','age','diag_1','diag_2','diag_3']#,'representative_term_first']
    for c in category_types:
        X[c] = X[c].astype('category')
    return X

def minify(X_train, X_test):
    mini_features = ['diag_1','diag_2','diag_3','number_inpatient','admission_source_id','age','num_medications']
    
    X_train_mini = X_train.loc[:,mini_features]
    X_test_mini = X_test.loc[:,mini_features]
    return X_train_mini, X_test_mini

def train_model(X_train, y_train):
    params = {
        'n_estimators':1000,
        "objective": "binary:logistic",
        "eval_metric": "auc",
    }
    model = xgb.XGBClassifier(**params,
                            enable_categorical=True,
                            verbosity=1,
                            device="cuda")
    model.fit(X_train, y_train)
    return model

def calculate_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)

    output_metrics = {}
    output_metrics['confusion_matrix'] = metrics.confusion_matrix(y_test,y_pred)
    output_metrics['precision'] = metrics.precision_score(y_test, y_pred)
    output_metrics['recall'] = metrics.recall_score(y_test, y_pred)
    return output_metrics

def load_champion_metrics(model_champion_filename):
    if os.path.isfile(model_champion_filename):
        with open(model_champion_filename, 'r') as f:
            try:
                json_dict = json.load(f)
                read = True
            except json.JSONDecodeError as je:
                read = False
    else:
        read = False
    if not read:
        json_dict = {}
    return read, json_dict

def compare_current_to_champion(model, output_metrics, comparison_metric_type):
    model_directory = u.get_directory('models')
    model_champion_filename = f'{model_directory}/champion.json'
    output_json = {}
    read, json_dict = load_champion_metrics(model_champion_filename)
    
    if read:
        model_hash_champion = json_dict['model_hash']
        comparison_metric_champion = json_dict[comparison_metric_type]
        if float(comparison_metric_champion) - float(output_metrics[comparison_metric_type]) > -0.0001:
            print('Champion outperforming new version, retaining champion')
            champion_model = joblib.load(f'{model_directory}/{model_hash_champion}.joblib')
            output_json['model_hash'] = model_hash_champion
            output_json['recall'] = json_dict['recall']
            output_json['precision'] = json_dict['precision']
            output_json['confusion_matrix'] = json_dict['confusion_matrix']
        else:
            print('New version outperforming champion, saving new version')
            champion_model = model
            output_json['model_hash'] = joblib.hash(model)
            output_json['recall'] = str(output_metrics['recall'])
            output_json['precision'] = str(output_metrics['precision'])
            output_json['confusion_matrix'] = str(output_metrics['confusion_matrix'].tolist()) #tolist() ensures serializability
    else:
        print('No valid champion version detected, using new model')
        champion_model = model
        output_json['model_hash'] = joblib.hash(model)
        output_json['recall'] = str(output_metrics['recall'])
        output_json['precision'] = str(output_metrics['precision'])
        output_json['confusion_matrix'] = str(output_metrics['confusion_matrix'].tolist()) #tolist() ensures serializability

    model_hash = output_json['model_hash']
    print(output_json)
    joblib.dump(champion_model, f'{model_directory}/{model_hash}.joblib')
    with open(model_champion_filename, 'w') as f:
        json.dump(output_json, f)
    return champion_model