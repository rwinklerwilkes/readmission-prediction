from sklearn.model_selection import train_test_split
import feature_engineering as f

def split_X_and_y(feature_df):
    Xf, yf = f.get_features_to_use()
    X = feature_df[Xf]
    y = feature_df[yf]
    return X,y

## TODO: Convert rest of file to functional style
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

category_types = ['race','gender','age','diag_1','diag_2','diag_3']#,'representative_term_first']
for c in category_types:
    X_train[c] = X_train[c].astype('category')
    X_test[c] = X_test[c].astype('category')

mini_features = ['diag_1','diag_2','diag_3','number_inpatient','admission_source_id','age','num_medications']

X_train_mini = X_train.loc[:,mini_features]
X_test_mini = X_test.loc[:,mini_features]

import xgboost as xgb

params = {
    'n_estimators':1000,
    "objective": "binary:logistic",
    "eval_metric": "auc",
}
#     "eta": 0.01,
    
#     "subsample": 0.5,
#     "base_score": np.mean(y_train),
    
# }

# d_train = xgb.DMatrix(X_train_mini, label=y_train,enable_categorical=True,)
# d_test = xgb.DMatrix(X_test_mini, label=y_test,enable_categorical=True,)

X_train_to_use = X_train
X_test_to_use = X_test

clf = xgb.XGBClassifier(**params,
                        enable_categorical=True,
                        verbosity=1,
                        device="cuda")
clf.fit(X_train_to_use, y_train)

y_pred = clf.predict(X_test_to_use)

from sklearn import metrics as m

print(m.confusion_matrix(y_test,y_pred))
print(m.precision_score(y_test, y_pred))
print(m.recall_score(y_test, y_pred))