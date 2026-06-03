import numpy as np
import pandas as pd
import production.utils as u


def load_icd9():
    data_directory = u.get_directory('data')
    icd9 = pd.read_csv(f"{data_directory}/icd9.txt",sep='\t',encoding='mbcs')
    icd9.columns = ['diagnosis','long','short','other']
    icd9.drop('other',inplace=True,axis=1)
    return icd9

def create_dummies(df):
    df_copy = df.copy()
    df_copy['readmitted_dummy'] = (df_copy['readmitted']!='NO')*1
    return df_copy

def count_all_diag(df):
    diag = pd.concat([df['diag_1'].drop_duplicates(),df['diag_2'].drop_duplicates(),df['diag_3'].drop_duplicates()]).drop_duplicates().reset_index()
    diag.columns = ['index','diag']
    diag = diag.drop('index',axis=1)

    for i in range(1,4):
        diag_df = df.groupby(f'diag_{i}').count()['encounter_id']
        diag_df = diag_df.reset_index()
        diag_df.columns = ['diag',f'diag_{i}_ct']
        diag = pd.merge(diag, diag_df,on='diag',how='left')
    
    diag = diag.fillna(0)
    diag['total_ct'] = diag['diag_1_ct'] + diag['diag_2_ct'] + diag['diag_3_ct']
    return diag

def parse_weight(x):
    if x == '?':
        return np.nan
    elif x == '>200':
        return 200
    else:
        weights = [int(i) for i in x[1:-1].split('-')]
        return np.mean(weights)

def parse_weight_to_number(df):
    #Only about 3% of the 100K rows have weight populated - it may be important for those patients but we probably shouldn't impute for remaining.
    #I also wonder about the validity of the data that we do have - [0,25) for a 60-70 year old woman is clearly wrong
    #Lastly, I'm going to convert these from ranges to the midpoints so they are sortable
    df_copy = df.copy()
    df_copy.loc[df_copy['weight']!='?','has_weight'] = 'Y'
    df_copy.loc[df_copy['weight']=='?','has_weight'] = 'N'
    df_copy['weight_parsed'] = df_copy['weight'].apply(parse_weight)
    # weight_readmission = calculate_readmitted_rate(has_weight,['gender','weight_parsed'])
    return df_copy
    

def join_diag_to_icd9(diag, icd9):
    diag_copy = diag.copy()
    #This method is very imperfect - just doing a rough check for now, deserves more time to refine
    diag_copy['diagnosis'] = diag_copy['diag'].str.replace(r'[^a-zA-Z0-9]','',regex=True)
    diag_copy = pd.merge(diag_copy,icd9,on='diagnosis',how='left')
    diag_copy = diag_copy.sort_values('total_ct',ascending=False)
    return diag_copy

def stash_data(df, diag):
    """Stashes several tables for use in the Streamlit frontend.
    I chose 100 diagnoses so that we would have the most popular/likely diagnoses available to test and wouldn't overwhelm the user with options.
    The other columns have little enough data available that we can save the whole table."""
    frontend_directory = u.get_directory('frontend')
    try:
        diag.iloc[:100].loc[:,['diag','total_ct']].to_csv(f'{frontend_directory}/valid_diag.csv',index=False)
        df['race'].drop_duplicates().to_csv(f'{frontend_directory}/valid_race.csv',index=False)
        df['age'].drop_duplicates().to_csv(f'{frontend_directory}/valid_age.csv',index=False)
        df['gender'].drop_duplicates().to_csv(f'{frontend_directory}/valid_gender.csv',index=False)
        return True
    except Exception as e:
        raise e

def calculate_readmitted_rate(df, grouping=None):
    if grouping:
        rate = df.groupby(grouping).agg({'patient_nbr':'count', 'readmitted_dummy':'sum'})
        rate['readmitted_rate'] = rate['readmitted_dummy']/rate['patient_nbr']
    else:
        rate = df['readmitted_dummy'].sum()/df['patient_nbr'].count()
    return rate

def add_prior_admissions(df):
    df_prior = df.copy()
    df_prior['prior_readmissions'] = df_prior.sort_values(['patient_nbr','encounter_id']).groupby('patient_nbr').cumcount()
    return df_prior

def filter_for_still_alive(df):
    """These are defined in IDS_Mapping.csv"""
    expired = df[df['discharge_disposition_id'].isin([11,19,20,21])]
    alive = df[~df['discharge_disposition_id'].isin([11,19,20,21])]
    return alive, expired

def convert_medicine_columns(x):
    if x=='No':
        return 0
    elif x in ['Down','Up','Steady']:
        return 1
    else:
        raise ValueError(x)

def compress_medicine_columns(df):
    df_medicine = df.copy()
    medicine_columns = ['metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
       'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
       'tolazamide', 'examide', 'citoglipton', 'insulin',
       'glyburide-metformin', 'glipizide-metformin',
       'glimepiride-pioglitazone', 'metformin-rosiglitazone',
       'metformin-pioglitazone']
    
    for m in medicine_columns:
        df_medicine[f'on_{m}'] = df_medicine[m].apply(convert_medicine_columns)
    on_medicine = ['on_' + m for m in medicine_columns]
    df_medicine['number_of_diabetes_medications'] = df_medicine[on_medicine].sum(axis=1)
    return df_medicine

def icd9_featurization(icd9):
    """Attempting to do some feature engineering on the ICD9 data to see if I can combine similar diagnoses together"""
    icd9['first_three'] = icd9['diagnosis'].str[0:3]
    icd9['long_tokenized'] = icd9['long'].str.replace('[^a-zA-Z0-9\s]','',regex=True).str.split(' ')
    
    representative_term_first = icd9.sort_values(['first_three','diagnosis']).groupby('first_three').first()['long'].reset_index()
    representative_term_first.columns = ['first_three','representative_term_first']
    icd9=pd.merge(icd9,representative_term_first,on='first_three')
    
    for i in icd9['first_three'].drop_duplicates():
        to_agg = icd9.loc[icd9['first_three']==i,['long_tokenized']]
        term_counts = to_agg.explode('long_tokenized').dropna().value_counts()
        if len(term_counts.index) > 1:
            max_term = term_counts.index[0][0] + ' ' + term_counts.index[1][0]
        else:
            max_term = term_counts.index[0][0]
        icd9.loc[icd9['first_three']==i,'representative_term']=max_term
        
    icd9_features = icd9[['first_three','representative_term_first']].drop_duplicates()
    return icd9_features

def add_icd9_features(df, icd9_features):
    #Right now the icd9 data isn't clean compared to what's in my current dataset - different formats, some periods, unclear how to tie these together
    #Best I can do is use the first three characters and hope that the diagnoses are correct
    df_icd9 = df.copy()
    df_icd9['first_three'] = df_icd9['diag_1'].str[0:3]
    df_icd9 = pd.merge(df_icd9, icd9_features, on=['first_three'],how='left')
    return df_icd9

def add_random_feature(df):
    df = df.copy()
    df['random'] = np.random.rand(df.shape[0],1)
    return df

def get_features_to_use():
    X_features_to_use = ['race', 'gender', 'age','admission_type_id',
                       'discharge_disposition_id', 'admission_source_id','time_in_hospital',
                       'num_lab_procedures', 'num_procedures', 'num_medications',
                       'number_outpatient', 'number_emergency', 'number_inpatient', 'diag_1',
                       'diag_2', 'diag_3', 'number_diagnoses','number_of_diabetes_medications',
                       'prior_readmissions'
                      ]
    y_feature_to_use = ['readmitted_dummy']
    return X_features_to_use, y_feature_to_use

def select_final_features(df):
    Xf, yf = get_features_to_use()
    feature_df = df[Xf+yf]
    return feature_df