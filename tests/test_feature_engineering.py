import pandas as pd
import numpy as np
import pytest
from production import feature_engineering as fe


@pytest.fixture
def my_dataframe():
    input_data = [[1,'abc','NO','?',11],
                  [2,'def','NO','?',19],
                  [3,'abc','NO','[0-10)',8],
                  [4,'def','NO','[10-20)',26],
                  [5,'abc','<30','[50-100)',15],
                  [6,'abc','>30','[150-200)',1]]
    df = pd.DataFrame(input_data,columns=['encounter_id','class','readmitted',
                                          'weight','discharge_disposition_id'])
    return df

@pytest.fixture
def my_diag_dataframe():
    input_data = [[1,'mno','abc','pqr'],
                  [2,'abc','def','pqr'],
                  [3,'pqr','abc','def'],
                  [4,'abc','def','stu'],
                  [5,'stu','abc','def'],
                  [6,'stu','abc','def']]
    df = pd.DataFrame(input_data,columns=['encounter_id','diag_1','diag_2','diag_3'])
    return df

@pytest.fixture
def my_readmitted_dataframe():
    input_data = [[1,'mno',0,'a'],
                  [2,'abc',0,'a'],
                  [3,'pqr',0,'b'],
                  [4,'abc',1,'a'],
                  [5,'stu',0,'b'],
                  [6,'stu',1,'b'],
                  [7,'pqr',1,'b']
                 ]
    df = pd.DataFrame(input_data,columns=['encounter_id','patient_nbr','readmitted_dummy','fake_grouping'])
    return df


def test_create_dummies(my_dataframe):
    df = fe.create_dummies(my_dataframe)
    assert df['readmitted_dummy'].sum() == 2

def test_count_all_diag(my_diag_dataframe):
    all_diag = fe.count_all_diag(my_diag_dataframe)
    assert all_diag.shape[0] == 5
    expected_values = {'abc':{'diag_1_ct':2,
                              'diag_2_ct':4,
                              'diag_3_ct':0,
                              'total_ct':6,},
                       'def':{'diag_1_ct':0,
                              'diag_2_ct':2,
                              'diag_3_ct':3,
                              'total_ct':5,},
                       'mno':{'diag_1_ct':1,
                              'diag_2_ct':0,
                              'diag_3_ct':0,
                              'total_ct':1,},
                       'stu':{'diag_1_ct':2,
                              'diag_2_ct':0,
                              'diag_3_ct':1,
                              'total_ct':3,},
                       'pqr':{'diag_1_ct':1,
                              'diag_2_ct':0,
                              'diag_3_ct':2,
                              'total_ct':3,}
                      }
    for diag, checks in expected_values.items():
        for column, expected_val in checks.items():
            assert all_diag.loc[all_diag['diag']==diag,column].values[0]==expected_val


def test_parse_weight():
    assert np.isnan(fe.parse_weight('?'))
    assert fe.parse_weight('>200') == 200
    assert fe.parse_weight('(0-100]') == 50
    assert fe.parse_weight('(0-10]') == 5

def test_parse_weight_to_number(my_dataframe):
    df_parsed = fe.parse_weight_to_number(my_dataframe)
    count_by_flag = df_parsed.groupby('has_weight').count()['encounter_id'].reset_index()
    assert count_by_flag.loc[count_by_flag['has_weight']=='Y','encounter_id'].values[0] == 4
    assert count_by_flag.loc[count_by_flag['has_weight']=='N','encounter_id'].values[0] == 2
    expected_vals = [np.nan, np.nan, 5, 15, 75, 175]
    expected_masked = np.ma.masked_where(np.isnan(expected_vals), expected_vals)
    assert np.all(expected_masked == df_parsed.loc[:,'weight_parsed'].values)

def test_calculate_readmitted_rate(my_readmitted_dataframe):
    rate = fe.calculate_readmitted_rate(my_readmitted_dataframe)
    assert np.abs(rate - 3/7.0) < 0.01
    
    rate_in_group = fe.calculate_readmitted_rate(my_readmitted_dataframe,grouping = 'fake_grouping')
    a_rate = rate_in_group.loc[rate_in_group['fake_grouping']=='a','readmitted_rate'].values[0]
    b_rate = rate_in_group.loc[rate_in_group['fake_grouping']=='b','readmitted_rate'].values[0]
    assert np.abs(a_rate - 1/3.0) < 0.01
    assert np.abs(b_rate - 2/4.0) < 0.01

def test_add_prior_admissions(my_readmitted_dataframe):
    df = fe.add_prior_admissions(my_readmitted_dataframe)
    should_be_one = df.loc[df['encounter_id'].isin([4,6,7]),'prior_readmissions']
    should_be_zero = df.loc[~df['encounter_id'].isin([4,6,7]),'prior_readmissions']
    assert np.max(should_be_one) == 1
    assert np.max(should_be_one) == np.min(should_be_one)
    assert np.max(should_be_zero) == 0
    assert np.max(should_be_zero) == np.min(should_be_zero)

def test_filter_for_still_alive(my_dataframe):
    alive, expired = fe.filter_for_still_alive(my_dataframe)
    assert alive.shape[0] == 4
    assert expired.shape[0] == 2


