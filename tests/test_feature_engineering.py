import pandas as pd
import numpy as np
import pytest
from production import feature_engineering as fe


@pytest.fixture
def my_dataframe():
    input_data = [[1,'abc','NO'],
                  [2,'def','NO'],
                  [3,'abc','NO'],
                  [4,'def','NO'],
                  [5,'abc','<30'],
                  [6,'abc','>30']]
    df = pd.DataFrame(input_data,columns=['encounter_id','class','readmitted'])
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