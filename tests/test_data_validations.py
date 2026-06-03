import pandas as pd
import numpy as np
import pytest
from production import data_validations as dv


@pytest.fixture
def my_dataframe():
    input_data = [[1,'abc'],
                  [2,'def'],
                  [3,'abc'],
                  [4,'def'],
                  [5,'abc'],
                  [6,'abc']]
    df = pd.DataFrame(input_data,columns=['encounter_id','class'])
    return df

def get_class(x):
    if x < 0.5:
        return 'abc'
    else:
        return 'def'

def get_different_class(x):
    if x == 'abc':
        return 'ghi'
    if x == 'def':
        return 'jkl'

@pytest.fixture
def my_big_dataframe():
    rng = np.arange(0,1000).reshape(1,-1)
    rand_values = np.random.rand(1,1000)
    arr = np.vstack((rng, rand_values)).T
    df = pd.DataFrame(arr,columns=['encounter_id','rand_val'])
    df['class'] = df['rand_val'].apply(get_class)
    return df

@pytest.fixture
def my_class_different_dataframe(my_big_dataframe):
    df = my_big_dataframe.copy()
    df['class'] = df['class'].apply(get_different_class)
    return df
    
def test_ratio_by_group(my_dataframe):
    results = dv.ratio_by_group(my_dataframe, 'class')
    assert abs(results.loc[results['class']=='abc','ratio'].values[0] - 2/3) < 0.01
    assert abs(results.loc[results['class']=='def','ratio'].values[0] - 1/3) < 0.01

def test_ratio_by_group_big(my_big_dataframe):
    results = dv.ratio_by_group(my_big_dataframe, 'class')
    #Because these are randomly generated it's possible to get a weird ratio
    assert abs(results.loc[results['class']=='abc','ratio'].values[0] - 1/2) < 0.05
    assert abs(results.loc[results['class']=='def','ratio'].values[0] - 1/2) < 0.05

def test_psi_same_df(my_big_dataframe):
    compare_ratio, psi = dv.population_stability_index(my_big_dataframe, my_big_dataframe, 'class')
    assert abs(psi) < 0.01

def test_psi_different_df(my_big_dataframe, my_class_different_dataframe):
    compare_ratio, psi = dv.population_stability_index(my_big_dataframe, my_class_different_dataframe, 'class')
    assert abs(psi) > 1