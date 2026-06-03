import production.utils as u
import pandera.pandas as pa
import pandas as pd
import numpy as np

def validation_schema():
    schema = pa.DataFrameSchema({
        "encounter_id": pa.Column(int, unique=True),
        "patient_nbr": pa.Column(int, nullable=False),
        "race": pa.Column(str),
        "gender": pa.Column(str, pa.Check(lambda s: s.isin(["Female", "Male", "Unknown/Invalid"]))),
        "age": pa.Column(str),
        "weight": pa.Column(str),
        'admission_type_id': pa.Column(int, [pa.Check.greater_than(0), pa.Check.le(8)]), #Defined in data/IDS_mapping.csv
        'discharge_disposition_id': pa.Column(int, [pa.Check.greater_than(0), pa.Check.le(29)]), #Defined in data/IDS_mapping.csv
        'admission_source_id': pa.Column(int, [pa.Check.greater_than(0), pa.Check.le(26)]), #Defined in data/IDS_mapping.csv
        'time_in_hospital': pa.Column(int, pa.Check.greater_than(0))
    }
    )
    return schema

def validate_against_schema(df):
    schema = validation_schema()
    try:
        schema.validate(df)
    except pa.errors.SchemaErrors as exc:
        # Strict column checks are reported as SchemaErrors (one or more entries)
        for err in exc.schema_errors:
            print(err)

def ratio_by_group(df, grouping):
    agg_df = df.groupby(grouping).count()['encounter_id'].reset_index()
    agg_df.columns = [grouping,'count']
    agg_df['ratio'] = agg_df['count']/df.count()['encounter_id']
    return agg_df

def population_stability_index(old_df, new_df, grouping, pregrouped=False):
    if not pregrouped:
        old_ratio = ratio_by_group(old_df, grouping)
        new_ratio = ratio_by_group(new_df, grouping)
    else:
        old_ratio = old_df
        new_ratio = new_df
    compare_ratio = pd.merge(old_ratio, new_ratio, on=grouping, how='outer', suffixes=('_old','_new'))
    compare_ratio = compare_ratio.fillna(0.0001)
    compare_ratio['diff'] = compare_ratio['ratio_old'] - compare_ratio['ratio_new']
    compare_ratio['log_diff'] = np.log(compare_ratio['ratio_old']/compare_ratio['ratio_new'])
    compare_ratio['psi'] = compare_ratio['diff'] * compare_ratio['log_diff']
    return compare_ratio, np.sum(compare_ratio['psi'])

PSI_COLUMNS = ['gender','race','age','admission_type_id','discharge_disposition_id','admission_source_id']

def calculate_psi_statistics(df, initial=False):
    data_validation_directory = u.get_directory('data_validation')
    stats = {}
    for col in PSI_COLUMNS:
        col_ratio = ratio_by_group(df, col)
        if initial:
            col_ratio.to_csv(f'{data_validation_directory}/validation_{col}.csv')
        else:
            stats[col] = col_ratio
    return stats

def load_initial_statistics():
    data_validation_directory = u.get_directory('data_validation')
    stats = {}
    for col in PSI_COLUMNS:
        stats[col] = pd.read_csv(f'{data_validation_directory}/validation_{col}.csv')
    return stats