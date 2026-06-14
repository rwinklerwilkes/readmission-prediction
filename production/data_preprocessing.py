import pandas as pd
from . import utils as u


def load_file():
    data_directory = u.get_directory("data")
    pd.set_option("display.max_columns", None)
    df = pd.read_csv(f"{data_directory}/diabetic_data.csv")
    return df
