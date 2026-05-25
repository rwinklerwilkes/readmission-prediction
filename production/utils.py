import os

def get_directory(directory_name):
    cwd = os.getcwd()
    one_level_up =os.path.dirname(cwd)
    directory = os.path.join(one_level_up,directory_name)
    return directory