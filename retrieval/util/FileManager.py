import csv
import os
from typing import Union, Any
import pickle


def read_tsv(file_name: str) -> Union[list, str]:
    file = open(file_name)
    if file_name.endswith('.tsv'):
        return list(csv.reader(file, delimiter="\t"))
    raise RuntimeError("Invalid file type: only '.tsv' files are accepted.")


def write_txt(file_name: str, data: str, mode='w'):
    directory = os.path.dirname(file_name)
    if directory and os.path.exists(directory):
        os.makedirs(directory)
    file = open(file_name, mode=mode)
    file.write(data)
    file.close()


def write_pickle(data: object, file_name: str):
    pickle.dump(data, open(file_name, "wb"))


def read_pickle(file_name: str) -> Any:
    return pickle.load(open(file_name, "rb"))
