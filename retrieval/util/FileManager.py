import csv
from typing import Union
import pickle


def read_tsv(file_name: str) -> Union[list, str]:
    file = open(file_name)
    if file_name.endswith('.tsv'):
        return list(csv.reader(file, delimiter="\t"))
    raise RuntimeError("Invalid file type: only '.tsv' files are accepted.")


def write_pickle(data: object, file_name: str):
    pickle.dump(data, open(file_name, "wb"))


def read_pickle(file_name: str):
    return pickle.load(open(file_name, "rb"))
