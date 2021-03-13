import csv
from typing import Union, IO


def read(file_name: str) -> Union[list, str]:
    file = open(file_name)
    if file_name.endswith('.tsv'):
        return _read_tsv(file)
    elif file_name.endswith('.txt'):
        return _read_txt(file)
    else:
        raise RuntimeError("Invalid file type: only '.tsv' and '.txt' files are accepted.")


def _read_tsv(file: IO) -> list:
    return list(csv.reader(file, delimiter="\t"))


def _read_txt(file: IO) -> str:
    return file.read()
