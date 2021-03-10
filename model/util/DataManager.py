import csv


# Reads a .tsv or .txt file, turns each row into a list, and returns every row in a single list.
def read(file_name):
    file = open(file_name)
    if file_name.endswith('.tsv'):
        rows = list(csv.reader(file, delimiter="\t"))
    elif file_name.endswith('.txt'):
        rows = file.readlines()
    else:
        raise RuntimeError("Invalid file type: only '.tsv' and '.txt' files are accepted.")
    file.close()
    return rows


def lists_to_dicts(lists: list, keys: list) -> list:
    new_list = []
    for list_ in lists:
        list_to_dict(list_, keys)
    return new_list


def list_to_dict(list_: list, keys: list) -> dict:
    """
    Converts a list into a dictionary by assigning each value to the corresponding key.

    Args:
        list_: The list containing the dictionary values.
        keys: The list containing the dictionary keys.

    Returns:
        A dictionary with each element in the list assigned to the corresponding key.

    Raises:
        RuntimeError: Raises an exception if the number of keys is not equal to the number
        of elements within the list.
    """
    if len(list_) != len(keys):
        raise RuntimeError("Number of keys is not equal to number of elements.")
    dict_ = dict(zip(keys, list_))
    return dict_
