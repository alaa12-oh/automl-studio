
import pandas as pd
import os


def load_data(file):

    if isinstance(file, (str, os.PathLike)):

        if not os.path.exists(file):
            raise FileNotFoundError("File does not exist")

    try:
        data = pd.read_csv(file)

    except pd.errors.EmptyDataError:
        raise ValueError("The CSV file is empty")

    except pd.errors.ParserError:
        raise ValueError("The CSV file has an invalid format")

    if data.empty:
        raise ValueError("The CSV file contains no data")

    return data

