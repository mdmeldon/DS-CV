from pathlib import Path

DIR_DATA = Path(__file__).absolute().parent.parent / 'data'
DIR_DATA_LOGS = DIR_DATA / 'logs'
DIR_DATA_LOGS.mkdir(parents=True, exist_ok=True)

DIR_DATA_RAW = DIR_DATA / 'raw'
DIR_DATA_RAW.mkdir(parents=True, exist_ok=True)

DIR_DATA_INTERHIM = DIR_DATA / 'interhim'
DIR_DATA_INTERHIM.mkdir(parents=True, exist_ok=True)

DIR_DATA_PROCESSED = DIR_DATA / 'processed'
DIR_DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

DIR_DATA_MODELS = DIR_DATA / 'models'
DIR_DATA_MODELS.mkdir(parents=True, exist_ok=True)

DIR_DATA_NUMPY = DIR_DATA / 'numpy'
DIR_DATA_NUMPY.mkdir(parents=True, exist_ok=True)

def iterfiles(directory: Path, include: callable = lambda x: True):
    """An iterator that returns a list of the names of all files in the specified directory and its subdirectories.

    Args:
        directory: pathlib.Path directory object.
        include: (optional) Lambda function for filtering files (takes a file name, returns True to take a file, False otherwise).

    Returns:
        Generator listing file names (pathlib.Path objects).
    """
    for x in directory.iterdir():
        if x.is_file() and include(x):
            yield x
        else:
            yield from iterfiles(directory / x, include)
