from pathlib import PurePath
from numpy.random import RandomState
from numpy import ndarray

def loadDataTXT(filepath: PurePath, keepSplit: float = 0.6, seed: int = 42)    ->  ndarray:
    random: RandomState = RandomState(seed)

    with open(filepath, "r") as dataFile:
        data: list = dataFile.readlines()
        dataFile.close()

    data = [d.strip() for d in data]
    data = random.choice(data, size=round(len(data) * keepSplit), replace=False)

    return data
