from pathlib import PurePath

def loadDataTXT(filepath: PurePath)    ->  list:
    with open(filepath, "r") as dataFile:
        data: list = dataFile.readlines()
        dataFile.close()

    data = [d.strip() for d in data]

    return data
