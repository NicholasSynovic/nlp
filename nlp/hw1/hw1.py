from pathlib import PurePath
from utils import loadDataTXT

def main()  ->  None:
    positiveDataFile: PurePath = PurePath("positiveSentiment.txt")
    negativeDataFile: PurePath = PurePath("negativeSentiment.txt")

    positiveDataFile: list = loadDataTXT(positiveDataFile)
    negativeDataFile: list = loadDataTXT(negativeDataFile)



if __name__ == "__main__":
    main()
