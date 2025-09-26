from pandas import read_csv
class DataLoader:

    def load(self, file_path):
        return read_csv(file_path)
    
