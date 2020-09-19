from load_data import Data

class kddNShot:
    def __init__(self, path, batch_size):
        self.data_generator = Data(path, batch_size)
        self.norm_adj = self.data_generator.get_adj_mat()
    
    def next(self):
        return self.data_generator.sample()

    def get_test(self):
        return self.data_generator.sample('test')


    


    
    