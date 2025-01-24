from torch.utils.data import Dataset
# we do not used this class
class CustomDataset(Dataset):
    def __init__(self, fo_input, stgraph, target):

        self.fo_input = fo_input
        self.stgraph = stgraph
        self.target = target
        self.n_samples = len(fo_input)
        
    def __getitem__(self, index):
        return self.fo_input[index], self.stgraph[index], self.target[index]
    
    def __len__(self):
        return self.n_samples
