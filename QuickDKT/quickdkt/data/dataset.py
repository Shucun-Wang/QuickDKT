import torch

class KTDataset(torch.utils.data.Dataset):
    def __init__(self, data, n_q):
        super(KTDataset, self).__init__()
        self.data = data
        self.n_q = n_q

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        q, qa = self.data[idx]
        target = (qa - 1) // self.n_q

        return (
            torch.LongTensor(q),
            torch.LongTensor(qa),
            torch.Tensor(target)
        )
    
class AKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, n_q):
        super(AKTDataset, self).__init__()
        self.data = data
        self.n_q = n_q

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        q, qa, p = self.data[idx]
        target = (qa - 1) // self.n_q

        return (
            torch.LongTensor(q),
            torch.LongTensor(qa),
            torch.LongTensor(p),
            torch.Tensor(target)
        )