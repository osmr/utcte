from torch.utils.data import Dataset


class MnistDataset(Dataset):

    def __init__(self,
                 img,
                 lbl):
        super(MnistDataset, self).__init__()
        self.img = img
        self.lbl = lbl

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        return self.img[index], self.lbl[index]

    def __len__(self):
        return len(self.img)