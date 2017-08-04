
class DataAugmentation(object):
    """ Data Augmentation.

    Base class for applying common real-time data augmentation.

    This class is meant to be used as an argument of `input_data`. When training
    a model, the defined augmentation methods will be applied at training
    time only. Note that DataPreprocessing is similar to DataAugmentation,
    but applies at both training time and testing time.

    Arguments:
        None

    Parameters:
        methods: `list of function`. The augmentation methods to apply.
        args: A `list` of arguments list to use for these methods.

    """

    def __init__(self):
        self.methods = []
        self.args = []

    def apply(self, batch):
        for i, m in enumerate(self.methods):
            if self.args[i]:
                batch = m(batch, *self.args[i])
            else:
                batch = m(batch)
        return batch
