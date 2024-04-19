from random import randint
from torch.utils.data import Dataset, DataLoader
from torch import zeros


class Dataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def one_hot_encoding(data, values):
    length = len(values)

    out = []

    for datapoint in data:
        tensor = zeros(size=(length, ))

        idx = values.index(datapoint)
        tensor[idx] = 1.
        out.append(tensor)

    return out

def dict_lists_to_list_of_dicts(input_dict:dict):
    """
    A function that convert a dictionary of lists into a list of dictionaries
    Parameters
    ----------
    input_dict:dict
        The dictionary to convert
    
    Outputs
    -------
    The list of dictionaries
    """
    keys = input_dict.keys()
    list_lengths = [len(input_dict[key]) for key in keys]

    if len(set(list_lengths)) > 1:
        raise ValueError("All lists in the input dictionary must have the same length.")

    list_of_dicts = [{key: input_dict[key][i] for key in keys} for i in range(list_lengths[0])]

    return list_of_dicts

def get_dataloader(x, y, batch_size, test_buckets = []):
    BUCKETS = 10

    N_ELEMENTS = len(x)

    BUCKET_SIZE = N_ELEMENTS // BUCKETS

    TEST_BUCKETS = 1


    x_local = x.copy()
    y_local = y.copy()
    x_test, y_test = [], []
    
    if len(test_buckets) == 0: 
        for _ in range(TEST_BUCKETS):
            idx = randint(0, BUCKETS)
            while idx in test_buckets:
                idx = randint(0, BUCKETS)
            test_buckets.append(idx)

    for bucket in test_buckets:
        idx = bucket * BUCKET_SIZE
        for _ in range(BUCKET_SIZE):
            x_test.append(x_local.pop(idx))
            y_test.append(y_local.pop(idx))

    train_elements = (len(y_local) // 10) * 9
    x_train = x_local[:train_elements]
    y_train = y_local[:train_elements]

    x_validation = x_local[train_elements:]
    y_validation = y_local[train_elements:]

    train_dataset, val_dataset, test_dataset = Dataset(x_train, y_train), Dataset(x_validation, y_validation), Dataset(x_test, y_test)
    
    return (DataLoader(train_dataset, batch_size=batch_size, shuffle=True), 
            DataLoader(val_dataset, batch_size=batch_size), 
            DataLoader(test_dataset, batch_size=batch_size)) 

