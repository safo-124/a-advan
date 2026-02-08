
from dataset_class import MyDataset
from torch.utils.data import DataLoader
from pathlib import Path

def load_data(data_path, split, batch_size, shuffle, drop_last, num_workers):
    """Load data for a specific split.
    
    :param data_path: Path to the dataset root directory.
    :param split: The split to load ('training', 'validation', 'testing').
    :param batch_size: Batch size for the data loader.
    :param shuffle: Whether to shuffle the data.
    :param drop_last: Whether to drop the last incomplete batch.
    :param num_workers: Number of workers for data loading.
    :return: DataLoader object.
    """
    # Map split names to directory names
    split_dir_map = {
        'training': 'train',
        'validation': 'val',
        'testing': 'test'
    }
    
    # Get the appropriate directory name
    dir_name = split_dir_map.get(split, split)
    
    # Create an object of class MyDataset and give it the data folder full path to read data from.
    dataset = MyDataset(data_dir=dir_name, data_parent_dir=str(data_path))
    
    # Create a DataLoader object and pass it as input the "dataset" object and other needed input parameters
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )
    
    return data_loader


def main():
    
    batch_size = 8
    data_path = Path(__file__).parent.parent / 'genres_dataset'
    
    # loading the training data
    print('Loading the training data')
    split = "training"     
    train_loader = load_data(data_path, split, batch_size, shuffle=True, drop_last=True, num_workers=1)
    train_files = train_loader.dataset.files
    print('The number of total training files are : ')
    print(len(train_files))
    
    # load the validation data
    print('Loading the validation data')
    split = "validation"
    validation_loader = load_data(data_path, split, batch_size, shuffle=True, drop_last=True, num_workers=1)
    validation_files = validation_loader.dataset.files
    print('The number of total validation files are : ')
    print(len(validation_files))
    
    # load the testing data
    print('Loading the testing data')
    split = "testing"
    # Note: testing data should NOT be shuffled to keep track of predictions
    test_loader = load_data(data_path, split, batch_size, shuffle=False, drop_last=False, num_workers=1)
    test_files = test_loader.dataset.files
    print('The number of total testing files are : ')
    print(len(test_files))


if __name__ == '__main__':
    main()

# EOF