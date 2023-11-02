from datasets import DatasetDict


def split_data(data, num_splits: int, test_size: float = 0.05, val_size: float = 0.05):
    """
        Splits the provided data into the specified number of splits for training, validation, and testing.

        Args:
            data (Hugging Face dataset): The data to be split.
            num_splits (int): The number of splits to be generated (2 and 3 supported).
            test_size (float, optional): The size of the test data split (default is 0.1).
            val_size (float, optional): The size of the validation data split (default is 0.5).

        Returns:
            dict: A dictionary containing the split data with keys 'train', 'validation', and 'test'.
                  Each key maps to a Hugging Face dataset split.
        """
    if num_splits == 3:
        # 90% train, 10% test + validation
        train_test_val = data.train_test_split(test_size=test_size + val_size)

        # Split the 10% test + valid in half test, half valid
        test_val = train_test_val['test'].train_test_split(test_size=test_size)

        data_split = DatasetDict({
            'train': train_test_val['train'],
            'test': test_val['test'],
            'validation': test_val['train']})

        return data_split

    elif num_splits == 2:
        train_val = data.train_test_split(test_size=val_size)
        data_split = DatasetDict({
            'train': train_val['train'],
            'validation': train_val['test']})

        return data_split
