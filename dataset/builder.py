import torch
from .leave_pair_out import LeavePairOutTrainDataset, LeavePairOutTestDataset
from .walk_path_leave_pair_out import WalkPathLeavePairOutTrainDataset, WalkPathLeavePairOutTestDataset
from utils.seed import init_seed

def build_dataset(arg, split='train'):
    """
    Dataset + DataLoader を返す統一関数

    Args:
        arg: argparse or configで読み込んだ設定
        split: 'train' または 'test'

    Returns:
        torch.utils.data.DataLoader
    """

    if arg.evaluation_method == 'leave_pair_out':
        print("this evaluation method is leave_pair_out method")
        DatasetClass = LeavePairOutTrainDataset if split == 'train' else LeavePairOutTestDataset
        dataset_kwargs = {
            'data_path': arg.train_data_path if split=='train' else arg.test_data_path,
            'label_path': arg.train_label_path if split=='train' else arg.test_label_path,
        }
        if split == 'train':
            dataset_kwargs['leave_pair'] = arg.leave_pair
            dataset_kwargs['data_augment'] = arg.data_augment
        else:
            dataset_kwargs['leave_pair'] = arg.leave_pair


    elif arg.evaluation_method == 'walk_path_leave_pair_out' or arg.evaluation_method == 'val2':
        print("this evaluation method is walk_path_leave_pair_out method")
        DatasetClass = WalkPathLeavePairOutTrainDataset if split=='train' else WalkPathLeavePairOutTestDataset
        dataset_kwargs = {
            'data_path': arg.train_data_path if split=='train' else arg.test_data_path,
            'label_path': arg.train_label_path if split=='train' else arg.test_label_path,
            'leave_pair': arg.leave_pair,
        }
        if split == 'train':
            dataset_kwargs['data_augment'] = arg.data_augment
            dataset_kwargs['train_walkpath'] = arg.train_walkpath
        else:
            dataset_kwargs['test_walkpath'] = arg.test_walkpath

    else:
        raise ValueError(f"Unknown evaluation_method: {arg.evaluation_method}")

    dataset = DatasetClass(**dataset_kwargs)

    dataloader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=arg.batch_size,
        shuffle=(split=='train'),
        num_workers=arg.num_worker,
        drop_last=(split=='train'),
        worker_init_fn=init_seed
    )

    return dataloader