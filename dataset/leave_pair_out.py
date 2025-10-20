from dataset.base_dataset import BaseDataset
from dataset import tools
import numpy as np
import torch


class LeavePairOutTrainDataset(BaseDataset):
    def __init__(self, data_path, label_path, leave_pair, data_augment=None):
        super().__init__()
        self.data_augment = data_augment or {}
        self.data, self.label = self._load_leave_pair_data(data_path, label_path, leave_pair)
        
        
    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]

        # データ拡張をループで適用
        for aug_name, aug_flag in self.data_augment.items():
            if aug_flag:
                data = getattr(tools, aug_name)(data)

        return data, label, index

    def _load_leave_pair_data(self, data_path, label_path, leave_pair):
        tmp_data = np.load(data_path).astype(np.float32)
        tmp_label = np.load(label_path)
        PAIR_NUM = 52
        
        N, C, T, V, M = tmp_data.shape
        one_pair_data_num = N // PAIR_NUM
        all_pair = list(range(1, PAIR_NUM + 1))
        not_leave_pair = [p for p in all_pair if p not in leave_pair]

        data = np.concatenate([
            tmp_data[(p-1)*one_pair_data_num:p*one_pair_data_num]
            for p in not_leave_pair
        ], axis=0)

        label = np.concatenate([
            tmp_label[(p-1)*one_pair_data_num:p*one_pair_data_num]
            for p in not_leave_pair
        ], axis=0)

        return data, label
    


class LeavePairOutTestDataset(BaseDataset):

    """leave-pair-outのテスト用Dataset"""
    def __init__(self, data_path, label_path, leave_pair):
        super().__init__()

        tmp = np.load(data_path).astype(np.float32)
        PAIR_NUM = 52
        WALK_PATH_NUM = 4

        N, C, T, V, M = tmp.shape
        slash = N // WALK_PATH_NUM  # 各walk_pathのサンプル数（不要なら削除可）

        one_pair_data_num = N // PAIR_NUM
        leave_pair_num = len(leave_pair)

        # データ初期化
        self.data = np.empty((one_pair_data_num * leave_pair_num, C, T, V, M), dtype=tmp.dtype)
        for i, index in enumerate(leave_pair):
            start_idx = (index-1) * one_pair_data_num
            end_idx = index * one_pair_data_num
            self.data[i*one_pair_data_num:(i+1)*one_pair_data_num] = tmp[start_idx:end_idx]

        # ラベルも同様
        tmp_label = np.load(label_path)
        self.label = np.empty([one_pair_data_num * leave_pair_num], dtype=tmp_label.dtype)
        for i, index in enumerate(leave_pair):
            start_idx = (index-1) * one_pair_data_num
            end_idx = index * one_pair_data_num
            self.label[i*one_pair_data_num:(i+1)*one_pair_data_num] = tmp_label[start_idx:end_idx]

    def __getitem__(self, index):
        """DataLoaderで使われるgetitem"""
        data = self.data[index]
        label = self.label[index]
        return data, label, index
    


if __name__ == '__main__':
    data_augument = {
        "random_move" : True
    }
    a = torch.utils.data.DataLoader(dataset=LeavePairOutTrainDataset(data_path=f'data/MB_3DP/2/data.npy', label_path=f'data/MB_3DP/2/label.npy', leave_pair=[1], data_augment=data_augument), batch_size=32, shuffle=True)
    for data in (a):
        print(data[0].size())
        
