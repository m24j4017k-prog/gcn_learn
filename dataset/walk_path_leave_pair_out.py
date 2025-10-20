import numpy as np
from .base_dataset import BaseDataset
from . import tools


class WalkPathLeavePairOutTrainDataset(BaseDataset):
    def __init__(self, data_path, label_path, train_walkpath, leave_pair=None, data_augment=None):
        """
        walk_paths: 使用したい歩行経路のリスト（例: [1, 3]）
        leave_pair: 除外したいペア番号のリスト（例: [2,5]）
        """
        super().__init__()
        print(data_augment)
        self.data_augment = data_augment or {}
        leave_pair = leave_pair or []

        tmp_data = np.load(data_path).astype(np.float32)
        tmp_label = np.load(label_path)

        PAIR_NUM = 52
        WALK_PATH_NUM = 4
        N, M, T, V, C = tmp_data.shape
        samples_per_path = N // WALK_PATH_NUM
        samples_per_pair = samples_per_path // PAIR_NUM

        data_list = []
        label_list = []

        for wp in train_walkpath:
            start = (wp - 1) * samples_per_path
            end = wp * samples_per_path
            wp_data = tmp_data[start:end]
            wp_label = tmp_label[start:end]

            keep_pair = [i for i in range(1, PAIR_NUM + 1) if i not in leave_pair]

            for p in keep_pair:
                p_start = (p - 1) * samples_per_pair
                p_end = p * samples_per_pair
                data_list.append(wp_data[p_start:p_end])
                label_list.append(wp_label[p_start:p_end])

        self.data = np.concatenate(data_list, axis=0)
        self.label = np.concatenate(label_list, axis=0)



    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]

        # データ拡張
        for aug_name, aug_flag in self.data_augment.items():
            if aug_flag:
                data = getattr(tools, aug_name)(data)

        return data, label, index
    
    
import numpy as np
import torch
from . import tools
from .base_dataset import BaseDataset  # 必要に応じて修正

class WalkPathLeavePairOutTestDataset(BaseDataset):
    def __init__(self, data_path, label_path, test_walkpath, leave_pair=None):
        """
        walk_paths: 使用したい歩行経路のリスト（例: [1, 3]）
        leave_pair: テストに使用したいペア番号のリスト（例: [2,5]）
        """
        super().__init__()
        leave_pair = leave_pair or []

        tmp_data = np.load(data_path).astype(np.float32)
        tmp_label = np.load(label_path)

        PAIR_NUM = 52
        WALK_PATH_NUM = 4
        N, M, T, V, C = tmp_data.shape
        samples_per_path = N // WALK_PATH_NUM
        samples_per_pair = samples_per_path // PAIR_NUM

        data_list = []
        label_list = []

        for wp in test_walkpath:
            start = (wp - 1) * samples_per_path
            end = wp * samples_per_path
            wp_data = tmp_data[start:end]
            wp_label = tmp_label[start:end]

            # テストでは leave_pair のみを使用する
            for p in leave_pair:
                p_start = (p - 1) * samples_per_pair
                p_end = p * samples_per_pair
                data_list.append(wp_data[p_start:p_end])
                label_list.append(wp_label[p_start:p_end])

        # 結合
        self.data = np.concatenate(data_list, axis=0)
        self.label = np.concatenate(label_list, axis=0)
        
        print(self.data.shape)


    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data, label, index

# class WalkPathLeavePairOutTrainDataset(BaseDataset):
#     def __init__(self, data_path, label_path, train_walkpath, leave_pair, data_augment=None):
#         super().__init__()
#         self.data_augment = data_augment or {}
#         self.data, self.label = self._load_data(data_path, label_path, train_walkpath, leave_pair)

#     def _load_data(self, data_path, label_path, train_walkpath, leave_pair):
#         tmp_data = np.load(data_path).astype(np.float32)
#         tmp_label = np.load(label_path)
#         PAIR_NUM = 52
#         WALK_PATH_NUM = 4

#         # 指定のwalk_pathだけを抽出
#         N, C, T, V, M = tmp_data.shape
#         slash = N // WALK_PATH_NUM
#         walk_data = tmp_data[(train_walkpath-1)*slash : train_walkpath*slash]
#         walk_label = tmp_label[(train_walkpath-1)*slash : train_walkpath*slash]

#         # leave_pair を除外したデータ
#         one_pair_data_num = walk_data.shape[0] // PAIR_NUM
#         all_pair = list(range(1, PAIR_NUM+1))
#         not_leave_pair = [p for p in all_pair if p not in leave_pair]

#         data = np.concatenate([
#             walk_data[(p-1)*one_pair_data_num : p*one_pair_data_num]
#             for p in not_leave_pair
#         ], axis=0)

#         label = np.concatenate([
#             walk_label[(p-1)*one_pair_data_num : p*one_pair_data_num]
#             for p in not_leave_pair
#         ], axis=0)

#         return data, label

#     def __getitem__(self, index):
#         data = self.data[index]
#         label = self.label[index]

#         # データ拡張
#         for aug_name, aug_flag in self.data_augment.items():
#             if aug_flag:
#                 data = getattr(tools, aug_name)(data)

#         return data, label, index


# class WalkPathLeavePairOutTestDataset(BaseDataset):
#     def __init__(self, data_path, label_path, test_walkpath, leave_pair):
#         super().__init__()
#         tmp_data = np.load(data_path).astype(np.float32)
#         tmp_label = np.load(label_path)
#         PAIR_NUM = 52
#         WALK_PATH_NUM = 4

#         N, C, T, V, M = tmp_data.shape
#         slash = N // WALK_PATH_NUM
#         walk_data = tmp_data[(test_walkpath-1)*slash : test_walkpath*slash]
#         walk_label = tmp_label[(test_walkpath-1)*slash : test_walkpath*slash]

#         one_pair_data_num = walk_data.shape[0] // PAIR_NUM
#         leave_pair_num = len(leave_pair)

#         # leave_pair のみ抽出
#         self.data = np.concatenate([
#             walk_data[(p-1)*one_pair_data_num : p*one_pair_data_num]
#             for p in leave_pair
#         ], axis=0)

#         self.label = np.concatenate([
#             walk_label[(p-1)*one_pair_data_num : p*one_pair_data_num]
#             for p in leave_pair
#         ], axis=0)

#     def __getitem__(self, index):
#         data = self.data[index]
#         label = self.label[index]
#         return data, label, index

