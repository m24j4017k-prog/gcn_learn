import os
import torch
import torch.optim as optim
import numpy as np

from dataset.builder import build_dataset
from model.builder import build_model
from utils.util import import_class, save_arg
from utils.pytorchtools import EarlyStopping
from trainer.train_loop import train_one_epoch, validate_one_epoch


class Processor():
    def __init__(self, arg):
        
        # ---設定ファイル読み込み ---
        self.arg = arg
        
        # 実行結果ファイルの作成
        if (self.arg.evaluation_method == "walk_path_leave_pair_out"):
            self.work_dir = f"{self.arg.work_dir}/{self.arg.train_walkpath[0]}/{self.arg.leave_pair[0]}-{self.arg.leave_pair[-1]}"
            print("*****************")
            print(self.work_dir)
        else:
            self.work_dir = self.arg.work_dir
        

            
        # ---設定ファイル保存 ---
        save_arg(self.arg, self)
        
        # 実行済みの場合は処理を中止
        if os.path.isfile(f"{self.work_dir}/results/val_acc.npy") and self.arg.phase == 'train':
            print(f'{self.work_dir}/acc_file is already existed')
            exit()
      
        # --- モデル構築 ---
        self.model, ModelClass = build_model(model_path=self.arg.model, model_args=self.arg.model_args)

        # --- 損失関数ロード ---
        self.loss = self.build_loss()
        
        # --- 最適化関数をロード ---
        self.load_optimizer()

        # --- データローダの構築 ---
        self.data_loader = {
            'train': build_dataset(self.arg, split='train'),
            'test': build_dataset(self.arg, split='test') }
        
        # --- 学習履歴 ---
        self.train_loss_list = []
        self.train_acc_list = []
        self.val_loss_list = []
        self.val_acc_list = []
        
        # --- EarlyStopping ---
        self.early_stopping = EarlyStopping(patience=self.arg.patience, path=f"{self.work_dir}/best_model.pt")


    def start(self):
        if self.arg.phase == 'train':
            print("Starting train phase...")
            self.train()

        elif self.arg.phase == 'test':
            
            self.test_work_dir = f"{self.arg.work_dir}/test/{self.arg.train_walkpath[0]}-{self.arg.test_walkpath[0]}/{self.arg.leave_pair[0]}-{self.arg.leave_pair[-1]}"
            if not os.path.exists(self.test_work_dir):
                os.makedirs(self.test_work_dir)
            else:
                print("None train phase")
            

            print("Starting test phase...")
            self.model.load_state_dict(torch.load(f"{self.work_dir}/best_model.pt"))
           
            
            self.test() 

        
    def train(self):
        best_val_acc = 0.0

        for epoch in range(self.arg.num_epoch):
            
            # --- 訓練 ---
            train_loss, train_acc = train_one_epoch(self, epoch=epoch)
            
            # --- バリデーション ---
            val_loss, val_acc = validate_one_epoch(self)
            
            # early_stopping
            self.early_stopping(val_loss, self.model)
            if (self.early_stopping.early_stop):
                break

            # --- 履歴保存 ---
            self.train_loss_list.append(train_loss)
            self.train_acc_list.append(train_acc)
            self.val_loss_list.append(val_loss)
            self.val_acc_list.append(val_acc)
                
                            
            print(f"[Epoch {epoch+1}/{self.arg.num_epoch}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            
            
            # --- EarlyStopping 更新 ---
            self.early_stopping(val_loss, self.model)

            if self.early_stopping.early_stop:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
            
                    
        # --- 学習完了後に最終評価 ---
        self.model.load_state_dict(torch.load(self.early_stopping.path))
        self.model.eval()
        test_loss, test_acc = validate_one_epoch(self)
        print(f"Final Test | Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%")
        
        os.makedirs(f'{self.work_dir}/results', exist_ok=True)
        np.save(f'{self.work_dir}/results/train_loss', self.train_loss_list)
        np.save(f'{self.work_dir}/results/train_acc', self.train_acc_list)    
        np.save(f'{self.work_dir}/results/val_loss', self.val_loss_list)
        np.save(f'{self.work_dir}/results/val_acc', self.val_acc_list)


    def test(self):
        self.model.eval()
        val_loss = 0
        val_acc = 0

        with torch.no_grad():
            for data, label, index in self.data_loader['test']:
                data = data.cuda()
                label = label.cuda()
                
                output = self.model(data)
                loss = self.loss(output, label)
                val_loss += loss.item() 
                val_acc += (output.max(1)[1] == label).sum().item()

        avg_val_loss = val_loss / len(self.data_loader['test'].dataset)
        avg_val_acc = val_acc / len(self.data_loader['test'].dataset)
        
        print (avg_val_loss, avg_val_acc)
    
        # os.makedirs(f'{self.arg.save_dir}/results', exist_ok=True)
        np.save(f'{self.test_work_dir}/val_loss', avg_val_loss)
        np.save(f'{self.test_work_dir}/val_acc', avg_val_acc)

    
    # ==========================
    #  損失関数構築メソッド
    # ==========================
    def build_loss(self):
        from utils.util import import_class
        
        LossClass = import_class(self.arg.loss)   # 例: 'torch.nn.CrossEntropyLoss'
        loss_fn = LossClass(**self.arg.loss_args) # YAMLから渡されたパラメータを展開
        return loss_fn
    
    
    # ==========================
    #  最適化関数構築メソッド
    # ==========================
    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr = self.arg.base_lr,
                momentum = self.arg.momentum,
                nesterov = self.arg.nesterov,
                weight_decay = self.arg.weight_decay                
            )
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError(f"{self.arg.optimizer} is not implemented")