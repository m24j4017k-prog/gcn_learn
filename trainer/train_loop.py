import torch
import numpy as np
from tqdm import tqdm

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """MixUp処理関数"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam



def train_one_epoch(processor, epoch):
    model = processor.model
    optimizer = processor.optimizer
    loss_fn = processor.loss
    data_loader = processor.data_loader['train']

    use_mixup = getattr(processor.arg, 'use_mixup', False)
    mixup_alpha = getattr(processor.arg, 'mixup_alpha', 0.5)

    model.train()
    total_loss, total_acc = 0, 0

    with tqdm(data_loader) as pbar:
        pbar.set_description(f"[Epoch {epoch+1}/{processor.arg.num_epoch}]")

        for data, label, index in pbar:
            data = data.cuda()
            label = label.cuda()

            optimizer.zero_grad()

            if use_mixup:
                label_onehot = torch.nn.functional.one_hot(label, num_classes=processor.arg.num_class).float()
                mixed_data, y_a, y_b, lam = mixup_data(data, label_onehot, alpha=mixup_alpha, use_cuda=True)
                output = model(mixed_data)
                loss = lam * loss_fn(output, y_a) + (1 - lam) * loss_fn(output, y_b)
            else:
                output = model(data)
                loss = loss_fn(output, label)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * data.size(0)
            total_acc += (output.max(1)[1] == label).sum().item()

    avg_loss = total_loss / len(data_loader.dataset)
    avg_acc = total_acc / len(data_loader.dataset)

    return avg_loss, avg_acc


def validate_one_epoch(processor):
    model = processor.model
    loss_fn = processor.loss
    data_loader = processor.data_loader['test']

    model.eval()
    total_loss, total_acc = 0, 0

    with torch.no_grad():
        for data, label, index in data_loader:
            data = data.cuda()
            label = label.cuda()

            output = model(data)
            loss = loss_fn(output, label)

            total_loss += loss.item() * data.size(0)
            total_acc += (output.max(1)[1] == label).sum().item()

    avg_loss = total_loss / len(data_loader.dataset)
    avg_acc = total_acc / len(data_loader.dataset)

    return avg_loss, avg_acc