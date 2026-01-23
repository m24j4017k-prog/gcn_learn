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


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def get_cut_joint_matrix(data, cut_joint_size):
    batch, c, t, v, m = data.size()
    M = np.ones((batch, c, t, v, m), dtype=np.float32)

    cut_joint = np.random.choice(np.arange(v), cut_joint_size, replace=False)
    M[:, :, :, cut_joint, :] = 0

    return torch.from_numpy(M).to(data.device)


def cutmix_data(data, label, alpha):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size, _, _, v, _ = data.size()
    index = torch.randperm(batch_size).to(data.device)

    cut_joint_size = int(v * (1 - lam))
    if cut_joint_size == 0:
        return data, label, label, 1.0

    M = get_cut_joint_matrix(data, cut_joint_size)

    cutmixed_x = M * data + (1 - M) * data[index]

    lam_adj = 1 - cut_joint_size / v
    y_a, y_b = label, label[index]

    return cutmixed_x, y_a, y_b, lam_adj



def train_one_epoch(processor, epoch):
    model = processor.model
    optimizer = processor.optimizer
    loss_fn = processor.loss
    data_loader = processor.data_loader['train']

    use_mixup = getattr(processor.arg, 'use_mixup', False)
    use_cutmix = getattr(processor.arg, 'use_cutmix', False)
    mixup_alpha = getattr(processor.arg, 'mixup_alpha', 0.5)
    cutmix_alpha = getattr(processor.arg, 'cutmix_alpha', 0.5)

    model.train()
    total_loss, total_acc = 0, 0

    with tqdm(data_loader) as pbar:
        pbar.set_description(f"[Epoch {epoch+1}/{processor.arg.num_epoch}]")

        for data, label, index in pbar:
            data = data.cuda()
            label = label.cuda()

            optimizer.zero_grad()

            if use_cutmix:
                data, y_a, y_b, lam = cutmix_data(data, label, alpha=cutmix_alpha)
                output = model(data)
                loss = lam * loss_fn(output, y_a) + (1 - lam) * loss_fn(output, y_b)

            elif use_mixup:
                data, y_a, y_b, lam = mixup_data(
                    data, label, alpha=mixup_alpha, use_cuda=True
                )
                output = model(data)
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