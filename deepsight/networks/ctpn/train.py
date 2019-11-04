"""Class to handle the network training"""
import os
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.optim.optimizer import Optimizer


def train(net: nn.Module, criterion: nn.Module, optimizer: Optimizer,
          train_loader: DataLoader, test_loader: DataLoader,
          epochs: int, use_cuda: bool = False):

    net.train()
    if use_cuda:
        net = net.cuda()
        criterion = criterion.cuda()

    for epoch in range(epochs):
        total_loss = 0.0
        # print("epoch = ", epoch)
        for i, (inputs, targets) in enumerate(train_loader):
            # print("i = ", i)
            if use_cuda:
                inputs = inputs.cuda()

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (i + 1) % 10 == 0:
                print(
                    "epoch = {epoch}, i = {i}, total_loss = {total_loss}". \
                    format(epoch=epoch, i=i, total_loss=total_loss)
                )
                check_loss = total_loss
                total_loss = 0.0

        save(locals())


def save(local_vars: dict, model_folder: str = "./checkpoints"):
    epoch = local_vars["epoch"]
    model = local_vars["net"]
    optimizer = local_vars["optimizer"]
    loss = local_vars["check_loss"]
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    model_name = "{epoch}_{loss}.checkpoint".format(epoch=epoch, loss=loss)
    torch.save(checkpoint, os.path.join(model_folder, model_name))

    clean(model_folder)


def clean(model_folder, top_n: int = 10):
    tops = []
    for f in os.listdir(model_folder):
        if not f.endswith(".checkpoint"):
            continue
        epoch, loss = f.split(".")[0].split("_")
        tops.append((f, float(loss)))

    if len(tops) > top_n:
        tops = sorted(tops, key=lambda x: x[1])
        for f, _ in tops[top_n:]:
            file_path = os.path.join(model_folder, f)
            os.remove(file_path)


if __name__ == "__main__":
    from argparse import ArgumentParser
    import torch
    import torch.optim as optim
    from deepsight.networks.ctpn import CTPN, CTPNLoss, ctpn_transformer
    from deepsight.networks.ctpn import CTPNFolder
    from deepsight.datasets import train_test_split

    parser = ArgumentParser()
    parser.add_argument("-root", type=str,
                        help="GT folder",
                        default="../data/ctpn/")
    parser.add_argument("-epochs", type=int,
                        help="GT folder",
                        default=100)

    args = parser.parse_args()
    root = args.root
    root = "/home/mo/Datasets/ctpn/"
    # root = "/home/mo/Datasets/ctpn_zero/no_pos"
    # root = "/home/mo/Datasets/ctpn_error/"
    epochs = args.epochs

    ctpn_folder = CTPNFolder(root=root,
                             transformer=ctpn_transformer)
    train_dataset, test_dataset = train_test_split(ctpn_folder, test_size=0.1)
    train_loader = DataLoader(train_dataset, shuffle=True)
    test_loader = DataLoader(test_dataset, shuffle=True)

    net = torch.nn.DataParallel(CTPN())
    criterion = CTPNLoss()
    optimizer = optim.SGD(net.parameters(),
                          lr=1e-4,
                          momentum=0.9,
                          weight_decay=1e-5)
    use_cuda = True if torch.cuda.is_available() else False
    print("use_cuda: ", use_cuda)

    train(net, criterion, optimizer, train_loader, test_loader, epochs, use_cuda)
    # train(net, criterion, optimizer, train_loader, train_loader, epochs, use_cuda)
