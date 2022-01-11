# coding=utf-8
import argparse
import os
import warnings

import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from model import ConfidenceControl

warnings.filterwarnings("ignore")
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm

# from model import ConfidenceControl, ConvAngularPenCC
from utils import ImageReader, MPerClassSampler

import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc

# see : https://docs.cupy.dev/en/stable/install.html
import cupy as cp

def setCuda():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Device:', device)
    if device.type != 'cpu':
        print('Current cuda device:', torch.cuda.current_device())
        print('Count of using GPUs:', torch.cuda.device_count())
        device_count = torch.cuda.device_count()
    else:
        device_count = 1

    return device, device_count


def setSeed():
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parseArgument():
    parser = argparse.ArgumentParser(description='Train Model')

    parser.add_argument('--data_path', default='/data/hbyoo', type=str, help='datasets path')
    parser.add_argument('--data_name', default='mnist', type=str,
                        choices=['car', 'cub', 'sop', 'isc', 'mnist'],
                        help='dataset name')
    parser.add_argument('--crop_type', default='uncropped', type=str, choices=['uncropped', 'cropped'],
                        help='crop data or not, it only works for car or cub dataset')

    parser.add_argument('--batch_size', default=32, type=int, help='train batch size')
    parser.add_argument('--num_sample', default=4, type=int, help='samples within each class')
    parser.add_argument('--feature_dim', default=2048, type=int, help='feature dim')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--lr_gamma', default=0.1, type=float, help='learning rate scheduler gamma')
    parser.add_argument('--num_epochs', default=10, type=int, help='train epoch number')

    opt = parser.parse_args()

    return opt.data_path, opt.data_name, opt.crop_type, opt.batch_size, opt.num_sample, opt.feature_dim, opt.lr, opt.lr_gamma, opt.num_epochs


def loadData(data_path, data_name, crop_type, batch_size, num_sample):
    train_data_set = ImageReader(data_path, data_name, 'train', crop_type)
    test_data_set = ImageReader(data_path, data_name, 'query' if data_name == 'isc' else 'test', crop_type)

    train_sample = MPerClassSampler(train_data_set.labels, batch_size, num_sample)
    train_data_loader = DataLoader(train_data_set, batch_sampler=train_sample, num_workers=4)
    test_data_loader = DataLoader(test_data_set, batch_size, shuffle=False,
                                  num_workers=4)

    number_of_class = len(train_data_set.class_to_idx)
    return train_data_loader, test_data_loader, number_of_class


def loadGalleryData(data_path, data_name, crop_type, batch_size):
    gallery_data_set = ImageReader(data_path, data_name, 'gallery', crop_type)
    gallery_data_loader = DataLoader(gallery_data_set, batch_size, shuffle=False,
                                     num_workers=4)
    return gallery_data_loader


def setEvalDict(data_name, test_data_loader, gallery_data_loader):
    eval_dict = {'test': {'data_loader': test_data_loader}}
    if data_name == 'isc':
        eval_dict['gallery'] = {'data_loader': gallery_data_loader}
    return eval_dict


def setModel(device, feature_dim, number_of_class, model_type, set_as_normal_classifier):
    model = ConfidenceControl(feature_dim, number_of_class, model_type, set_as_normal_classifier)
    # writer.add_graph(model)
    return model.to(device)


def setOptimizer(device, model, lr):
    # 첫번째 에폭에는 feature extractor의 가중치는 변경시키고 싶지 않음, 피쳐 추출기의 가중치는 완벽하니까, 분류기가 그에 맞추도록
    optimizer_init = SGD(
        [{'params': model.feature_extractor.refactor.parameters()}, {'params': model.classifier.parameters()}],
        lr=lr, momentum=0.9, weight_decay=1e-4)
    # 두번째 에폭에는 분류기가 어느정도 안정되었을 것이므로, 피쳐 추출기도 같이 학습시키자.
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

    return optimizer, optimizer_init


def accuracy(net, device, dataloader):
    net.eval()
    correct = 0
    total = len(dataloader.dataset)
    for inputs, labels in tqdm(dataloader, dynamic_ncols=True):
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = net.forward(inputs)
            pred = outputs.argmax(dim=1)
            correct += torch.eq(pred, labels).sum().float().item()

    return correct, total


def train(net, train_data_loader, device, criterion, optimizer):
    total_loss = 0
    iternum = 0
    for inputs, labels in tqdm(train_data_loader, dynamic_ncols=True):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = net.forward(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        iternum += 1

    average_loss = total_loss / iternum
    return average_loss


def get_embed(net, device, dataloader, feature_dim, batch_size):
    embedding_list = np.zeros(shape=(len(dataloader.dataset.labels), feature_dim))
    label_list = np.zeros(shape=(len(dataloader.dataset.labels)))

    iter_num = 0
    for inputs, labels in tqdm(dataloader, dynamic_ncols=True):
        with torch.no_grad():
            outputs = net.forward_feature(inputs.to(device))
        labels = labels.detach().cpu().numpy().ravel()
        number_in_batch = labels.shape[0]
        label_list[iter_num * batch_size:iter_num * batch_size + number_in_batch] = labels
        embedding_list[iter_num * batch_size:iter_num * batch_size + number_in_batch] = outputs.detach().cpu().numpy()

        iter_num += 1

    return embedding_list, label_list



def main(arg_data_name=None):
    device, device_count = setCuda()
    setSeed()
    data_path, data_name, crop_type, batch_size, num_sample, feature_dim, lr, lr_gamma, num_epochs = parseArgument()
    if arg_data_name is not None:
        data_name = arg_data_name

    train_data_loader, test_data_loader, number_of_class = loadData(data_path, data_name, crop_type,
                                                                    batch_size, num_sample)
    model = setModel(device, feature_dim, number_of_class,
                     model_type="resnet",
                     set_as_normal_classifier=True)

    optimizer, optimizer_init = setOptimizer(device, model, lr)
    lr_scheduler = StepLR(optimizer, step_size=15, gamma=lr_gamma)
    loss_criterion = nn.CrossEntropyLoss()



    model.eval()
    train_embedding_list, train_label_list = get_embed(model, device, train_data_loader, feature_dim,
                                                       batch_size)

    train_embedding_list = cp.asarray(train_embedding_list)
    train_embedding_norm = cp.linalg.norm(train_embedding_list, ord=2, axis=1)
    normed_train_embedding_list = train_embedding_list / cp.expand_dims(train_embedding_norm, axis=-1)

    test_embedding_list, test_label_list = get_embed(model, device, test_data_loader, feature_dim, batch_size)
    test_embedding_list = cp.asarray(test_embedding_list)
    test_embedding_norm = cp.linalg.norm(test_embedding_list, ord=2, axis=1)

    normed_test_embedding_list = test_embedding_list / cp.expand_dims(test_embedding_norm, axis=-1)
    fig = plt.figure(figsize=(10, 10))
    for c in np.unique(train_label_list):
        idx = np.where(train_label_list == c)
        emb = normed_train_embedding_list[idx]
        s = cp.linalg.svd(emb, full_matrices=False, compute_uv=False)
        plt.plot(s.get(), c='b')

    for c in np.unique(test_label_list):
        idx = np.where(test_label_list == c)
        emb = normed_test_embedding_list[idx]
        s = cp.linalg.svd(emb, full_matrices=False, compute_uv=False)
        plt.plot(s.get(), c='r')

    plt.plot([], c='b', label='train')
    plt.plot([], c='r', label='test')
    plt.legend()
    writer.add_figure('singular-value-of-embedding', fig, 0)




    for epoch in range(num_epochs):

        model.train()
        average_loss = train(model, train_data_loader, device, loss_criterion,
                             optimizer_init if epoch == 1 else optimizer)
        writer.add_scalar("Loss/average_train_loss", average_loss, epoch)

        if epoch % 2 == 0:
            model.eval()
            train_embedding_list, train_label_list = get_embed(model, device, train_data_loader, feature_dim,
                                                               batch_size)

            train_embedding_list = cp.asarray(train_embedding_list)
            train_embedding_norm = cp.linalg.norm(train_embedding_list, ord=2, axis=1)
            normed_train_embedding_list = train_embedding_list / cp.expand_dims(train_embedding_norm, axis=-1)

            test_embedding_list, test_label_list = get_embed(model, device, test_data_loader, feature_dim, batch_size)
            test_embedding_list = cp.asarray(test_embedding_list)
            test_embedding_norm = cp.linalg.norm(test_embedding_list, ord=2, axis=1)

            normed_test_embedding_list = test_embedding_list / cp.expand_dims(test_embedding_norm, axis=-1)
            fig = plt.figure(figsize=(10, 10))
            for c in np.unique(train_label_list):
                idx = np.where(train_label_list == c)
                emb = normed_train_embedding_list[idx]
                s = cp.linalg.svd(emb, full_matrices=False, compute_uv=False)
                plt.plot(s.get(), c='b')

            for c in np.unique(test_label_list):
                idx = np.where(test_label_list == c)
                emb = normed_test_embedding_list[idx]
                s = cp.linalg.svd(emb, full_matrices=False, compute_uv=False)
                plt.plot(s.get(), c='r')
            plt.plot([], c='b', label='train')
            plt.plot([], c='r', label='test')
            plt.legend()
            writer.add_figure('singular-value-of-embedding', fig, epoch+1)

        if epoch >= 2:
            lr_scheduler.step()





os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # "0""None"

if __name__ == '__main__':

    # for dataset in ['car', 'cub', 'sop', 'isc', 'mnist']:
    dataset = 'cub'
    print(dataset)
    dt = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    writer = SummaryWriter("/data/hbyoo/tensorboard/%s/%s" % (dt, dataset))
    main(arg_data_name=dataset)
    writer.flush()
    writer.close()



    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # plt.scatter(data[:, 0], data[:, 1], data[:, 2])
    # plt.show()
