# coding=utf-8
import os

from numpy import ndarray

from model import ConfidenceControl

import argparse

import numpy as np
import pandas as pd
import torch
from thop import profile, clever_format
from torch import nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, ConcatDataset, Subset
from torchvision import transforms
from torchvision.datasets import MNIST

import warnings

warnings.filterwarnings("ignore")
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
from tqdm import tqdm

import torch.nn.functional as F
# from model import ConfidenceControl, ConvAngularPenCC
from utils import recall, ImageReader, MPerClassSampler
from torch.distributions import normal
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import phate

import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc

from sklearn.metrics.pairwise import pairwise_distances


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


def get_embed(net, device, dataloader, feature_dim):
    embedding_list = np.zeros(shape=(0, feature_dim))
    label_list = np.zeros(shape=(0))
    for inputs, labels in tqdm(dataloader, dynamic_ncols=True):
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = net.forward_feature(inputs)
        label_list = np.concatenate((label_list, labels.detach().cpu().numpy().ravel()))
        embedding_list = np.concatenate([embedding_list, outputs.detach().cpu().numpy()], axis=0)

    return embedding_list, label_list

# 2차원 사영된 데이터를 위함
def generate_embedding_space_scatter_figure_with_weight(data, y, number_of_class, weight_mark=True):
    fig = plt.figure(figsize=(10, 10))
    plt.axis('off')
    sns.set_style('darkgrid')
    if weight_mark:
        palette = sns.color_palette(cc.glasbey, n_colors=number_of_class * 2)
        size = np.ones_like(y) + 2
        size[len(size) - number_of_class:len(size)] = 90
        sns.scatterplot(data[:, 0], data[:, 1], s=size, hue=y, legend='full', palette=palette)
    else:
        palette = sns.color_palette(cc.glasbey, n_colors=number_of_class)
        sns.scatterplot(data[:, 0], data[:, 1], hue=y, legend='full', palette=palette)
    return fig




def transform_euclidean_to_hypersphere_coordinate(embedding_list, feature_dim):
    theta = []
    r = np.linalg.norm(embedding_list, ord=2, axis=1)
    phi = np.arctan2(embedding_list[:, 1], embedding_list[:, 0])
    theta.append(r)
    theta.append(phi)
    for i in range(feature_dim - 2):
        theta.append(np.arctan2(np.linalg.norm(embedding_list[:, 0:2 + i], ord=2, axis=1), embedding_list[:, 2 + i]))
    return np.array(theta).T


def generate_feature_radius_dist_fig(model, device, train_data_loader, test_data_loader, feature_dim):
    final_weights = model.classifier.weight.detach().cpu().numpy()
    test_embedding_list, test_label_list = get_embed(model, device, test_data_loader, feature_dim)
    train_embedding_list, train_label_list = get_embed(model, device, train_data_loader, feature_dim)

    fig = plt.figure(figsize=(10, 10))
    rad_test = np.linalg.norm(test_embedding_list, ord=2, axis=1)
    rad_train = np.linalg.norm(train_embedding_list, ord=2, axis=1)
    plt.xlim(0, 100)
    plt.ylim(0.0, 1)
    sns.kdeplot(rad_train, label="train", shade=True)
    sns.kdeplot(rad_test, label="test", shade=True)
    plt.xlabel("radius i.e. confidence")
    plt.legend()
    # plt.show()

    return fig


def main(arg_data_name=None):
    device, device_count = setCuda()
    setSeed()
    data_path, data_name, crop_type, batch_size, num_sample, feature_dim, lr, lr_gamma, num_epochs = parseArgument()
    if arg_data_name is not None:
        data_name = arg_data_name

    train_data_loader, test_data_loader, number_of_class = loadData(data_path, data_name, crop_type,
                                                                    batch_size, num_sample)

    gallery_data_loader = loadGalleryData(data_path, data_name, crop_type, batch_size) if data_name == 'isc' else None

    eval_dict = setEvalDict(data_name, test_data_loader, gallery_data_loader)

    model = setModel(device, feature_dim, number_of_class,
                     model_type="resnet",
                     set_as_normal_classifier=True)

    optimizer, optimizer_init = setOptimizer(device, model, lr)
    lr_scheduler = StepLR(optimizer, step_size=15, gamma=lr_gamma)
    loss_criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):

        model.train()
        average_loss = train(model, train_data_loader, device, loss_criterion,
                             optimizer_init if epoch == 1 else optimizer)
        writer.add_scalar("Loss/average_train_loss", average_loss, epoch)

        if epoch % 2 == 0:
            model.eval()
            fig = generate_feature_radius_dist_fig(model, device, train_data_loader, test_data_loader, feature_dim)
            writer.add_figure('embed-feature-radius-dist', fig, epoch)

        if epoch >= 2:
            lr_scheduler.step()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # "0""None"


if __name__ == '__main__':
    # for dataset in ['car', 'cub', 'sop', 'isc', 'mnist']:
    dataset = 'isc'

    dt = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    print('%s/%s' % (dt, dataset))
    writer = SummaryWriter("/home/hbyoo/tensorboard/%s/%s" % (dt, dataset))
    main(arg_data_name=dataset)
    writer.flush()
    writer.close()
