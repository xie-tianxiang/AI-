#--------------------------------------------------------
# corruption evaluation on CIFAR-10-C
#--------------------------------------------------------
import torch
import numpy as np
import torchvision
from VGG import *
import os
import argparse


def get_test_loader(x, y, test_batch_size):
    test_data = torch.from_numpy(x).float()
    test_label = torch.from_numpy(y).long()
    test_dataset = torch.utils.data.TensorDataset(test_data, test_label)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=test_batch_size,
                                              shuffle=False,
                                              drop_last=True)
    return test_loader


def test_model(model, data_loader, distortion_name, model_name):
    model.eval()
    correct = 0
    total = 0
    for step, (x, y) in enumerate(data_loader):
        # x = x.cuda()
        # y = y.cuda()
        with torch.no_grad():
            h = model(x)
        _, predicted = torch.max(h.data, 1)
        total += y.size(0)
        correct += (predicted == y).sum().item()
    acc_rate = 100.0 * correct / total
    err_rate = 100.0 - acc_rate
    print('Error rate of the model VGG_' + model_name + ' on the ' + distortion_name + ': {:.2f} %'.format(err_rate))
    model.train(True)
    return err_rate


#calculate all kinds of corruption's CE and mCE(unnormalized) of one model 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CIFAR10-C test')
    parser.add_argument('--batchsize', type=int, default=128, help='model batch size')
    parser.add_argument('--model_name', default="NAT", help='model\'s name')
    parser.add_argument('--model_path', default="/vgg16_all/lat_param.pkl", help='model path')
    parser.add_argument('--distotion_root', default="E:/AI安全/代码/ANP-master/CIFAR-10-C/",
                        help='the path of the folder which contains all kinds of distotions.npy and label.npy')
    args = parser.parse_args()

    #initial setting   
    distortion_name = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur',
                       'zoom_blur',
                       'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate',
                       'jpeg_compression', 'speckle_noise',
                       'gaussian_blur', 'spatter', 'saturate']
    laebl_name = 'labels.npy'

    #load model
    net = VGG16(enable_lat=False,
                alpha=0.6,
                epsilon=0,
                pro_num=1,
                batch_size=args.batchsize,
                layerlist="all",
                if_dropout=True)
    #net.cuda()
    net.load_state_dict(torch.load(args.model_path, weights_only=True))

    error_rates = []
    for i in range(len(distortion_name)):
        data_root = args.distotion_root + distortion_name[i] + '.npy'
        label_root = args.distotion_root + laebl_name
        #load data
        x = np.load(data_root)
        x = x.transpose((0, 3, 1, 2))
        x = x / 255.0
        y = np.load(label_root)
        #data_loader
        test_loader = get_test_loader(x, y, args.batchsize)
        err = test_model(net, test_loader, distortion_name[i], args.model_name)
        error_rates.append(err)

    print('mCE (unnormalized by VGG_' + args.model_name + ' errors) (%):{:.2f}'.format(np.mean(error_rates)))
