# -*- coding = utf-8 -*-
# @File Name : extract_features.
# @Date : 2023/6/8 12:17
# @Author : zhiweideng
# @E-mail : zhiweide@usc.edu


import os
import torch
import dataset
import network
import argparse
from tqdm import tqdm
from datetime import date
from train import read_json
from torch.utils.data import DataLoader


def extract_features(config_file, output_folder, model_path, split='train', cuda=True):
    config = read_json(config_file)
    # define the dataset and model
    data_loader = DataLoader(getattr(dataset, config[split]['type'])(**config[split]['args']), batch_size=2)
    model = getattr(network, config['model']['type'])(**config['model']['args'])
    # send to gpu devices
    model = model.cuda() if cuda else model
    checkpoint = torch.load(model_path, map_location=next(model.parameters()).device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # feature hooks and construct the
    features = {}
    output_folder = os.path.join(output_folder,  str(date.today()), config['name'])
    os.makedirs(output_folder, exist_ok=True)

    def feature_hook(name):
        def hook(mod, inp, out):
            features[name] = out.detach()
        return hook

    # register the hook of the model
    model.recon_conv.conv1.register_forward_hook(feature_hook('sem_feat'))
    model.direction_conv.conv1.register_forward_hook(feature_hook('dir_feat'))
    model.radius_conv.conv1.register_forward_hook(feature_hook('rad_feat'))

    # forward process to extract features
    print('Start Feature Extraction Process')
    for idx, batch in enumerate(tqdm(data_loader, desc='0', unit='b')):
        images = batch['image']
        images = images.cuda() if cuda else images
        image_indices = batch['image_id']
        patch_indices = batch['patch_id']
        _ = model(images)
        for i in range(images.size(0)):
            patch_loc = '{}-{}'.format(image_indices[i], patch_indices[i])
            sem_output_file = os.path.join(output_folder, '{}-sem-{}.pt'.format(split, patch_loc))
            dir_output_file = os.path.join(output_folder, '{}-dir-{}.pt'.format(split, patch_loc))
            rad_output_file = os.path.join(output_folder, '{}-rad-{}.pt'.format(split, patch_loc))
            torch.save(features['sem_feat'][i], sem_output_file)
            torch.save(features['dir_feat'][i], dir_output_file)
            torch.save(features['sem_feat'][i], rad_output_file)


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_file', type=str, default='./configs/drive/adaptive_lc.json')
parser.add_argument('-o', '--output_folder', type=str, default='../features')
parser.add_argument('-s', '--split', type=str, default='train')
parser.add_argument('-g', '--gpu', type=bool, default=True)
parser.add_argument('-p', '--model_path', type=str, default='/ifs/loni/faculty/shi/spectrum/zdeng/MSA_Data/' +
                                                            'SpectralVessel/trained_models/ADAPTIVE_LC/2023-06-09/' +
                                                            'ADAPTIVE_LC-1000-epoch-2023-06-09.pt')


if __name__ == '__main__':
    args = parser.parse_args()
    extract_features(args.config_file, args.output_folder, args.model_path, args.split, args.gpu)
