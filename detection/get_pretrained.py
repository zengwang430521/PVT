import argparse
import torch


parser = argparse.ArgumentParser(description='Train a detector')
parser.add_argument('src', help='train config file path')
parser.add_argument('tar', help='the dir to save logs and models')
args = parser.parse_args()

src = torch.load(args.src,  map_location='cpu')
tar = src['model']
torch.save(tar, args.tar)

