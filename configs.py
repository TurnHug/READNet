import os
import sys

sys.path.append('..')
from model import READNet

# train path
CSOD10K = "/mnt/d/data/CSOD10K/"
WXSOD = "/mnt/d/data/WXSOD_data/train_sys"

# test path
CSOD10K_test ="/mnt/d/data/CSOD10K/test/"
WXSOD_real = "/mnt/d/data/WXSOD_data/test_real/"
WXSOD_sys = "/mnt/d/data/WXSOD_data/test_sys/"
test_datasets = {'WXSOD_real': WXSOD_real,'WXSOD_sys': WXSOD_sys}
results_path = './results/'
results_save_path = os.path.join('.', 'results/')

image_root = os.path.join(CSOD10K, 'train')
val_root = os.path.join(CSOD10K, 'test')
# image_list = './data/DUTS/DUTS-TR/train_pair.lst'
train_size = 352
batch_size = 16
num_workers = 8
ckpt_path = './saved_models/WXSOD/'
config = {
    'image_root': image_root,
    'val_root': val_root, # Added validation root
    # 'image_list': image_list,
    'train_size': train_size,
    'test_size': train_size,
    'batch_size': batch_size,
    'num_workers': num_workers,
    'mode': 'train',
    'shuffle': True,
    'epochs': 120,
    'lr': 1e-4,
    'momentum': 0.9,
    'weight_decay': 5e-4,
    'pin_memory': True,
    'decay_rate': 0.1,
    'decay_epoch': 50,
    'means': [0.485, 0.456, 0.406],
    'stds': [0.229, 0.224, 0.225],
}


def getNet():
    return READNet().cuda()


def infer(net, inputs):
    final_out,side_out2,side_out3,side_out4= net(inputs)
    return final_out

if __name__ == '__main__':
    print(config['image_root'])
