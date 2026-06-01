import argparse

import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import sys

sys.path.append('mobilevit_v3/')
import time
import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
# from eval.score_config import *
from configs import *

sys.path.append('')

from train_utils.misc import check_mkdir
import ttach as tta

results_save_path += 'predictions/'
torch.manual_seed(42)
# torch.cuda.set_device(0)
# ckpt_path = 'saved_models/FTN/'
parser = argparse.ArgumentParser()
parser.add_argument('--s', type=str,default="READNet_best.pth" ,help='snapshot')
# parser.add_argument('--mode', required=True, type=str, help='e p d h t')
parser.add_argument('--train_size', default=(352, 352), type=tuple, help='input size')
opt = parser.parse_args()
args = {
    'snapshot': opt.s,
    'crf_refine': False,
    'save_results': True
}
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()

# transforms = tta.Compose(
#     [
#         # tta.HorizontalFlip(),
#         # tta.Scale(scales=[0.75, 1, 1.25], interpolation='bilinear', align_corners=False),
#         tta.Scale(scales=[1], interpolation='bilinear', align_corners=False),
#     ][
# )
input_size = opt.train_size


def main():
    # load  dateset
    print('--------options--------')
    print('snapshot: ', opt.s)
    print('input size ', opt.train_size)
    net = getNet()
    print('load snapshot \'%s\' for testing' % args['snapshot'])
    net.load_state_dict(torch.load(os.path.join(ckpt_path, args['snapshot'])))
    net.eval()
    time_s = time.time()
    img_num = 0
    with torch.no_grad():
        for name, root in test_datasets.items():
            print(name, root)
            root1 = os.path.join(root, 'input')
            img_list = [os.path.splitext(f) for f in os.listdir(root1)]
            img_num += len(img_list)
            for idx, img_name in enumerate(img_list):
                img_name = img_name[0]
                print('predicting for %s/%s: %d / %d' % (name, img_name, idx + 1, len(img_list)))
                rgb_png_path = os.path.join(root1, img_name+'.png')
                rgb_jpg_path = os.path.join(root1, img_name+'.jpg')
                # img = Image.open(rgb_png_path).convert('RGB')
                if os.path.exists(rgb_png_path):
                    img = Image.open(rgb_png_path).convert('RGB')
                else:
                    img = Image.open(rgb_jpg_path).convert('RGB')
                w_, h_ = img.size
                img_resize = img.resize(input_size, Image.BILINEAR)
                img_var = Variable(img_transform(img_resize).unsqueeze(0)).cuda()
                n, c, h, w = img_var.size()
                # mask = []
                # for transformer in transforms:  # custom transforms or e.g. tta.aliases.d4_transform()
                #     rgb_trans = transformer.augment_image(img_var)
                prediction = infer(net, img_var)
                # deaug_mask = transformer.deaugment_mask(out)
                # mask.append(deaug_mask)
                # prediction = torch.mean(torch.stack(mask, dim=0), dim=0)
                prediction = prediction.sigmoid()
                prediction = to_pil(prediction.data.squeeze(0).cpu())
                prediction = prediction.resize((w_, h_), Image.BILINEAR)
                # if args['crf_refine']:
                #     prediction = crf_refine(np.array(img), np.array(prediction))
                if args['save_results']:
                    check_mkdir(os.path.join(results_save_path, args['snapshot'], name))
                    image_name = img_name.split('.')[0]
                    prediction.save(
                        os.path.join(results_save_path, args['snapshot'], name, image_name + '.png'))
    time_e = time.time()
    print('%s images Speed: %f FPS' % (img_num, img_num / (time_e - time_s)))
    print('Test Done!')


if __name__ == '__main__':
    main()
