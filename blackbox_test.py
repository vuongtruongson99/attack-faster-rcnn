from __future__ import division
import os
from PIL import Image
from wider import WIDER
from skimage import transform as sktsf
from data.dataset import Dataset, TestDataset,inverse_normalize
from data.dataset import pytorch_normalze
import numpy as np
import ipdb
import torch
import matplotlib
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.config import opt
from model import FasterRCNNVGG16
from torch.autograd import Variable
from torch.utils import data as data_
from trainer import FasterRCNNTrainer,VictimFasterRCNNTrainer
from utils import array_tool as at
import utils.vis_tool as vz
from utils.vis_tool import visdom_bbox
from utils.vis_tool import vis_bbox,visdom_bbox
from utils.eval_tool import eval_detection_voc
import pdb
from data.util import  read_image
import pandas as pd
from PIL import Image
import attacks
import cv2

WIDER_BBOX_LABEL_NAMES = (
    'Face')

class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.csv_file = '../ext_face/cropped_global_300w.csv'
        self.globalDF = pd.read_csv(self.csv_file)
	self.g_images = self.globalDF['imgPath']
        self.save_dir = '/media/drive/ibug/300W_cropped/frcnn_3/'
        self.save_dir_adv = '/media/drive/ibug/300W_cropped/frcnn_adv_3/'
        self.save_dir_comb = '/media/drive/ibug/300W_cropped/frcnn_comb_3/'
        self.save_dir_perturb = '/media/drive/ibug/300W_cropped/frcnn_perturb_3/'

    def __len__(self):
        return len(self.g_images)

    def preprocess(self, img, min_size=600, max_size=1000):
        """Preprocess an image for feature extraction.

        The length of the shorter edge is scaled to :obj:`self.min_size`.
        After the scaling, if the length of the longer edge is longer than
        :param min_size:
        :obj:`self.max_size`, the image is scaled to fit the longer edge
        to :obj:`self.max_size`.

        After resizing the image, the image is subtracted by a mean image value
        :obj:`self.mean`.

        Args:
            img (~numpy.ndarray): An image. This is in CHW and RGB format.
                The range of its value is :math:`[0, 255]`.
             (~numpy.ndarray): An image. This is in CHW and RGB format.
                The range of its value is :math:`[0, 255]`.

        Returns:
            ~numpy.ndarray:
            A preprocessed image.

        """
        C, H, W = img.shape
        scale1 = min_size / min(H, W)
        scale2 = max_size / max(H, W)
        scale = min(scale1, scale2)
        img = img / 255.
        try:
            img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect')
        except:
            ipdb.set_trace()
        # both the longer and shorter should be less than
        # max_size and min_size
        normalize = pytorch_normalze
        return normalize(img)

    def __getitem__(self, idx):
        img = read_image(self.g_images[idx])
        try:
            img = self.preprocess(img)
        except:
            ipdb.set_trace()
	img = torch.from_numpy(img)[None]
        return img,self.g_images[idx]

def add_bbox(ax,bbox,label,score):
    for i, bb in enumerate(bbox):
        xy = (bb[1], bb[0])
        height = bb[2] - bb[0]
        width = bb[3] - bb[1]
        ax.add_patch(plt.Rectangle(
            xy, width, height, fill=False, edgecolor='red', linewidth=2))

        caption = list()
        label_names = list(WIDER_BBOX_LABEL_NAMES) + ['bg']
        if label is not None and label_names is not None:
            lb = label[i]
            if not (-1 <= lb < len(label_names)):  # modfy here to add backgroud
                raise ValueError('No corresponding name is given')
            caption.append(label_names[lb])
        if score is not None:
            sc = score[i]
            caption.append('{:.2f}'.format(sc))

        if len(caption) > 0:
            ax.text(bb[1], bb[0],
                    ': '.join(caption),
                    style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0})
    return ax

def img2jpg(img,img_suffix,quality):
    jpg_base = '/media/drive/ibug/300W_cropped/frcnn_adv_jpg/'
    img = img.transpose((1, 2, 0))
    img = Image.fromarray(img.astype('uint8'))
    if not os.path.exists(jpg_base):
        os.makedirs(jpg_base)
    jpg_path = jpg_base + img_suffix
    img.save(jpg_path, format='JPEG', subsampling=0, quality=quality)
    jpg_img = read_image(jpg_path)
    return jpg_img

if __name__ == '__main__':
    _data = FaceLandmarksDataset()
    faster_rcnn = FasterRCNNVGG16()
    attacker = attacks.DCGAN(train_adv=False)
    attacker.load('/home/joey/Desktop/simple-faster-rcnn-pytorch/checkpoints/max_min_attack_6.pth')
    trainer = VictimFasterRCNNTrainer(faster_rcnn,attacker).cuda()
    trainer.load('/home/joey/Desktop/simple-faster-rcnn-pytorch/checkpoints/fasterrcnn_full_03172016_10')
    quality_list = [100,90,80,70,60,50,40,30,20]
    threshold = [0.7]
    adv_det_list = []
    for quality in threshold:
        detected = 0
        jpg_detected = 0
        adv_detected = 0
        trainer.faster_rcnn.score_thresh = quality
        for i in range(len(_data)):
            img,im_path = _data[i]
            img = Variable(img.float().cuda())
            im_path_clone = b = '%s' % im_path
            save_path = _data.save_dir + 'frcnn' + im_path.split('/')[-1]
            save_path_adv = _data.save_dir_adv + 'frcnn_adv/' + im_path_clone.split('/')[-1]
            save_path_combined = _data.save_dir_comb + 'frcnn_comb_' + im_path_clone.split('/')[-1]
            save_path_perturb = _data.save_dir_perturb + 'frcnn_perturb_' + im_path_clone.split('/')[-1]
            if not os.path.exists(_data.save_dir):
                os.makedirs(_data.save_dir)
            if not os.path.exists(_data.save_dir_adv):
                os.makedirs(_data.save_dir_adv)
            if not os.path.exists(_data.save_dir_comb):
                os.makedirs(_data.save_dir_comb)
            if not os.path.exists(_data.save_dir_perturb):
                os.makedirs(_data.save_dir_perturb)
            ori_img_ = inverse_normalize(at.tonumpy(img[0]))
            _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_],\
                    new_score=quality,visualize=True)
            adv_img,perturb = trainer.attacker.perturb(img,save_perturb=save_path_perturb)
            adv_img_ = inverse_normalize(at.tonumpy(adv_img[0]))
            perturb = inverse_normalize(at.tonumpy(perturb[0]))
            perturb = perturb.transpose((1,2,0))
            cv2.imwrite(save_path_perturb,perturb)
            # jpg_adv_img_ = img2jpg(adv_img_,im_path_clone.split('/')[-1],quality)
            adv_bboxes, adv_labels, adv_scores = trainer.faster_rcnn.predict([adv_img_], \
                    new_score=quality,visualize=True,adv=True)
            # jpg_adv_bboxes, jpg_adv_labels, jpg_adv_scores = trainer.faster_rcnn.predict([jpg_adv_img_], \
                    # visualize=True,adv=True)
            print("Processing image %d" %(i))
            if adv_labels[0].size != 0:
                adv_detected += 1
            # if jpg_adv_labels[0].size != 0:
                # jpg_detected += 1
            if _labels[0].size != 0:
                detected += 1
            # print("Quality %d Detected Adversarial faces %d, JPG faces %d out of %d" \
                        # %(quality,adv_detected,jpg_detected,i))
            print("Detected %d faces and %d Adv Faces out of %d" %(detected,adv_detected,i))

            # fig, (ax1, ax2) = plt.subplots(1, 2), sharey=True, sharex=True)
            # ax1.axis('off')
            # ax2.axis('off')
            # ax1.imshow(ori_img_.astype(np.uint8))
            # ax2.imshow(adv_img_.astype(np.uint8))
            # ax1 = add_bbox(ax1,at.tonumpy(_bboxes[0]),at.tonumpy(_labels[0]),\
                    # at.tonumpy(_scores[0]))
            # ax2 = add_bbox(ax2,at.tonumpy(adv_bboxes[0]),at.tonumpy(adv_labels[0]),\
                    # at.tonumpy(adv_scores[0]))
            # fig.savefig(save_path_combined,bbox_inches='tight',pad_inches=0)
            # ori_img_ = ori_img_.transpose((1,2,0))
            # adv_img_ = adv_img_.transpose((1,2,0))
            # cv2.imwrite(save_path,ori_img_)
            # cv2.imwrite(save_path_adv,adv_img_)
            # ax = vis_bbox(ori_img_,
                     # at.tonumpy(_bboxes[0]),
                     # at.tonumpy(_labels[0]).reshape(-1),
                     # at.tonumpy(_scores[0]).reshape(-1))
            # ax.figure.savefig(save_path,bbox_inches='tight')
            # ax_2 = vis_bbox(adv_img_,
                     # at.tonumpy(adv_bboxes[0]),
                     # at.tonumpy(adv_labels[0]).reshape(-1),
                     # at.tonumpy(adv_scores[0]).reshape(-1))
            # ax_2.figure.savefig(save_path_adv)#,bbox_inches='tight')
            # ''' Hacky Way to combine the 2 images '''
            # saved_img1 = cv2.imread(save_path)
            # saved_img2 = cv2.imread(save_path_adv)
            # try:
                # vis = np.concatenate((saved_img1, saved_img2), axis=1)
            # except:
                # ipdb.set_trace()
            # cv2.imwrite(save_path_combined, vis)
        print("Detected %d faces out of %d" %(detected,len(_data)))
        print("Detected %d Adversarial faces out of %d" %(adv_detected,len(_data)))
        adv_det_list.append([adv_detected,detected])
    print("Printing ADV List LIST")
    print(adv_det_list)

