import os
import scipy.io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import transforms as tvtsf
from skimage import transform as sktsf
from data.util import *
import numpy as np
import pandas as pd
from trainer import VictimFasterRCNNTrainer
from utils import array_tool as at
from tqdm import tqdm
from torch.autograd import Variable
import matplotlib

matplotlib.use('Agg')
import warnings
warnings.filterwarnings('ignore')

def read_image(path, dtype=np.float32, color=True):
    """Read an image from a file.

    This function reads an image from given file. The image is CHW format and
    the range of its value is :math:`[0, 255]`. If :obj:`color = True`, the
    order of the channels is RGB.

    Args:
        path (str): A path of image file.
        dtype: The type of array. The default value is :obj:`~numpy.float32`.
        color (bool): This option determines the number of channels.
            If :obj:`True`, the number of channels is three. In this case,
            the order of the channels is RGB. This is the default behaviour.
            If :obj:`False`, this function returns a grayscale image.

    Returns:
        ~numpy.ndarray: An image.
    """

    f = Image.open(path)
    try:
        if color:
            img = f.convert('RGB')
        else:
            img = f.convert('P')
        img = np.asarray(img, dtype=dtype)
    finally:
        if hasattr(f, 'close'):
            f.close()

    if img.ndim == 2:
        # reshape (H, W) -> (1, H, W)
        return img[np.newaxis]
    else:
        # transpose (H, W, C) -> (C, H, W)
        return img.transpose((2, 0, 1))
    
class WIDERBboxDataset:
    def __init__(self, path_to_label, path_to_image, fname):
        self.path_to_label = path_to_label
        self.path_to_image = path_to_image
        self.f = scipy.io.loadmat(os.path.join(path_to_label, fname))
        self.event_list = self.f.get('event_list')
        self.file_list = self.f.get('file_list')
        self.face_bbx_list = self.f.get('face_bbx_list')
        self.label_names = WIDER_BBOX_LABEL_NAMES
        self.im_list, self.bbox_list = self.get_img_list()
        # ipdb.set_trace()
        self.is_difficult = False

    def get_img_list(self):
        im_list = []
        bbox_list = []
        for event_idx, event in enumerate(self.event_list):
            directory = event[0][0]
            for im_idx, im in enumerate(self.file_list[event_idx][0]):
                im_name = im[0][0]
                im_list.append(os.path.join(self.path_to_image, directory,\
                        im_name + '.jpg'))
                face_bbx = self.face_bbx_list[event_idx][0][im_idx][0]
                bbox_list.append(face_bbx)
        return im_list,bbox_list

    def __len__(self):
        return len(self.im_list)

    def get_example(self, i):
        """Returns the i-th example.
        Returns a color image and bounding boxes. The image is in CHW format.
        The returned image is RGB.

        Args:
            i (int): The index of the example.

        Returns:
            tuple of an image and bounding boxes
        """
        # Load a image
        img_file = self.im_list[i]
        face_bbx = self.bbox_list[i]
        img = read_image(img_file, color=True)
        bboxes = []
        label = []
        difficult = []
        for i in range(face_bbx.shape[0]):
            xmin = int(face_bbx[i][0])
            ymin = int(face_bbx[i][1])
            xmax = int(face_bbx[i][2]) + xmin
            ymax = int(face_bbx[i][3]) + ymin
            bboxes.append((ymin, xmin, ymax, xmax))
            label.append(WIDER_BBOX_LABEL_NAMES.index('Face'))
            difficult.append(self.is_difficult)
        bboxes = np.stack(bboxes).astype(np.float32)
        label = np.stack(label).astype(np.int32)
        difficult = np.array(difficult, dtype=np.bool_).astype(np.uint8)  # PyTorch don't support np.bool
        return img, bboxes, label, difficult

    __getitem__ = get_example


WIDER_BBOX_LABEL_NAMES = (
    'Face')

def pytorch_normalize(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    # normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
    #                             std=[0.229, 0.224, 0.225])
    normalize = tvtsf.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])
    img = normalize(t.from_numpy(img))
    return img.numpy()

def preprocess(img, min_size=600, max_size=1000):
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

    img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect')
    # both the longer and shorter should be less than
    # max_size and min_size

    # Only using pytorch normalize
    normalize = pytorch_normalize
    return normalize(img)

class Transform(object):
    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img = preprocess(img, self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = resize_bbox(bbox, (H, W), (o_H, o_W))

        # # horizontally flip
        img, params = random_flip(img, x_random=True, return_param=True)
        bbox = flip_bbox(bbox, (o_H, o_W), x_flip=params['x_flip'])

        return img, bbox, label, scale
    
# Only for wider face
class Dataset:
    def __init__(self, label_dir, data_dir, fname_mat, min_size=600, max_size=1000):
        self.db = WIDERBboxDataset(label_dir, data_dir, fname_mat)
        self.tsf = Transform(min_size, max_size)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)

        img, bbox, label, scale = self.tsf((ori_img, bbox, label))
        # TODO: check whose stride is negative to fix this instead copy all
        # some of the strides of a given numpy array are negative.
        return img.copy(), bbox.copy(), label.copy(), scale

    def __len__(self):
        return len(self.db)
    
from torch.utils import data as data_

# opt
wider_label_dir = '/home/son/Desktop/Research/simple-faster-rcnn-pytorch/data/wider_face_split'
wider_data_dir = '/home/son/Desktop/Research/simple-faster-rcnn-pytorch/data/WIDER_train/images'
wider_fname_mat = 'wider_face_train'
num_workers = 1

dataset = Dataset(wider_label_dir, wider_data_dir, wider_fname_mat)
dataloader = data_.DataLoader(dataset, \
                              batch_size=1, \
                              shuffle=True, \
                              pin_memory=True,\
                              num_workers=num_workers)

from model.faster_rcnn_vgg16 import FasterRCNNVGG16

faster_rcnn = FasterRCNNVGG16()
class DCGAN(nn.Module):
    def __init__(self, num_channels=3, ngf=100, cg=0.05, learning_rate=1e-4, train_adv=False):
        """
		Initialize a DCGAN. Perturbations from the GAN are added to the inputs to
		create adversarial attacks.

		- num_channels is the number of channels in the input
		- ngf is size of the conv layers
		- cg is the normalization constant for perturbation (higher means encourage smaller perturbation)
		- learning_rate is learning rate for generator optimizer
		- train_adv is whether the model being attacked should be trained adversarially
		"""
        super(DCGAN, self).__init__()
        self.generator = nn.Sequential(
            # 3 x 32 x 32
            nn.Conv2d(num_channels, ngf, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),

            # 1000 x 32 x 32
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),

            # 1000 x 32 x 32
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),

            # 1000 x 32 x 32
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),

            # 1000 x 32 x 32
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),

            # 1000 x 32 x 32
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),

            # 1000 x 32 x 32
            nn.Conv2d(ngf, ngf, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),

            # 3 x 32 x32
            nn.Conv2d(ngf, num_channels, 1, 1, 0, bias=True),
            nn.Tanh()
        )
        self.cuda = t.cuda.is_available()

        if self.cuda:
            self.generator.cuda()
            cudnn.benchmark = True
        
        self.criterion = nn.CrossEntropyLoss(size_average=False)
        self.cg = cg
        self.optimizer = optim.Adam(self.generator.parameters(), lr=learning_rate)
        self.train_adv = train_adv
        self.max_iter = 20
        self.c_misclassify = 1
        self.confidence = 0

    def forward(self, inputs, model, labels=None, bboxes=None, scale=None,\
                model_feats=None, model_optimizer=None, *args):
        num_unperturbed = 10 # cái này là cái gì???
        iter_cnt = 0
        loss_perturb = 20
        loss_misclassify = 10

        while loss_misclassify > 0 and loss_perturb > 1:
            perturbation = self.generator(inputs)
            adv_inputs = inputs + perturbation
            adv_inputs = t.clamp(adv_inputs, -1.0, 1.0) # giới hạn các pixel trong khoảng -1 dến 1
            scores, gt_labels = model(adv_inputs, bboxes, labels, scale, attack=True)

attacker = DCGAN(train_adv=False)

def inverse_normalize(img):
    return (img * 0.5 + 0.5).clip(min=0, max=1) * 255

from utils.vis_tool import visdom_bbox

trainer = VictimFasterRCNNTrainer(faster_rcnn, attacker).cuda()
trainer.vis.text(dataset.db.label_names, win='labels')

for epoch in range(20):
    trainer.reset_meters()
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{20}", leave=False)
    for ii, (img, bbox_, label_, scale) in enumerate(progress_bar):
        scale = at.scalar(scale)
        img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
        img, bbox, label = Variable(img), Variable(bbox), Variable(label)
        adv_inputs = trainer.train_step(img, bbox, label, scale)
        # # print(adv_inputs)
        
        # ori_img_ = inverse_normalize(at.tonumpy(img[0]))
        # gt_img = visdom_bbox(ori_img_,
        #                      at.tonumpy(bbox_[0]),
        #                      at.tonumpy(label_[0]))
        # trainer.vis.img('gt_img', gt_img)
        # _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
        # pred_img = visdom_bbox(ori_img_,
        #                        at.tonumpy(_bboxes[0]),
        #                        at.tonumpy(_labels[0]).reshape(-1),
        #                        at.tonumpy(_scores[0]))
        # trainer.vis.img('pred_img', pred_img)

        # # rpn confusion matrix(meter)
        # trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
        # roi confusion matrix
        # trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())
        if epoch == 13:
            best_path = trainer.save(epochs=epoch)
            break
    # break