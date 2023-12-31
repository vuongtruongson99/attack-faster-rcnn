from pprint import pprint


# Default Configs for training
# NOTE that, config items could be overwriten by passing argument through command line.
# e.g. --voc-data-dir='./data/'

class Config:
    # data
    # voc_data_dir = '/home/joey/Desktop/faster-rcnn.pytorch/VOCdevkit/VOC2007/'
    wider_label_dir = '/home/son/Desktop/Research/simple-faster-rcnn-pytorch/data/wider_face_split'
    wider_data_dir =  '/home/son/Desktop/Research/simple-faster-rcnn-pytorch/data/WIDER_train/images'
    wider_fname_mat = 'wider_face_train.mat'
    wider_val_data_dir = '/home/son/Desktop/Research/simple-faster-rcnn-pytorch/data/WIDER_val/images'
    wider_val_fname_mat = 'wider_face_val.mat'
    min_size = 600  # image resize
    max_size = 800 # image resize
    num_workers = 8
    test_num_workers = 8

    # sigma for l1_smooth_loss
    rpn_sigma = 3.
    roi_sigma = 1.

    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    weight_decay = 0.0005
    lr_decay = 0.1  # 1e-3 -> 1e-4
    lr = 1e-3


    # visualization
    env = 'faster-rcnn'  # visdom env
    port = 8097
    plot_every = 40  # vis every N iter

    # preset
    data = 'wider'
    pretrained_model = 'vgg16'

    # training
    epoch = 14


    use_adam = True # Use Adam optimizer
    use_chainer = False # try match everything as chainer
    use_drop = False # use dropout in RoIHead
    # debug
    debug_file = '/tmp/debugf'

    # test_num = 3000
    # # model
    # load_path = '/home/joey/Desktop/simple-faster-rcnn-pytorch/checkpoints/fasterrcnn_full_03172016_10'
    # load_attacker = '/home/joey/Desktop/simple-faster-rcnn-pytorch/checkpoints/max_min_attack_4.pth'
    # # load_path = '/home/joey/Desktop/simple-faster-rcnn-pytorch/checkpoints/fasterrcnn_02050841_13'

    # caffe_pretrain = False # use caffe pretrained model instead of torchvision
    # caffe_pretrain_path = 'checkpoints/vgg16-caffe.pth'
    
    test_num = 3000
    # model
    load_path = None

    caffe_pretrain = True # use caffe pretrained model instead of torchvision
    caffe_pretrain_path = '/home/son/Desktop/Research/simple-faster-rcnn-pytorch/checkpoints/vgg16_caffe.pth'

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}


opt = Config()
