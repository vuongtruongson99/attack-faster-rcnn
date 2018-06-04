import os
import gc
import ipdb
import pdb
import matplotlib
import attacks
from tqdm import tqdm
from utils.config import opt
from data.dataset import Dataset, TestDataset,inverse_normalize
from data.dataset import pytorch_normalze
from model import FasterRCNNVGG16
from torch.autograd import Variable
from torch.utils import data as data_
from trainer import FasterRCNNTrainer,VictimFasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc
# fix for ulimit
# https://github.com/pytorch/pytorch/issues/973#issuecomment-346405667
import resource
# rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
# resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))
matplotlib.use('agg')

def eval(dataloader, faster_rcnn, test_num=2000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader),total=test_num):
        sizes = [sizes[0][0], sizes[1][0]]
        try:
            pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        except:
            ipdb.set_trace()
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result

def train(**kwargs):
    opt._parse(kwargs)
    dataset = Dataset(opt)
    # 300w_dataset = FaceLandmarksDataset()
    print('load data')
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  pin_memory=True,\
                                  num_workers=opt.num_workers)
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=True
                                       )
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    attacker = attacks.DCGAN(train_adv=False)
    if opt.load_attacker:
        attacker.load(opt.load_attacker)
        print('load attacker model from %s' % opt.load_attacker)
    trainer = VictimFasterRCNNTrainer(faster_rcnn,attacker,attack_mode=True).cuda()
    # trainer = VictimFasterRCNNTrainer(faster_rcnn).cuda()
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)

    trainer.vis.text(dataset.db.label_names, win='labels')
    # eval_result = eval(test_dataloader, faster_rcnn, test_num=2000)
    best_map = 0
    for epoch in range(opt.epoch):
        trainer.reset_meters(adv=True)
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            ipdb.set_trace()
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            img, bbox, label = Variable(img), Variable(bbox), Variable(label)
            trainer.train_step(img, bbox, label, scale)

            if (ii) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                # plot loss
                trainer.vis.plot_many(trainer.get_meter_data())
                trainer.vis.plot_many(trainer.get_meter_data(adv=True))

                # plot groud truth bboxes
                ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                gt_img = visdom_bbox(ori_img_,
                                     at.tonumpy(bbox_[0]),
                                     at.tonumpy(label_[0]))
                trainer.vis.img('gt_img', gt_img)

                # plot predicted bboxes
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
                pred_img = visdom_bbox(ori_img_,
                                       at.tonumpy(_bboxes[0]),
                                       at.tonumpy(_labels[0]).reshape(-1),
                                       at.tonumpy(_scores[0]))
                trainer.vis.img('pred_img', pred_img)
                if trainer.attacker is not None:
                    adv_img = trainer.attacker.perturb(img)
                    adv_img_ = inverse_normalize(at.tonumpy(adv_img[0]))
                    _bboxes, _labels, _scores = trainer.faster_rcnn.predict([adv_img_], visualize=True)
                    adv_pred_img = visdom_bbox(adv_img_,
                                           at.tonumpy(_bboxes[0]),
                                           at.tonumpy(_labels[0]).reshape(-1),
                                           at.tonumpy(_scores[0]))
                    trainer.vis.img('adv_img', adv_pred_img)
                # rpn confusion matrix(meter)
                trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                # roi confusion matrix
                trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())

                if (ii) % 500 == 0:
                    best_path = trainer.save(epochs=epoch,save_rcnn=True)

        if epoch % 2 == 0:
            best_path = trainer.save(epochs=epoch)

if __name__ == '__main__':
    import fire
    fire.Fire()
