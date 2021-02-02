'''
PointGroup train.py
Written by Li Jiang
'''

import torch
import torch.optim as optim
import time, sys, os, random
from tensorboardX import SummaryWriter
import numpy as np
import json 
from copy import deepcopy

from config.config_pointgroup import cfg
from utils.log import logger
from lib.loss_helper import get_pointgroup_cap_loss
from lib.eval_helper import eval_cap_pointgroup
import utils.utils_pointgroup as utils

SCANREFER_TRAIN = json.load(open(os.path.join('data', "ScanRefer_filtered_train.json")))
SCANREFER_VAL = json.load(open(os.path.join('data', "ScanRefer_filtered_val.json")))


def init():
    # copy important files to backup
    backup_dir = os.path.join(cfg.exp_path, 'backup_files')
    os.makedirs(backup_dir, exist_ok=True)
    os.system('cp train_pointgroup.py {}'.format(backup_dir))
    os.system('cp {} {}'.format(cfg.model_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.dataset_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.config, backup_dir))

    # log the config
    logger.info(cfg)

    # summary writer
    global writer
    writer = SummaryWriter(cfg.exp_path)

    # random seed
    random.seed(cfg.manual_seed)
    np.random.seed(cfg.manual_seed)
    torch.manual_seed(cfg.manual_seed)
    torch.cuda.manual_seed_all(cfg.manual_seed)

def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(cfg.data_root,'meta_data', "scannetv2_{}.txt".format(split)))])

    return scene_list

def get_scanrefer(scanrefer_train, scanrefer_val, num_scenes):
    if cfg.no_caption:
        train_scene_list = get_scannet_scene_list("train")
        new_scanrefer_train = []
        for scene_id in train_scene_list:
            data = deepcopy(SCANREFER_TRAIN[0])
            data["scene_id"] = scene_id
            new_scanrefer_train.append(data)

        val_scene_list = get_scannet_scene_list("val")
        new_scanrefer_val = []
        for scene_id in val_scene_list:
            data = deepcopy(SCANREFER_VAL[0])
            data["scene_id"] = scene_id
            new_scanrefer_val.append(data)
    else:
        # get initial scene list
        train_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_train])))
        val_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_val])))
        if num_scenes == -1: 
            num_scenes = len(train_scene_list)
        else:
            assert len(train_scene_list) >= num_scenes
        
        # slice train_scene_list
        train_scene_list = train_scene_list[:num_scenes]

        # filter data in chosen scenes
        new_scanrefer_train = []
        for data in scanrefer_train:
            if data["scene_id"] in train_scene_list:
                new_scanrefer_train.append(data)

        new_scanrefer_val = scanrefer_val

    # all scanrefer scene
    all_scene_list = train_scene_list + val_scene_list

    print("train on {} samples and val on {} samples".format(len(new_scanrefer_train), len(new_scanrefer_val)))

    return new_scanrefer_train, new_scanrefer_val, all_scene_list

def train_epoch(train_loader, model, optimizer, epoch):
    iter_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    am_dict = {}

    model.train()
    start_epoch = time.time()
    end = time.time()

    iteration_count = len(train_loader)

    for i, data_dict in enumerate(train_loader):
        data_time.update(time.time() - end)
        torch.cuda.empty_cache()

        ##### adjust learning rate
        utils.step_learning_rate(optimizer, cfg.lr, epoch - 1, cfg.step_epoch, cfg.multiplier)

        
        #move to cuda 
        for key in data_dict:
            if type(data_dict[key]) == torch.Tensor:
                data_dict[key] = data_dict[key].cuda()
            else:
                pass
        
        ##### prepare input and forward
        data_dict = model(data_dict, epoch, use_tf=True, is_eval=False)
        #loss, _, visual_dict, meter_dict = model_fn(batch, model, epoch)

        ##### loss calculation
        loss, loss_dict, visual_dict, meter_dict = get_pointgroup_cap_loss(data_dict,cfg,epoch,
            not cfg.no_detection, not cfg.no_caption, cfg.use_distance)

        ##### meter_dict
        for k, v in meter_dict.items():
            if k not in am_dict.keys():
                am_dict[k] = utils.AverageMeter()
            am_dict[k].update(v[0], v[1])

        ##### backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ##### time and print
        current_iter = (epoch - 1) * len(train_loader) + i + 1
        max_iter = cfg.epochs * len(train_loader)
        remain_iter = max_iter - current_iter

        iter_time.update(time.time() - end)
        end = time.time()

        remain_time = remain_iter * iter_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if i == iteration_count // 2:
            utils.checkpoint_save_mid_epoch(model, cfg.exp_path, cfg.config.split('/')[-1][:-5], epoch, use_cuda)

        sys.stdout.write(
            "epoch: {}/{} iter: {}/{} loss: {:.4f}({:.4f}) data_time: {:.2f}({:.2f}) iter_time: {:.2f}({:.2f}) remain_time: {remain_time}\n".format
            (epoch, cfg.epochs, i + 1, len(train_loader), am_dict['loss'].val, am_dict['loss'].avg,
             data_time.val, data_time.avg, iter_time.val, iter_time.avg, remain_time=remain_time))
        if (i == len(train_loader) - 1): print()


    logger.info("epoch: {}/{}, train loss: {:.4f}, time: {}s".format(epoch, cfg.epochs, am_dict['loss'].avg, time.time() - start_epoch))

    utils.checkpoint_save(model, cfg.exp_path, cfg.config.split('/')[-1][:-5], epoch, cfg.save_freq, use_cuda)

    for k in am_dict.keys():
        if k in visual_dict.keys():
            writer.add_scalar(k+'_train', am_dict[k].avg, epoch)


def eval_epoch(val_loader, model, epoch,dataset):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    
    am_dict,visual_dict = eval_cap_pointgroup(model,cfg,epoch,dataset,val_loader,
        no_detection=cfg.no_detection,no_caption=cfg.no_caption,force=True)

    # am_dict = {}

    # with torch.no_grad():
    #     model.eval()
    #     start_epoch = time.time()
    #     for i, data_dict in enumerate(val_loader):

    #         #move to cuda 
    #         for key in data_dict:
    #             if type(data_dict[key]) == torch.Tensor:
    #                 data_dict[key] = data_dict[key].cuda()
    #             else:
    #                 pass

    #         ##### prepare input and forward
    #         data_dict = model(data_dict, epoch, use_tf=False, is_eval=True)

    #         loss, loss_dict, visual_dict, meter_dict = get_pointgroup_cap_loss(data_dict,cfg,epoch)

    #         ##### meter_dict
    #         for k, v in meter_dict.items():
    #             if k not in am_dict.keys():
    #                 am_dict[k] = utils.AverageMeter()
    #             am_dict[k].update(v[0], v[1])

    #         ##### print
    #         sys.stdout.write("\riter: {}/{} loss: {:.4f}({:.4f})".format(i + 1, len(val_loader), am_dict['loss'].val, am_dict['loss'].avg))
    #         if (i == len(val_loader) - 1): print()

    logger.info("epoch: {}/{}, val loss: {:.4f}, time: {}s".format(epoch, cfg.epochs, am_dict['loss'].avg, time.time() - start_epoch))

    for k in am_dict.keys():
        if k in visual_dict.keys():
            writer.add_scalar(k + '_eval', am_dict[k].avg, epoch)


if __name__ == '__main__':
    ##### init
    init()

    ##### get model version and data version
    exp_name = cfg.config.split('/')[-1][:-5] #exp_name = pointgroup_default_scannet
    model_name = exp_name.split('_')[0]
    data_name = exp_name.split('_')[-1]

    ##### model
    logger.info('=> creating model ...')

    if model_name == 'pointgroup':
        from models.capnet import CapNet
        #from models.pointgroup import PointGroup as Network
        #from models.pointgroup import model_fn_decorator
    else:
        print("Error: no model - " + model_name)
        exit(0)

    ##### model_fn (criterion)
    #model_fn = model_fn_decorator()

    ##### dataset
    if cfg.dataset == 'scannet_data':
        if data_name == 'scannet':
            scanrefer_train, scanrefer_val, all_scene_list = get_scanrefer(SCANREFER_TRAIN, SCANREFER_VAL, -1)
            scanrefer = {
                "train": scanrefer_train,
                "val": scanrefer_val
            }
            import data.dataset_pointgroup
            dataset = data.dataset_pointgroup.Dataset(scanrefer)
            dataset.trainLoader()
            dataset.valLoader()
        else:
            print("Error: no data loader - " + data_name)
            exit(0)
    
    vocabulary = dataset.vocabulary
    embeddings = dataset.glove

    model = CapNet(vocabulary, embeddings, cfg, 'pointgroup',no_caption=cfg.no_caption,prepare_epochs=cfg.prepare_epochs)

    use_cuda = torch.cuda.is_available()
    logger.info('cuda available: {}'.format(use_cuda))
    assert use_cuda
    model = model.cuda()

    # logger.info(model)
    logger.info('#classifier parameters: {}'.format(sum([x.nelement() for x in model.parameters()])))

    ##### optimizer
    if cfg.optim == 'Adam':
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr)
    elif cfg.optim == 'SGD':
        optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)

    ##### resume
    start_epoch = utils.checkpoint_restore(model, cfg.exp_path, cfg.config.split('/')[-1][:-5], use_cuda)      # resume from the latest epoch, or specify the epoch to restore

    ##### train and val
    for epoch in range(start_epoch, cfg.epochs + 1):
        train_epoch(dataset.train_data_loader, model, optimizer, epoch)

        if utils.is_multiple(epoch, cfg.save_freq,) or utils.is_power2(epoch):
            # eval_epoch(dataset.val_data_loader, model, epoch, dataset)
            pass