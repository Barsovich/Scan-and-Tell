'''
    Modified from: https://github.com/daveredrum/ScanRefer/blob/master/scripts/train.py

'''

import os
import json
import torch
import torch.optim as optim
import numpy as np

from torch.utils.data import DataLoader
from datetime import datetime
from copy import deepcopy

from data.scannet.model_util_scannet import ScannetDatasetConfig
from data.dataset_votenet import ScannetReferenceDataset
from lib.solver import Solver
from config.config_votenet import CONF
from config.config_pointgroup import cfg as args
from models.capnet import CapNet

SCANREFER_TRAIN = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_train.json")))
SCANREFER_VAL = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_val.json")))

# constants
DC = ScannetDatasetConfig()

def get_dataloader(args, scanrefer, all_scene_list, split, config, augment,scan2cad_rotation=None):
    dataset = ScannetReferenceDataset(
        scanrefer=scanrefer, 
        scanrefer_all_scene=all_scene_list, 
        split=split, 
        num_points=args.num_points, 
        use_height=(not args.no_height),
        use_color=args.use_color, 
        use_normal=args.use_normal, 
        use_multiview=args.use_multiview,
        augment=augment,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    return dataset, dataloader

def get_model(args,dataset,device):
    # initiate model
    input_channels = int(args.use_multiview) * 128 + int(args.use_normal) * 3 + int(args.use_color) * 3 + int(not args.no_height)
    model = CapNet(
        num_class=DC.num_class,
        vocabulary=dataset.vocabulary,
        embeddings=dataset.glove,
        num_heading_bin=DC.num_heading_bin,
        num_size_cluster=DC.num_size_cluster,
        mean_size_arr=DC.mean_size_arr,
        input_feature_dim=input_channels,
        num_proposal=args.num_proposals,
        no_caption=args.no_caption,
        use_topdown=args.use_topdown,
        num_locals=args.num_locals,
        query_mode=args.query_mode,
        graph_mode=args.graph_mode,
        num_graph_steps=args.num_graph_steps,
        use_relation=args.use_relation,
        use_orientation=args.use_orientation,
        use_distance=args.use_distance,
        use_new=args.use_new
    )

    # trainable model
    if args.use_pretrained:
        # load model
        print("loading pretrained VoteNet...")
        pretrained_model = CapNet(
            num_class=DC.num_class,
            vocabulary=dataset.vocabulary,
            embeddings=dataset.glove,
            num_heading_bin=DC.num_heading_bin,
            num_size_cluster=DC.num_size_cluster,
            mean_size_arr=DC.mean_size_arr,
            num_proposal=args.num_proposals,
            input_feature_dim=input_channels,
            no_caption=True
        )

        pretrained_path = os.path.join(CONF.PATH.OUTPUT, args.use_pretrained, "model_last.pth")
        pretrained_model.load_state_dict(torch.load(pretrained_path), strict=False)

        # mount
        model.backbone_net = pretrained_model.backbone_net
        model.vgen = pretrained_model.vgen
        model.proposal = pretrained_model.proposal

        if args.no_detection:
            # freeze pointnet++ backbone
            for param in model.backbone_net.parameters():
                param.requires_grad = False

            # freeze voting
            for param in model.vgen.parameters():
                param.requires_grad = False
            
            # freeze detector
            for param in model.proposal.parameters():
                param.requires_grad = False
    
    # to device
    model.to(device)

    return model

def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = int(sum([np.prod(p.size()) for p in model_parameters]))

    return num_params

def get_solver(args, dataset, dataloader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(args, dataset["train"], device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    if args.use_checkpoint:
        print("loading checkpoint {}...".format(args.use_checkpoint))
        stamp = args.use_checkpoint
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        checkpoint = torch.load(os.path.join(CONF.PATH.OUTPUT, args.use_checkpoint, "checkpoint.tar"))
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if args.tag: stamp += "_"+args.tag.upper()
        root = os.path.join(CONF.PATH.OUTPUT, stamp)
        os.makedirs(root, exist_ok=True)

    # scheduler parameters for training solely the detection pipeline
    LR_DECAY_STEP = [80, 120, 160] if args.no_caption else None
    LR_DECAY_RATE = 0.1 if args.no_caption else None
    BN_DECAY_STEP = 20 if args.no_caption else None
    BN_DECAY_RATE = 0.5 if args.no_caption else None

    solver = Solver(
        model=model, 
        device=device,
        config=DC, 
        dataset=dataset,
        dataloader=dataloader, 
        optimizer=optimizer, 
        stamp=stamp, 
        val_step=args.val_step,
        detection=not args.no_detection,
        caption=not args.no_caption, 
        orientation=args.use_orientation,
        distance=args.use_distance,
        use_tf=args.use_tf,
        report_ap=args.report_ap,
        lr_decay_step=LR_DECAY_STEP,
        lr_decay_rate=LR_DECAY_RATE,
        bn_decay_step=BN_DECAY_STEP,
        bn_decay_rate=BN_DECAY_RATE,
        criterion=args.criterion
    )
    num_params = get_num_params(model)

    return solver, num_params, root

def save_info(args, root, num_params, dataset):
    info = {}
    for key, value in vars(args).items():
        info[key] = value
    
    info["num_train"] = len(dataset["train"])
    info["num_eval_train"] = len(dataset["eval"]["train"])
    info["num_eval_val"] = len(dataset["eval"]["val"])
    info["num_train_scenes"] = len(dataset["train"].scene_list)
    info["num_eval_train_scenes"] = len(dataset["eval"]["train"].scene_list)
    info["num_eval_val_scenes"] = len(dataset["eval"]["val"].scene_list)
    info["num_params"] = num_params

    with open(os.path.join(root, "info.json"), "w") as f:
        json.dump(info, f, indent=4)

def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META, "scannetv2_{}.txt".format(split)))])

    return scene_list

def get_scanrefer(scanrefer_train, scanrefer_val):

    if args.no_caption:
        train_scene_list = get_scannet_scene_list("train")

        new_scanrefer_train = []
        for scene_id in train_scene_list:
            data = deepcopy(SCANREFER_TRAIN[0])
            data["scene_id"] = scene_id
            new_scanrefer_train.append(data)

        new_scanrefer_eval_train = []
        for scene_id in train_scene_list:
            data = deepcopy(SCANREFER_TRAIN[0])
            data["scene_id"] = scene_id
            new_scanrefer_eval_train.append(data)

        val_scene_list = get_scannet_scene_list("val")
        new_scanrefer_eval_val = []
        for scene_id in val_scene_list:
            data = deepcopy(SCANREFER_VAL[0])
            data["scene_id"] = scene_id
            new_scanrefer_eval_val.append(data)
    else:
        # get initial scene list
        train_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_train])))
        val_scene_list = sorted(list(set([data["scene_id"] for data in scanrefer_val])))
        
        # filter data in chosen scenes
        new_scanrefer_train = []
        for data in scanrefer_train:
            if data["scene_id"] in train_scene_list:
                new_scanrefer_train.append(data)

        # eval on train
        new_scanrefer_eval_train = []
        for scene_id in train_scene_list:
            data = deepcopy(SCANREFER_TRAIN[0])
            data["scene_id"] = scene_id
            new_scanrefer_eval_train.append(data)
        
        new_scanrefer_eval_val = []
        for scene_id in val_scene_list:
            data = deepcopy(SCANREFER_TRAIN[0])
            data["scene_id"] = scene_id
            new_scanrefer_eval_val.append(data)

    # all scanrefer scene
    all_scene_list = train_scene_list + val_scene_list

    print("train on {} samples from {} scenes".format(len(new_scanrefer_train), len(train_scene_list)))
    print("eval on {} scenes from train and {} scenes from val".format(len(new_scanrefer_eval_train), len(new_scanrefer_eval_val)))


    return new_scanrefer_train, new_scanrefer_eval_train, new_scanrefer_eval_val, all_scene_list


def train(args):
    # init training dataset
    print("preparing data...")
    scanrefer_train, scanrefer_eval_train, scanrefer_eval_val, all_scene_list = get_scanrefer(SCANREFER_TRAIN, SCANREFER_VAL)

    # dataloader
    train_dataset, train_dataloader = get_dataloader(args, scanrefer_train, all_scene_list, "train", DC, True, None)
    eval_train_dataset, eval_train_dataloader = get_dataloader(args, scanrefer_eval_train, all_scene_list, "val", DC, False)
    eval_val_dataset, eval_val_dataloader = get_dataloader(args, scanrefer_eval_val, all_scene_list, "val", DC, False)
    dataset = {
        "train": train_dataset,
        "eval": {
            "train": eval_train_dataset,
            "val": eval_val_dataset
        }
    }

    dataloader = {
        "train": train_dataloader,
        "eval": {
            "train": eval_train_dataloader,
            "val": eval_val_dataloader
        }
    }

    print("initializing...")
    solver, num_params, root = get_solver(args, dataset, dataloader)

    print("Start training...\n")
    save_info(args, root, num_params, dataset)
    solver(args.epoch, args.verbose)


if __name__ == "__main__":
    # setting
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # reproducibility
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    train(args)