# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import os
import sys
import json
import torch
import pickle
import argparse
import time

import numpy as np

from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import DataLoader
from numpy.linalg import inv
from lib.pointgroup_ops.functions import pointgroup_ops

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder

import lib.capeval.bleu.bleu as capblue
import lib.capeval.cider.cider as capcider
import lib.capeval.rouge.rouge as caprouge
import lib.capeval.meteor.meteor as capmeteor

from data.scannet.model_util_scannet import ScannetDatasetConfig
from config.config_votenet import CONF
#from utils.nn_distance import nn_distance, huber_loss
from lib.ap_helper import parse_predictions
from lib.loss import SoftmaxRankingLoss
from utils.box_util import box3d_iou, box3d_iou_batch_tensor
import utils.utils_pointgroup as utils_pointgroup
from lib.loss_helper import get_scene_cap_loss, get_pointgroup_cap_loss
import utils.pointgroup.eval as pg_eval
from utils.eval_det import get_iou
import lib.ap_helper as ap_helper

# constants
DC = ScannetDatasetConfig()

SCANREFER = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered.json")))
SCANREFER_ORGANIZED = json.load(open(os.path.join(CONF.PATH.DATA, "ScanRefer_filtered_organized.json")))

def prepare_corpus(scanrefer, max_len=CONF.TRAIN.MAX_DES_LEN):
    scene_ids = list(set([data["scene_id"] for data in scanrefer]))

    corpus = {}
    for data in SCANREFER:
        scene_id = data["scene_id"]

        if scene_id not in scene_ids: continue

        object_id = data["object_id"]
        object_name = data["object_name"]
        token = data["token"][:max_len]
        description = " ".join(token)

        # add start and end token
        description = "sos " + description
        description += " eos"

        key = "{}|{}|{}".format(scene_id, object_id, object_name)
        # key = "{}|{}".format(scene_id, object_id)

        if key not in corpus:
            corpus[key] = []

        corpus[key].append(description)

    return corpus

def decode_caption(raw_caption, idx2word):
    decoded = ["sos"]
    for token_idx in raw_caption:
        token_idx = token_idx.item()
        token = idx2word[str(token_idx)]
        decoded.append(token)
        if token == "eos": break

    if "eos" not in decoded: decoded.append("eos")
    decoded = " ".join(decoded)

    return decoded

def check_candidates(corpus, candidates):
    placeholder = "sos eos"
    corpus_keys = list(corpus.keys())
    candidate_keys = list(candidates.keys())
    missing_keys = [key for key in corpus_keys if key not in candidate_keys]

    if len(missing_keys) != 0:
        for key in missing_keys:
            candidates[key] = [placeholder]

    return candidates

def organize_candidates(corpus, candidates):
    new_candidates = {}
    for key in corpus.keys():
        new_candidates[key] = candidates[key]

    return new_candidates

def eval_ref_one_sample(pred_bbox, gt_bbox):
    """ Evaluate one reference prediction

    Args:
        pred_bbox: 8 corners of prediction bounding box, (8, 3)
        gt_bbox: 8 corners of ground truth bounding box, (8, 3)
    Returns:
        iou: intersection over union score
    """

    iou = box3d_iou(pred_bbox, gt_bbox)

    return iou

def construct_bbox_corners(center, box_size):
    sx, sy, sz = box_size
    x_corners = [sx/2, sx/2, -sx/2, -sx/2, sx/2, sx/2, -sx/2, -sx/2]
    y_corners = [sy/2, -sy/2, -sy/2, sy/2, sy/2, -sy/2, -sy/2, sy/2]
    z_corners = [sz/2, sz/2, sz/2, sz/2, -sz/2, -sz/2, -sz/2, -sz/2]
    corners_3d = np.vstack([x_corners, y_corners, z_corners])
    corners_3d[0,:] = corners_3d[0,:] + center[0];
    corners_3d[1,:] = corners_3d[1,:] + center[1];
    corners_3d[2,:] = corners_3d[2,:] + center[2];
    corners_3d = np.transpose(corners_3d)

    return corners_3d

def decode_detections(data_dict):
    pred_center = data_dict['center'].detach().cpu().numpy() # (B,K,3)
    pred_heading_class = torch.argmax(data_dict['heading_scores'], -1) # B,num_proposal
    pred_heading_residual = torch.gather(data_dict['heading_residuals'], 2, pred_heading_class.unsqueeze(-1)).detach().cpu().numpy() # B,num_proposal
    pred_heading_class = pred_heading_class.detach().cpu().numpy() # B,num_proposal
    pred_size_class = torch.argmax(data_dict['size_scores'], -1) # B,num_proposal
    pred_size_residual = torch.gather(data_dict['size_residuals'], 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)) # B,num_proposal,1,3
    pred_size_class = pred_size_class.detach().cpu().numpy() # B,num_proposal
    pred_size_residual = pred_size_residual.squeeze(2).detach().cpu().numpy() # B,num_proposal,3

    batch_size, num_bbox, _ = pred_center.shape
    bbox_corners = []
    for batch_id in range(batch_size):
        batch_corners = []
        for bbox_id in range(num_bbox):
            pred_obb = DC.param2obb(pred_center[batch_id, bbox_id], pred_heading_class[batch_id, bbox_id], pred_heading_residual[batch_id, bbox_id],
                    pred_size_class[batch_id, bbox_id], pred_size_residual[batch_id, bbox_id])
            pred_bbox = construct_bbox_corners(pred_obb[0:3], pred_obb[3:6])
            batch_corners.append(pred_bbox)
        
        batch_corners = np.stack(batch_corners, axis=0)
        bbox_corners.append(batch_corners)

    bbox_corners = np.stack(bbox_corners, axis=0) # batch_size, num_proposals, 8, 3

    return bbox_corners

def decode_targets(data_dict):
    pred_center = data_dict['center_label'] # (B,MAX_NUM_OBJ,3)
    pred_heading_class = data_dict['heading_class_label'] # B,K2
    pred_heading_residual = data_dict['heading_residual_label'] # B,K2
    pred_size_class = data_dict['size_class_label'] # B,K2
    pred_size_residual = data_dict['size_residual_label'] # B,K2,3

    # assign
    pred_center = torch.gather(pred_center, 1, data_dict["object_assignment"].unsqueeze(2).repeat(1, 1, 3)).detach().cpu().numpy()
    pred_heading_class = torch.gather(pred_heading_class, 1, data_dict["object_assignment"]).detach().cpu().numpy()
    pred_heading_residual = torch.gather(pred_heading_residual, 1, data_dict["object_assignment"]).unsqueeze(-1).detach().cpu().numpy()
    pred_size_class = torch.gather(pred_size_class, 1, data_dict["object_assignment"]).detach().cpu().numpy()
    pred_size_residual = torch.gather(pred_size_residual, 1, data_dict["object_assignment"].unsqueeze(2).repeat(1, 1, 3)).detach().cpu().numpy()

    batch_size, num_bbox, _ = pred_center.shape
    bbox_corners = []
    for batch_id in range(batch_size):
        batch_corners = []
        for bbox_id in range(num_bbox):
            pred_obb = DC.param2obb(pred_center[batch_id, bbox_id], pred_heading_class[batch_id, bbox_id], pred_heading_residual[batch_id, bbox_id],
                    pred_size_class[batch_id, bbox_id], pred_size_residual[batch_id, bbox_id])
            pred_bbox = construct_bbox_corners(pred_obb[0:3], pred_obb[3:6])
            batch_corners.append(pred_bbox)
        
        batch_corners = np.stack(batch_corners, axis=0)
        bbox_corners.append(batch_corners)

    bbox_corners = np.stack(bbox_corners, axis=0) # batch_size, num_proposals, 8, 3

    return bbox_corners

def feed_scene_cap(model, device, dataset, dataloader, phase, folder, 
    use_tf=False, is_eval=True, max_len=CONF.TRAIN.MAX_DES_LEN, save_interm=False, min_iou=CONF.TRAIN.MIN_IOU_THRESHOLD):
    candidates = {}
    intermediates = {}
    for data_dict in tqdm(dataloader):
        # move to cuda
        for key in data_dict:
            data_dict[key] = data_dict[key].cuda()

        with torch.no_grad():
            data_dict = model(data_dict, use_tf=use_tf, is_eval=is_eval)
            data_dict = get_scene_cap_loss(data_dict, device, DC, weights=dataset.weights, detection=True, caption=False)

        # unpack
        captions = data_dict["lang_cap"].argmax(-1) # batch_size, num_proposals, max_len - 1
        dataset_ids = data_dict["dataset_idx"]
        batch_size, num_proposals, _ = captions.shape

        # post-process
        # config
        POST_DICT = {
            "remove_empty_box": True, 
            "use_3d_nms": True, 
            "nms_iou": 0.25,
            "use_old_type_nms": False, 
            "cls_nms": True, 
            "per_class_proposal": True,
            "conf_thresh": 0.05,
            "dataset_config": DC
        }

        # nms mask
        _ = parse_predictions(data_dict, POST_DICT)
        nms_masks = torch.LongTensor(data_dict["pred_mask"]).cuda()

        # objectness mask
        obj_masks = torch.argmax(data_dict["objectness_scores"], 2).long()

        # final mask
        nms_masks = nms_masks * obj_masks

        # pick out object ids of detected objects
        detected_object_ids = torch.gather(data_dict["scene_object_ids"], 1, data_dict["object_assignment"])

        # bbox corners
        assigned_target_bbox_corners = torch.gather(
            data_dict["gt_box_corner_label"], 
            1, 
            data_dict["object_assignment"].view(batch_size, num_proposals, 1, 1).repeat(1, 1, 8, 3)
        ) # batch_size, num_proposals, 8, 3
        detected_bbox_corners = data_dict["bbox_corner"] # batch_size, num_proposals, 8, 3
        detected_bbox_centers = data_dict["center"] # batch_size, num_proposals, 3
        
        # compute IoU between each detected box and each ground truth box
        ious = box3d_iou_batch_tensor(
            assigned_target_bbox_corners.view(-1, 8, 3), # batch_size * num_proposals, 8, 3
            detected_bbox_corners.view(-1, 8, 3) # batch_size * num_proposals, 8, 3
        ).view(batch_size, num_proposals)
        
        # find good boxes (IoU > threshold)
        good_bbox_masks = ious > min_iou # batch_size, num_proposals

        # dump generated captions
        object_attn_masks = {}
        for batch_id in range(batch_size):
            dataset_idx = dataset_ids[batch_id].item()
            scene_id = dataset.scanrefer[dataset_idx]["scene_id"]
            object_attn_masks[scene_id] = np.zeros((num_proposals, num_proposals))
            for prop_id in range(num_proposals):
                if nms_masks[batch_id, prop_id] == 1 and good_bbox_masks[batch_id, prop_id] == 1:
                    object_id = str(detected_object_ids[batch_id, prop_id].item())
                    caption_decoded = decode_caption(captions[batch_id, prop_id], dataset.vocabulary["idx2word"])

                    # print(scene_id, object_id)
                    try:
                        ann_list = list(SCANREFER_ORGANIZED[scene_id][object_id].keys())
                        object_name = SCANREFER_ORGANIZED[scene_id][object_id][ann_list[0]]["object_name"]

                        # store
                        key = "{}|{}|{}".format(scene_id, object_id, object_name)
                        # key = "{}|{}".format(scene_id, object_id)
                        candidates[key] = [caption_decoded]

                        if save_interm:
                            if scene_id not in intermediates: intermediates[scene_id] = {}
                            if object_id not in intermediates[scene_id]: intermediates[scene_id][object_id] = {}

                            intermediates[scene_id][object_id]["object_name"] = object_name
                            intermediates[scene_id][object_id]["box_corner"] = detected_bbox_corners[batch_id, prop_id].cpu().numpy().tolist()
                            intermediates[scene_id][object_id]["description"] = caption_decoded
                            intermediates[scene_id][object_id]["token"] = caption_decoded.split(" ")

                            # attention context
                            # extract attention masks for each object
                            object_attn_weights = data_dict["topdown_attn"][:, :, :num_proposals] # NOTE only consider attention on objects
                            valid_context_masks = data_dict["valid_masks"][:, :, :num_proposals] # NOTE only consider attention on objects

                            cur_valid_context_masks = valid_context_masks[batch_id, prop_id] # num_proposals
                            cur_context_box_corners = detected_bbox_corners[batch_id, cur_valid_context_masks == 1] # X, 8, 3
                            cur_object_attn_weights = object_attn_weights[batch_id, prop_id, cur_valid_context_masks == 1] # X

                            intermediates[scene_id][object_id]["object_attn_weight"] = cur_object_attn_weights.cpu().numpy().T.tolist()
                            intermediates[scene_id][object_id]["object_attn_context"] = cur_context_box_corners.cpu().numpy().tolist()

                        # cache
                        object_attn_masks[scene_id][prop_id, prop_id] = 1
                    except KeyError:
                        continue

    # detected boxes
    if save_interm:
        print("saving intermediate results...")
        interm_path = os.path.join(CONF.PATH.OUTPUT, folder, "interm.json")
        with open(interm_path, "w") as f:
            json.dump(intermediates, f, indent=4)

    return candidates


def update_interm(interm, candidates, bleu, cider, rouge, meteor):
    for i, (key, value) in enumerate(candidates.items()):
        scene_id, object_id, object_name = key.split("|")
        if scene_id in interm:
            if object_id in interm[scene_id]:
                interm[scene_id][object_id]["bleu_1"] = bleu[1][0][i]
                interm[scene_id][object_id]["bleu_2"] = bleu[1][1][i]
                interm[scene_id][object_id]["bleu_3"] = bleu[1][2][i]
                interm[scene_id][object_id]["bleu_4"] = bleu[1][3][i]

                interm[scene_id][object_id]["cider"] = cider[1][i]

                interm[scene_id][object_id]["rouge"] = rouge[1][i]

                interm[scene_id][object_id]["meteor"] = meteor[1][i]

    return interm

def non_max_suppression(ious, scores, threshold):
    ixs = scores.argsort()[::-1]
    pick = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)
        iou = ious[i, ixs[1:]]
        remove_ixs = np.where(iou > threshold)[0] + 1
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
    return np.array(pick, dtype=np.int32)

def eval_cap(model, device, dataset, dataloader, phase, folder, 
    use_tf=False, is_eval=True, max_len=CONF.TRAIN.MAX_DES_LEN, force=False, 
    mode="scene", save_interm=False, no_caption=False, no_classify=False, min_iou=CONF.TRAIN.MIN_IOU_THRESHOLD):
    if no_caption:
        bleu = 0
        cider = 0
        rouge = 0
        meteor = 0

        if no_classify:
            cls_acc = 0
        else:
            print("evaluating classification accuracy...")
            cls_acc = []
            for data_dict in tqdm(dataloader):
                # move to cuda
                for key in data_dict:
                    data_dict[key] = data_dict[key].to(device)

                with torch.no_grad():
                    data_dict = model(data_dict, use_tf=use_tf, is_eval=is_eval)
                
                # unpack
                preds = data_dict["enc_preds"] # (B, num_cls)
                targets = data_dict["object_cat"] # (B,)
                
                # classification acc
                preds = preds.argmax(-1) # (B,)
                acc = (preds == targets).sum().float() / targets.shape[0]
                
                # dump
                cls_acc.append(acc.item())

            cls_acc = np.mean(cls_acc)
    else:
        # corpus
        corpus_path = os.path.join(CONF.PATH.OUTPUT, folder, "corpus_{}.json".format(phase))
        if not os.path.exists(corpus_path) or force:
            print("preparing corpus...")
            corpus = prepare_corpus(dataset.scanrefer, max_len)
            with open(corpus_path, "w") as f:
                json.dump(corpus, f, indent=4)
        else:
            print("loading corpus...")
            with open(corpus_path) as f:
                corpus = json.load(f)

        pred_path = os.path.join(CONF.PATH.OUTPUT, folder, "pred_{}.json".format(phase))
        # if not os.path.exists(pred_path) or force:
        # generate results
        print("generating descriptions...")
        if mode == "scene":
            candidates = feed_scene_cap(model, device, dataset, dataloader, phase, folder, use_tf, is_eval, max_len, save_interm, min_iou)
        elif mode == "object":
            pass
            #candidates, cls_acc = feed_object_cap(model, device, dataset, dataloader, phase, folder, use_tf, is_eval, max_len)
        elif mode == "oracle":
            pass
            #candidates = feed_oracle_cap(model, device, dataset, dataloader, phase, folder, use_tf, is_eval, max_len)
        else:
            raise ValueError("invalid mode: {}".format(mode))

        # check candidates
        # NOTE: make up the captions for the undetected object by "sos eos"
        candidates = check_candidates(corpus, candidates)

        candidates = organize_candidates(corpus, candidates)

        with open(pred_path, "w") as f:
            json.dump(candidates, f, indent=4)

        # compute scores
        print("computing scores...")
        bleu = capblue.Bleu(4).compute_score(corpus, candidates)
        cider = capcider.Cider().compute_score(corpus, candidates)
        rouge = caprouge.Rouge().compute_score(corpus, candidates)
        meteor = capmeteor.Meteor().compute_score(corpus, candidates)


        # update intermediates
        if save_interm:
            print("updating intermediate results...")
            interm_path = os.path.join(CONF.PATH.OUTPUT, folder, "interm.json")
            with open(interm_path) as f:
                interm = json.load(f)

            interm = update_interm(interm, candidates, bleu, cider, rouge, meteor)

            with open(interm_path, "w") as f:
                json.dump(interm, f, indent=4)

    if mode == "scene" or mode == "oracle":
        return bleu, cider, rouge, meteor
    else:
        return bleu, cider, rouge, meteor, cls_acc

def remap_semantic_ids(sematic_ids):
    semantic_label_idx = [3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39]
    remapper = np.ones(40) * (-100)
    for i, x in enumerate(semantic_label_idx):
        remapper[x] = i
    remapped_labels = np.zeros(sematic_ids.shape)
    for i, l in enumerate(sematic_ids):
        remapped_labels[i] = remapper[l]
    return remapped_labels

def feed_pointgroup_cap(model,cfg,epoch,dataset,dataloader,no_detection=False):

    semantic_label_idx = [3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39]
    
    candidates = {}

    with torch.no_grad():
        #model.eval()
        #start_epoch = time.time()
        scene_ids = []
        for data_dict in tqdm(dataloader):
            scene_id = data_dict['scene_id']
            if scene_id not in scene_ids:
                scene_ids.append(scene_id)
            else:
                continue
            #move to cuda 
            for key in data_dict:
                if type(data_dict[key]) == torch.Tensor:
                    data_dict[key] = data_dict[key].cuda()
                else:
                    pass

            ##### prepare input and forward
            data_dict = model(data_dict, epoch, use_tf=False, is_eval=True) 

            loss, loss_dict, visual_dict, meter_dict = get_pointgroup_cap_loss(data_dict,cfg,epoch,
            detection=not no_detection,caption=False)

            if epoch > cfg.prepare_epochs:
                # all cap related actions come here 
                # unpack
                N = data_dict['feats'].shape[0]
                semantic_scores = data_dict['semantic_scores']
                scores, proposals_idx, proposals_offset = data_dict['proposal_scores']
                captions = data_dict["lang_cap"].argmax(-1) # num_proposals, max_len - 1
                dataset_id = data_dict["id"][0]

                semantic_pred = semantic_scores.max(1)[1]
                scores_pred = torch.sigmoid(scores.view(-1))

                proposals_pred = torch.zeros((proposals_offset.shape[0] - 1, N), dtype=torch.int, device=scores_pred.device) # (nProposal, N), int, cuda
                proposals_pred[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1
                
                semantic_id = torch.tensor(semantic_label_idx, device=scores_pred.device)[semantic_pred[proposals_idx[:, 1][proposals_offset[:-1].long()].long()]] # (nProposal), long

                score_mask = (scores_pred > cfg.TEST_SCORE_THRESH)
                proposals_pointnum = proposals_pred.sum(1)
                npoint_mask = (proposals_pointnum > cfg.TEST_NPOINT_THRESH)
                mask = score_mask * npoint_mask

                proposals_pred = proposals_pred[mask]
                semantic_id = semantic_id[mask]
                scores_pred = scores_pred[mask]

                ##### nms
                if semantic_id.shape[0] == 0:
                    pick_idxs = np.empty(0)
                else:
                    proposals_pred_f = proposals_pred.float()  # (nProposal, N), float, cuda
                    intersection = torch.mm(proposals_pred_f, proposals_pred_f.t())  # (nProposal, nProposal), float, cuda
                    proposals_pointnum = proposals_pred_f.sum(1)  # (nProposal), float, cuda
                    proposals_pn_h = proposals_pointnum.unsqueeze(-1).repeat(1, proposals_pointnum.shape[0])
                    proposals_pn_v = proposals_pointnum.unsqueeze(0).repeat(proposals_pointnum.shape[0], 1)
                    cross_ious = intersection / (proposals_pn_h + proposals_pn_v - intersection)
                    pick_idxs = non_max_suppression(cross_ious.cpu().numpy(), scores_pred.cpu().numpy(), cfg.TEST_NMS_THRESH)  # int, (nCluster, N)
                clusters = proposals_pred[pick_idxs]
                cluster_scores = scores_pred[pick_idxs]
                cluster_semantic_id = semantic_id[pick_idxs]


                instance_labels = data_dict['instance_labels']     # (N), long, cuda, 0~total_nInst, -100
                instance_pointnum = data_dict['instance_pointnum'] # (total_nInst), int, cuda
                coords = data_dict['locs']                          # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
                labels = data_dict['labels']                      # (N), long, cuda
                gt_cluster_count = instance_pointnum.shape[0]

                # ious = pointgroup_ops.get_iou(proposals_idx[:, 1].cuda(), proposals_offset.cuda(), instance_labels,
                #                              instance_pointnum)  # (nProposal, nInstance), float
                # gt_ious, gt_instance_idxs = ious.max(1)  # (nProposal) float, long

                remapped_semantic_ids = remap_semantic_ids(cluster_semantic_id)
                pred_bboxes = ap_helper.calculate_pred_bboxes_pointgroup(coords, clusters, remapped_semantic_ids, cluster_scores)[0]
                gt_bboxes = ap_helper.calculate_gt_bboxes_pointgroup(coords, labels, instance_labels, gt_cluster_count)[0]

                # [
                #     semantic_class -> (float),
                #     bbox -> ([center_x, center_y, center_z, length_x, length_y, length_z]),
                #     bbox_score -> (float, [0 - 1])
                # ]
                ious_bbox = torch.zeros(len(pred_bboxes), len(gt_bboxes)).cuda()
                for i, pred_box_info in enumerate(pred_bboxes):
                    for j, gt_box_info in enumerate(gt_bboxes):
                        pred_bbox = np.array(pred_box_info[1])
                        gt_bbox = np.array(gt_box_info[1])
                        ious_bbox[i, j] = get_iou(pred_bbox, gt_bbox)
                gt_ious, gt_instance_idxs = ious_bbox.max(1)  # (nProposal) float, long

                # gt_ious = gt_ious[mask][pick_idxs]
                # gt_instance_idxs = gt_instance_idxs[mask][pick_idxs]
                captions = captions[mask][pick_idxs]

                iou_thresh_mask = gt_ious > 0.5
                gt_ious = gt_ious[iou_thresh_mask]
                gt_instance_idxs = gt_instance_idxs[iou_thresh_mask]
                captions = captions[iou_thresh_mask]
                # dump generated captions
                scene_id = dataset.val_data[dataset_id]["scene_id"]
                for prop_id in range(len(gt_ious)):
                    object_id = str(gt_instance_idxs[prop_id].item())
                    caption_decoded = decode_caption(captions[prop_id], dataset.vocabulary["idx2word"])
                    try:
                        ann_list = list(SCANREFER_ORGANIZED[scene_id][object_id].keys())
                        object_name = SCANREFER_ORGANIZED[scene_id][object_id][ann_list[0]]["object_name"]

                        # store
                        key = "{}|{}|{}".format(scene_id, object_id, object_name)
                        # key = "{}|{}".format(scene_id, object_id)
                        candidates[key] = [caption_decoded]
                    except KeyError:
                        continue
    return candidates, meter_dict, visual_dict

def process_gt2pred(gt2pred):
    gt2predlist = []
    for label, gts in gt2pred.items():
        for gt in gts:
            matched_pred_id = -1 if len(gt['matched_pred']) == 0 else gt['matched_pred'][0]['pred_id'] # select the prediction with the highest confidencec
            gt2predlist.append((gt['instance_id'] % 1000, matched_pred_id))
    return gt2predlist # [(gt_id, pred_id)]

def process_pred2gt(pred2gt):
    pred2gt_dict = {}
    for label, preds in pred2gt.items():
        for pred in preds:
            matched_gt_id = -1 if len(pred['matched_gt']) == 0 else pred['matched_gt'][0]['instance_id'] % 1000 # select the prediction with the highest confidencec
            pred2gt_dict[pred['pred_id']] = matched_gt_id
    return pred2gt_dict # [(pred_id, gt_id)]

def eval_cap_pointgroup(model,cfg,epoch,dataset,dataloader,no_detection=False,no_caption=False,force=True,max_len=CONF.TRAIN.MAX_DES_LEN):
    am_dict = {}

    if no_caption:
        with torch.no_grad():
            #model.eval()
            #start_epoch = time.time()
            for data_dict in tqdm(dataloader):

                #move to cuda 
                for key in data_dict:
                    if type(data_dict[key]) == torch.Tensor:
                        data_dict[key] = data_dict[key].cuda()
                    else:
                        pass

                ##### prepare input and forward
                #is_eval=False not important, captioning_module won't be used
                data_dict = model(data_dict, epoch, use_tf=False, is_eval=False) 

                loss, loss_dict, visual_dict, meter_dict = get_pointgroup_cap_loss(data_dict,cfg,epoch,
                detection=not no_detection,caption=False)

                ##### meter_dict
                for k, v in meter_dict.items():
                    if k not in am_dict.keys():
                        am_dict[k] = utils_pointgroup.AverageMeter()
                    am_dict[k].update(v[0], v[1])
    else:

        candidates, meter_dict, visual_dict = feed_pointgroup_cap(model,cfg,epoch,dataset,dataloader,no_detection)
        ##TODO: equivelent steps of feed_scene_cap()

        if epoch > cfg.prepare_epochs:
            # corpus
            corpus_path = os.path.join(cfg.exp_path, "epoch{}".format(epoch) + "_corpus_val.json")
            if not os.path.exists(corpus_path) or force:
                print("preparing corpus...")
                corpus = prepare_corpus(dataset.val_data, max_len)
                with open(corpus_path, "w") as f:
                    json.dump(corpus, f, indent=4)
            else:
                print("loading corpus...")
                with open(corpus_path) as f:
                    corpus = json.load(f)

            pred_path = os.path.join(cfg.exp_path, "epoch{}".format(epoch) + "_pred_val.json")
            # check candidates
            # NOTE: make up the captions for the undetected object by "sos eos"
            candidates = check_candidates(corpus, candidates)

            candidates = organize_candidates(corpus, candidates)

            with open(pred_path, "w") as f:
                json.dump(candidates, f, indent=4)

            # compute scores
            print("computing scores...")
            bleu = capblue.Bleu(4).compute_score(corpus, candidates)
            cider = capcider.Cider().compute_score(corpus, candidates)
            rouge = caprouge.Rouge().compute_score(corpus, candidates)
            meteor = capmeteor.Meteor().compute_score(corpus, candidates)
            visual_dict['bleu'] = bleu[0][3]
            visual_dict['cider'] = cider[0]
            visual_dict['rouge'] = rouge[0]
            visual_dict['meteor'] = meteor[0]
            if 'bleu' not in am_dict.keys():
                am_dict['bleu'] = utils_pointgroup.AverageMeter()
                am_dict['cider'] = utils_pointgroup.AverageMeter()
                am_dict['rouge'] = utils_pointgroup.AverageMeter()
                am_dict['meteor'] = utils_pointgroup.AverageMeter()
            am_dict['bleu'].update(bleu[0][3], 1)
            am_dict['cider'].update(cider[0], 1)
            am_dict['rouge'].update(rouge[0], 1)
            am_dict['meteor'].update(meteor[0], 1)
            import pdb; pdb.set_trace()
        ##### meter_dict
        for k, v in meter_dict.items():
            if k not in am_dict.keys():
                am_dict[k] = utils_pointgroup.AverageMeter()
            am_dict[k].update(v[0], v[1])

    return am_dict, visual_dict

def get_eval(data_dict, config, caption, use_lang_classifier=False, use_oracle=False, use_cat_rand=False, use_best=False, post_processing=None):
    """ Loss functions

    Args:
        data_dict: dict
        config: dataset config instance
        reference: flag (False/True)
        post_processing: config dict
    Returns:
        loss: pytorch scalar tensor
        data_dict: dict
    """

    objectness_preds_batch = torch.argmax(data_dict['objectness_scores'], 2).long()
    objectness_labels_batch = data_dict['objectness_label'].long()

    if post_processing:
        _ = parse_predictions(data_dict, post_processing)
        nms_masks = torch.LongTensor(data_dict['pred_mask']).cuda()

        # construct valid mask
        pred_masks = (nms_masks * objectness_preds_batch == 1).float()
        label_masks = (objectness_labels_batch == 1).float()
    else:
        # construct valid mask
        pred_masks = (objectness_preds_batch == 1).float()
        label_masks = (objectness_labels_batch == 1).float()

    
    pred_center = data_dict['center'] # (B,K,3)
    pred_heading_class = torch.argmax(data_dict['heading_scores'], -1) # B,num_proposal
    pred_heading_residual = torch.gather(data_dict['heading_residuals'], 2, pred_heading_class.unsqueeze(-1)) # B,num_proposal,1
    pred_heading_class = pred_heading_class # B,num_proposal
    pred_heading_residual = pred_heading_residual.squeeze(2) # B,num_proposal
    pred_size_class = torch.argmax(data_dict['size_scores'], -1) # B,num_proposal
    pred_size_residual = torch.gather(data_dict['size_residuals'], 2, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,3)) # B,num_proposal,1,3
    pred_size_class = pred_size_class
    pred_size_residual = pred_size_residual.squeeze(2) # B,num_proposal,3

    # store
    data_dict["pred_mask"] = pred_masks
    data_dict["label_mask"] = label_masks
    data_dict['pred_center'] = pred_center
    data_dict['pred_heading_class'] = pred_heading_class
    data_dict['pred_heading_residual'] = pred_heading_residual
    data_dict['pred_size_class'] = pred_size_class
    data_dict['pred_size_residual'] = pred_size_residual

    gt_ref = torch.argmax(data_dict["ref_box_label"], 1)
    gt_center = data_dict['center_label'] # (B,MAX_NUM_OBJ,3)
    gt_heading_class = data_dict['heading_class_label'] # B,K2
    gt_heading_residual = data_dict['heading_residual_label'] # B,K2
    gt_size_class = data_dict['size_class_label'] # B,K2
    gt_size_residual = data_dict['size_residual_label'] # B,K2,3

    
    # --------------------------------------------
    # Some other statistics
    obj_pred_val = torch.argmax(data_dict['objectness_scores'], 2) # B,K
    obj_acc = torch.sum((obj_pred_val==data_dict['objectness_label'].long()).float()*data_dict['objectness_mask'])/(torch.sum(data_dict['objectness_mask'])+1e-6)
    data_dict['obj_acc'] = obj_acc
    # detection semantic classification
    sem_cls_label = torch.gather(data_dict['sem_cls_label'], 1, data_dict['object_assignment']) # select (B,K) from (B,K2)
    sem_cls_pred = data_dict['sem_cls_scores'].argmax(-1) # (B,K)
    sem_match = (sem_cls_label == sem_cls_pred).float()
    data_dict["sem_acc"] = (sem_match * data_dict["pred_mask"]).sum() / data_dict["pred_mask"].sum()

    return data_dict
