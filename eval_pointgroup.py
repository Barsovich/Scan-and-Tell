'''
PointGroup test.py
Written by Li Jiang
'''

import torch
import time
import numpy as np
import random
import os

from config.config_pointgroup import cfg
cfg.task = 'test'
from utils.log import logger
import utils.utils_pointgroup as utils
import utils.pointgroup.eval as eval
import lib.ap_helper as ap_helper
from copy import deepcopy
import json
SCANREFER_TRAIN = json.load(open(os.path.join('data', "ScanRefer_filtered_train.json")))
SCANREFER_VAL = json.load(open(os.path.join('data', "ScanRefer_filtered_val.json")))

def remap_semantic_ids(sematic_ids):
    semantic_label_idx = [3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39]
    remapper = np.ones(40) * (-100)
    for i, x in enumerate(semantic_label_idx):
        remapper[x] = i
    remapped_labels = np.zeros(sematic_ids.shape)
    for i, l in enumerate(sematic_ids):
        remapped_labels[i] = remapper[l]
    return remapped_labels

def init():
    global result_dir
    result_dir = os.path.join(cfg.exp_path, 'result', 'epoch{}_nmst{}_scoret{}_npointt{}'.format(cfg.test_epoch, cfg.TEST_NMS_THRESH, cfg.TEST_SCORE_THRESH, cfg.TEST_NPOINT_THRESH), cfg.split)
    backup_dir = os.path.join(result_dir, 'backup_files')
    os.makedirs(backup_dir, exist_ok=True)
    os.makedirs(os.path.join(result_dir, 'predicted_masks'), exist_ok=True)
    os.system('cp eval_pointgroup.py {}'.format(backup_dir))
    os.system('cp {} {}'.format(cfg.model_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.dataset_dir, backup_dir))
    os.system('cp {} {}'.format(cfg.config, backup_dir))

    global semantic_label_idx
    semantic_label_idx = [3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39]
    logger.info(cfg)

    random.seed(cfg.test_seed)
    np.random.seed(cfg.test_seed)
    torch.manual_seed(cfg.test_seed)
    torch.cuda.manual_seed_all(cfg.test_seed)

def get_scannet_scene_list(split):
    scene_list = sorted([line.rstrip() for line in open(os.path.join(cfg.data_root,'meta_data', "scannetv2_{}.txt".format(split)))])

    return scene_list

def get_eval_data(cfg):
    eval_scene_list = get_scannet_scene_list("train") if cfg.use_train else get_scannet_scene_list("val")
    scanrefer_eval = []
    for scene_id in eval_scene_list:
        data = deepcopy(SCANREFER_TRAIN[0]) if args.use_train else deepcopy(SCANREFER_VAL[0])
        data["scene_id"] = scene_id
        scanrefer_eval.append(data)

    print("eval on {} samples".format(len(scanrefer_eval)))

    return scanrefer_eval, eval_scene_list


def test_detection(model, dataloader, epoch, val_scene_list):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')

    # scanrefer_train, scanrefer_val, all_scene_list = get_scanrefer(SCANREFER_TRAIN, SCANREFER_VAL, -1)
    # scanrefer = {
    #     "train": scanrefer_train,
    #     "val": scanrefer_val
    # }

    # if cfg.dataset == 'scannet_data':
    #     if data_name == 'scannet':
    #         from data.dataset_pointgroup import Dataset
    #         dataset = Dataset(scanrefer=scanrefer)
    #         dataset.valLoader()
    #     else:
    #         print("Error: no data loader - " + data_name)
    #         exit(0)
    # dataloader = dataset.val_data_loader

    with torch.no_grad():
        model = model.eval()
        start = time.time()

        matches = {}

        AP_IOU_THRESHOLDS = [0.25, 0.5]
        AP_CALCULATOR_LIST = [ap_helper.APCalculator(iou_thresh, point_group=True) for iou_thresh in AP_IOU_THRESHOLDS]

        for i, data_dict in enumerate(dataloader):

            #move to cuda 
            for key in data_dict:
                if type(data_dict[key]) == torch.Tensor:
                    data_dict[key] = data_dict[key].cuda()
                else:
                    pass

            N = data_dict['feats'].shape[0]
            #test_scene_name = dataset.val_file_names[int(batch['id'][0])].split('/')[-1][:12]
            test_scene_name = val_scene_list[data_dict['id'][0]]
            start1 = time.time()
            data_dict = model(data_dict, epoch, is_eval=True)
            end1 = time.time() - start1

            ##### get predictions (#1 semantic_pred, pt_offsets; #2 scores, proposals_pred)
            semantic_scores = data_dict['semantic_scores']  # (N, nClass=20) float32, cuda
            semantic_pred = semantic_scores.max(1)[1]  # (N) long, cuda

            pt_offsets = data_dict['pt_offsets']    # (N, 3), float32, cuda

            if (epoch > cfg.prepare_epochs):
                scores, proposals_idx, proposals_offset = data_dict['proposal_scores'] 
                #scores = preds['score']   # (nProposal, 1) float, cuda
                scores_pred = torch.sigmoid(scores.view(-1))

                #proposals_idx, proposals_offset = preds['proposals']
                # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
                # proposals_offset: (nProposal + 1), int, cpu
                proposals_pred = torch.zeros((proposals_offset.shape[0] - 1, N), dtype=torch.int, device=scores_pred.device) # (nProposal, N), int, cuda
                proposals_pred[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1

                semantic_id = torch.tensor(semantic_label_idx, device=scores_pred.device)[semantic_pred[proposals_idx[:, 1][proposals_offset[:-1].long()].long()]] # (nProposal), long

                ##### score threshold
                score_mask = (scores_pred > cfg.TEST_SCORE_THRESH)
                scores_pred = scores_pred[score_mask]
                proposals_pred = proposals_pred[score_mask]
                semantic_id = semantic_id[score_mask]

                ##### npoint threshold
                proposals_pointnum = proposals_pred.sum(1)
                npoint_mask = (proposals_pointnum > cfg.TEST_NPOINT_THRESH)
                scores_pred = scores_pred[npoint_mask]
                proposals_pred = proposals_pred[npoint_mask]
                semantic_id = semantic_id[npoint_mask]

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

                nclusters = clusters.shape[0]

                coords = data_dict['locs']                          # (N, 1 + 3), long, cuda, dimension 0 for batch_idx
                instance_labels = data_dict['instance_labels']    # (N), long, cuda, 0~total_nInst, -100
                labels = data_dict['labels']                      # (N), long, cuda
                instance_pointnum = data_dict['instance_pointnum']  # (total_nInst), int, cuda
                gt_cluster_count = instance_pointnum.shape[0]
                
                # calculate AP
                remapped_semantic_ids = remap_semantic_ids(cluster_semantic_id)
                pred_bboxes = ap_helper.calculate_pred_bboxes_pointgroup(coords, clusters, remapped_semantic_ids, cluster_scores)
                gt_bboxes = ap_helper.calculate_gt_bboxes_pointgroup(coords, labels, instance_labels, gt_cluster_count)

                for ap_calculator in AP_CALCULATOR_LIST:
                    ap_calculator.step(pred_bboxes, gt_bboxes)

                ##### prepare for evaluation
                if cfg.eval:
                    pred_info = {}
                    pred_info['conf'] = cluster_scores.cpu().numpy()
                    pred_info['label_id'] = cluster_semantic_id.cpu().numpy()
                    pred_info['mask'] = clusters.cpu().numpy()
                    gt_file = os.path.join(cfg.data_root, cfg.dataset, cfg.split + '_gt', test_scene_name + '.txt')
                    gt2pred, pred2gt = eval.assign_instances_for_scan(test_scene_name, pred_info, gt_file)
                    matches[test_scene_name] = {}
                    matches[test_scene_name]['gt'] = gt2pred
                    matches[test_scene_name]['pred'] = pred2gt


            ##### save files
            start3 = time.time()
            if cfg.save_semantic:
                os.makedirs(os.path.join(result_dir, 'semantic'), exist_ok=True)
                semantic_np = semantic_pred.cpu().numpy()
                np.save(os.path.join(result_dir, 'semantic', test_scene_name + '.npy'), semantic_np)

            if cfg.save_pt_offsets:
                os.makedirs(os.path.join(result_dir, 'coords_offsets'), exist_ok=True)
                pt_offsets_np = pt_offsets.cpu().numpy()
                coords_np = data_dict['locs_float'].numpy()
                coords_offsets = np.concatenate((coords_np, pt_offsets_np), 1)   # (N, 6)
                np.save(os.path.join(result_dir, 'coords_offsets', test_scene_name + '.npy'), coords_offsets)


            if(epoch > cfg.prepare_epochs and cfg.save_instance):
                f = open(os.path.join(result_dir, test_scene_name + '.txt'), 'w')
                for proposal_id in range(nclusters):
                    clusters_i = clusters[proposal_id].cpu().numpy()  # (N)
                    semantic_label = np.argmax(np.bincount(semantic_pred[np.where(clusters_i == 1)[0]].cpu()))
                    score = cluster_scores[proposal_id]
                    f.write('predicted_masks/{}_{:03d}.txt {} {:.4f}'.format(test_scene_name, proposal_id, semantic_label_idx[semantic_label], score))
                    if proposal_id < nclusters - 1:
                        f.write('\n')
                    np.savetxt(os.path.join(result_dir, 'predicted_masks', test_scene_name + '_%03d.txt' % (proposal_id)), clusters_i, fmt='%d')
                f.close()
            end3 = time.time() - start3
            end = time.time() - start
            start = time.time()

            ##### print
            logger.info("instance iter: {}/{} point_num: {} ncluster: {} time: total {:.2f}s inference {:.2f}s save {:.2f}s".format(data_dict['id'][0] + 1, len(dataset.val_file_names), N, nclusters, end, end1, end3))

        ##### evaluation
        if cfg.eval:
            ap_scores = eval.evaluate_matches(matches)
            avgs = eval.compute_averages(ap_scores)
            # report bounding box mAP and recall
            for i, ap_calculator in enumerate(AP_CALCULATOR_LIST):
                print()
                print("-" * 10, "iou_thresh: %f" % (AP_IOU_THRESHOLDS[i]), "-" * 10)
                metrics_dict = ap_calculator.compute_metrics()
                for key in metrics_dict:
                    print("eval %s: %f" % (key, metrics_dict[key]))
            eval.print_results(avgs)


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

def test_caption(model, dataloader, epoch, val_scene_list):

    bleu,cider,rouge,meteor = eval_cap_pointgroup(model,cfg,epoch,dataset,val_loader,
        no_detection=False,no_caption=False,force=True,min_iou=cfg.TEST_MIN_IOU,task='eval')

    # report
    print("\n----------------------Evaluation-----------------------")
    print("[BLEU-1] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[0][0], max(bleu[1][0]), min(bleu[1][0])))
    print("[BLEU-2] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[0][1], max(bleu[1][1]), min(bleu[1][1])))
    print("[BLEU-3] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[0][2], max(bleu[1][2]), min(bleu[1][2])))
    print("[BLEU-4] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[0][3], max(bleu[1][3]), min(bleu[1][3])))
    print("[CIDEr] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(cider[0], max(cider[1]), min(cider[1])))
    print("[ROUGE-L] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(rouge[0], max(rouge[1]), min(rouge[1])))
    print("[METEOR] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(meteor[0], max(meteor[1]), min(meteor[1])))
    print()


if __name__ == '__main__':
    init()

    ##### get model version and data version
    exp_name = cfg.config.split('/')[-1][:-5]
    model_name = exp_name.split('_')[0]
    data_name = exp_name.split('_')[-1]

    ##### model
    logger.info('=> creating model ...')
    logger.info('Classes: {}'.format(cfg.classes))

    if model_name == 'pointgroup':
        from models.capnet import CapNet
        #from models.pointgroup import PointGroup as Network
        #from models.pointgroup import model_fn_decorator
    else:
        print("Error: no model - " + model_name)
        exit(0)

    if cfg.dataset == 'scannet_data':
        if data_name == 'scannet':
            scanrefer_val, val_scene_list = get_eval_data(cfg)
            scanrefer = {
                "train": [],
                "val": scanrefer_val
            }
            import data.dataset_pointgroup
            dataset = data.dataset_pointgroup.Dataset(scanrefer)
            #dataset.trainLoader()
            dataset.valLoader()
        else:
            print("Error: no data loader - " + data_name)
            exit(0)

    dataloader = dataset.val_data_loader
    vocabulary = dataset.vocabulary
    embeddings = dataset.glove
    #val_scene_list = all_scene_list[-312:]

    model = CapNet(vocabulary, embeddings, cfg, 'pointgroup',no_caption=cfg.no_caption,prepare_epochs=cfg.prepare_epochs)

    use_cuda = torch.cuda.is_available()
    logger.info('cuda available: {}'.format(use_cuda))
    assert use_cuda
    model = model.cuda()

    # logger.info(model)
    logger.info('#classifier parameters (model): {}'.format(sum([x.nelement() for x in model.parameters()])))

    ##### model_fn (criterion)
    #model_fn = model_fn_decorator(test=True)

    ##### load model
    utils.checkpoint_restore(model, cfg.exp_path, cfg.config.split('/')[-1][:-5], use_cuda, cfg.test_epoch, strict=False, dist=False, f=cfg.pretrain)      # resume from the latest epoch, or specify the epoch to restore

    ##### evaluate
    if cfg.no_caption: 
        test_detection(model, dataloader, cfg.test_epoch, val_scene_list)
    else:
        test_caption(model,dataloader,cfg.test_epoch,val_scene_list)


