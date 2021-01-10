import os, sys, glob, math, numpy as np
import scipy.ndimage
import scipy.interpolate
import torch
import h5py
import pickle
import json
import multiprocessing as mp

from itertools import chain
from collections import Counter
from torch.utils.data import DataLoader

sys.path.append('../')

from config.config_pointgroup import cfg
from utils.log import logger
from lib.pointgroup_ops.functions import pointgroup_ops

GLOVE_PICKLE = os.path.join('data', "glove.p")
SCANREFER_VOCAB = os.path.join('data', "ScanRefer_vocabulary.json")
SCANREFER_VOCAB_WEIGHTS = os.path.join('data', "ScanRefer_vocabulary_weights.json")


class Dataset:
    def __init__(self,scanrefer,test=False):
        self.data_root = cfg.data_root
        self.dataset = cfg.dataset
        self.filename_suffix = cfg.filename_suffix

        self.train_data = scanrefer['train'] #list of dictionaries, len: 36665 if caption (total number of scanrefer train objects)
        self.val_data = scanrefer['val'] #list of dictionaries
        self.scanrefer = scanrefer['train'] + scanrefer['val']

        self.batch_size = cfg.batch_size
        self.train_workers = cfg.train_workers
        self.val_workers = cfg.train_workers

        self.full_scale = cfg.full_scale
        self.scale = cfg.scale
        self.max_npoint = cfg.max_npoint
        self.mode = cfg.mode

        self.use_multiview = cfg.use_multiview
        self.multiview_data = {}

        if test:
            self.test_split = cfg.split  # val or test
            self.test_workers = cfg.test_workers
            cfg.batch_size = 1

        # load language features
        self.glove = pickle.load(open(GLOVE_PICKLE, "rb"))
        self._build_vocabulary()
        self.num_vocabs = len(self.vocabulary["word2idx"].keys())
        self.lang, self.lang_ids = self._tranform_des()
        self._build_frequency()


    def trainLoader(self):
        train_file_names = sorted(list(set([data["scene_id"] for data in self.train_data])))
        #self.train_file_names = list(map(lambda name: os.path.join(self.data_root,self.dataset,'{}_pointgroup.pth'.format(data)),train_file_names))
        #train_file_names = sorted(glob.glob(os.path.join(self.data_root, self.dataset, 'train', '*' + self.filename_suffix)))
        #self.train_files = [torch.load(i) for i in train_file_names]

        logger.info('Training on {} object samples from {} scenes.'.format(len(self.train_data),len(train_file_names)))

        train_set = list(range(len(self.train_data)))
        self.train_data_loader = DataLoader(train_set, batch_size=self.batch_size, collate_fn=self.trainMerge, num_workers=self.train_workers,
                                            shuffle=True, sampler=None, drop_last=True, pin_memory=True)


    def valLoader(self):
        val_file_names = sorted(list(set([data["scene_id"] for data in self.val_data])))
        #self.val_file_names = list(map(lambda data: os.path.join(self.data_root,self.dataset,'{}_pointgroup.pth'.format(data[' scene_id'])),val_file_names))
        #val_file_names = sorted(glob.glob(os.path.join(self.data_root, self.dataset, 'val', '*' + self.filename_suffix)))
        #self.val_files = [torch.load(i) for i in val_file_names]

        logger.info('Validation on {} object samples from {} scenes.'.format(len(self.val_data),len(val_file_names)))

        val_set = list(range(len(self.val_data)))
        self.val_data_loader = DataLoader(val_set, batch_size=self.batch_size, collate_fn=self.valMerge, num_workers=self.val_workers,
                                          shuffle=False, drop_last=False, pin_memory=True)


    def testLoader(self):
        self.test_file_names = sorted([line.rstrip() for line in open(os.path.join(self.data_root,'meta_data/scannetv2_test.txt'))])
        self.test_file_names = list(map(lambda name: os.path.join(self.data_root,self.dataset,'{}_pointgroup.pth'.format(name)),test_file_names))
        #self.test_file_names = sorted(glob.glob(os.path.join(self.data_root, self.dataset, self.test_split, '*' + self.filename_suffix)))
        self.test_files = [torch.load(i) for i in self.test_file_names]

        logger.info('Testing samples ({}): {}'.format(self.test_split, len(self.test_files)))

        test_set = list(np.arange(len(self.test_files)))
        self.test_data_loader = DataLoader(test_set, batch_size=1, collate_fn=self.testMerge, num_workers=self.test_workers,
                                           shuffle=False, drop_last=False, pin_memory=True)

    #Elastic distortion
    def elastic(self, x, gran, mag):
        blur0 = np.ones((3, 1, 1)).astype('float32') / 3
        blur1 = np.ones((1, 3, 1)).astype('float32') / 3
        blur2 = np.ones((1, 1, 3)).astype('float32') / 3

        bb = np.abs(x).max(0).astype(np.int32)//gran + 3
        noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
        noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
        ax = [np.linspace(-(b-1)*gran, (b-1)*gran, b) for b in bb]
        interp = [scipy.interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]
        def g(x_):
            return np.hstack([i(x_)[:,None] for i in interp])
        return x + g(x) * mag


    def getInstanceInfo(self, xyz, instance_label):
        '''
        :param xyz: (n, 3)
        :param instance_label: (n), int, (0~nInst-1, -100)
        :return: instance_num, dict
        '''
        instance_info = np.ones((xyz.shape[0], 9), dtype=np.float32) * -100.0   # (n, 9), float, (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
        instance_pointnum = []   # (nInst), int
        instance_num = int(instance_label.max()) + 1
        for i_ in range(instance_num):
            inst_idx_i = np.where(instance_label == i_)

            ### instance_info
            xyz_i = xyz[inst_idx_i]
            min_xyz_i = xyz_i.min(0)
            max_xyz_i = xyz_i.max(0)
            mean_xyz_i = xyz_i.mean(0)
            instance_info_i = instance_info[inst_idx_i]
            instance_info_i[:, 0:3] = mean_xyz_i
            instance_info_i[:, 3:6] = min_xyz_i
            instance_info_i[:, 6:9] = max_xyz_i
            instance_info[inst_idx_i] = instance_info_i

            ### instance_pointnum
            instance_pointnum.append(inst_idx_i[0].size)

        return instance_num, {"instance_info": instance_info, "instance_pointnum": instance_pointnum}


    def dataAugment(self, xyz, jitter=False, flip=False, rot=False):
        m = np.eye(3)
        if jitter:
            m += np.random.randn(3, 3) * 0.1
        if flip:
            m[0][0] *= np.random.randint(0, 2) * 2 - 1  # flip x randomly
        if rot:
            theta = np.random.rand() * 2 * math.pi
            m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0], [-math.sin(theta), math.cos(theta), 0], [0, 0, 1]])  # rotation
        return np.matmul(xyz, m)


    def crop(self, xyz):
        '''
        :param xyz: (n, 3) >= 0
        '''
        xyz_offset = xyz.copy()
        valid_idxs = (xyz_offset.min(1) >= 0)
        assert valid_idxs.sum() == xyz.shape[0]

        full_scale = np.array([self.full_scale[1]] * 3)
        room_range = xyz.max(0) - xyz.min(0)
        while (valid_idxs.sum() > self.max_npoint):
            offset = np.clip(full_scale - room_range + 0.001, None, 0) * np.random.rand(3)
            xyz_offset = xyz + offset
            valid_idxs = (xyz_offset.min(1) >= 0) * ((xyz_offset < full_scale).sum(1) == 3)
            full_scale[:2] -= 32

        return xyz_offset, valid_idxs


    def getCroppedInstLabel(self, instance_label, valid_idxs):
        instance_label = instance_label[valid_idxs]
        j = 0
        while (j < instance_label.max()):
            if (len(np.where(instance_label == j)[0]) == 0):
                instance_label[instance_label == instance_label.max()] = j
            j += 1
        return instance_label


    def _tranform_des(self):
        lang = {}
        label = {}
        for data in self.scanrefer:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            ann_id = data["ann_id"]

            if scene_id not in lang:
                lang[scene_id] = {}
                label[scene_id] = {}

            if object_id not in lang[scene_id]:
                lang[scene_id][object_id] = {}
                label[scene_id][object_id] = {}

            if ann_id not in lang[scene_id][object_id]:
                lang[scene_id][object_id][ann_id] = {}
                label[scene_id][object_id][ann_id] = {}

            # trim long descriptions
            tokens = data["token"][:cfg.TRAIN_MAX_DES_LEN]

            # tokenize the description
            tokens = ["sos"] + tokens + ["eos"]
            embeddings = np.zeros((cfg.TRAIN_MAX_DES_LEN + 2, 300))
            labels = np.zeros((cfg.TRAIN_MAX_DES_LEN + 2)) # start and end

            # load
            for token_id in range(len(tokens)):
                token = tokens[token_id]
                try:
                    embeddings[token_id] = self.glove[token]
                    labels[token_id] = self.vocabulary["word2idx"][token]
                except KeyError:
                    embeddings[token_id] = self.glove["unk"]
                    labels[token_id] = self.vocabulary["word2idx"]["unk"]
            
            # store
            lang[scene_id][object_id][ann_id] = embeddings
            label[scene_id][object_id][ann_id] = labels

        return lang, label
    
    def _build_vocabulary(self):
        if os.path.exists(SCANREFER_VOCAB):
            self.vocabulary = json.load(open(SCANREFER_VOCAB))
        else:
            all_words = chain(*[data["token"][:cfg.TRAIN_MAX_DES_LEN] for data in self.train_data])
            word_counter = Counter(all_words)
            word_counter = sorted([(k, v) for k, v in word_counter.items() if k in self.glove], key=lambda x: x[1], reverse=True)
            word_list = [k for k, _ in word_counter]

            # build vocabulary
            word2idx, idx2word = {}, {}
            spw = ["pad_", "unk", "sos", "eos"] # NOTE distinguish padding token "pad_" and the actual word "pad"
            for i, w in enumerate(word_list):
                shifted_i = i + len(spw)
                word2idx[w] = shifted_i
                idx2word[shifted_i] = w

            # add special words into vocabulary
            for i, w in enumerate(spw):
                word2idx[w] = i
                idx2word[i] = w

            vocab = {
                "word2idx": word2idx,
                "idx2word": idx2word
            }
            json.dump(vocab, open(SCANREFER_VOCAB, "w"), indent=4)

            self.vocabulary = vocab

    def _build_frequency(self):
        if os.path.exists(SCANREFER_VOCAB_WEIGHTS):
            with open(SCANREFER_VOCAB_WEIGHTS) as f:
                weights = json.load(f)
                self.weights = np.array([v for _, v in weights.items()])
        else:
            all_tokens = []
            for scene_id in self.lang_ids.keys():
                for object_id in self.lang_ids[scene_id].keys():
                    for ann_id in self.lang_ids[scene_id][object_id].keys():
                        all_tokens += self.lang_ids[scene_id][object_id][ann_id].astype(int).tolist()

            word_count = Counter(all_tokens)
            word_count = sorted([(k, v) for k, v in word_count.items()], key=lambda x: x[0])
            
            # frequencies = [c for _, c in word_count]
            # weights = np.array(frequencies).astype(float)
            # weights = weights / np.sum(weights)
            # weights = 1 / np.log(1.05 + weights)

            weights = np.ones((len(word_count)))

            self.weights = weights
            
            with open(SCANREFER_VOCAB_WEIGHTS, "w") as f:
                weights = {k: v for k, v in enumerate(weights)}
                json.dump(weights, f, indent=4)


    def trainMerge(self, id): 
        locs = []
        locs_float = []
        feats = []
        labels = []
        instance_labels = []

        instance_infos = []  # (N, 9)
        instance_pointnum = []  # (total_nInst), int

        lang_feats = []
        lang_lens = []
        lang_ids = []

        ann_ids = []
        object_ids = []
        object_classes = []

        batch_offsets = [0]

        total_inst_num = 0
        for i, idx in enumerate(id):

            #get object 
            scene_id = self.train_data[idx]["scene_id"]
            object_id = int(self.train_data[idx]["object_id"])
            object_name = " ".join(self.train_data[idx]["object_name"].split("_"))
            ann_id = self.train_data[idx]["ann_id"]

            #get language features
            lang_feat = self.lang[scene_id][str(object_id)][ann_id]
            lang_len = len(self.scanrefer[idx]["token"]) + 2
            lang_len = lang_len if lang_len <= cfg.TRAIN_MAX_DES_LEN + 2 else cfg.TRAIN_MAX_DES_LEN + 2

            #get scene data
            data_file = os.path.join(self.data_root,self.dataset,'{}_pointgroup.pth'.format(scene_id))
            xyz_origin, rgb, label, instance_label = torch.load(data_file)

            ### jitter / flip x / rotation
            xyz_middle = self.dataAugment(xyz_origin, True, True, True)

            ### scale
            xyz = xyz_middle * self.scale

            ### elastic
            xyz = self.elastic(xyz, 6 * self.scale // 50, 40 * self.scale / 50)
            xyz = self.elastic(xyz, 20 * self.scale // 50, 160 * self.scale / 50)

            ### offset
            xyz -= xyz.min(0)

            ### crop
            xyz, valid_idxs = self.crop(xyz)

            xyz_middle = xyz_middle[valid_idxs]
            xyz = xyz[valid_idxs]
            rgb = rgb[valid_idxs]
            label = label[valid_idxs]
            instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)

            ### get instance information
            inst_num, inst_infos = self.getInstanceInfo(xyz_middle, instance_label.astype(np.int32))
            inst_info = inst_infos["instance_info"]  # (n, 9), (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
            inst_pointnum = inst_infos["instance_pointnum"]   # (nInst), list

            instance_label[np.where(instance_label != -100)] += total_inst_num
            total_inst_num += inst_num

            ### merge the scene to the batch
            ann_ids.append(int(ann_id))
            object_ids.append(int(object_id))
            batch_offsets.append(batch_offsets[-1] + xyz.shape[0])

            locs.append(torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(i), torch.from_numpy(xyz).long()], 1))
            locs_float.append(torch.from_numpy(xyz_middle))
            
            feat = torch.from_numpy(rgb) + torch.randn(3) * 0.1
            if self.use_multiview:
                pid = mp.current_process().pid
                if pid not in self.multiview_data:
                    self.multiview_data[pid] = h5py.File(os.path.join(self.data_root,self.dataset,'enet_feats_maxpool.hdf5'), "r", libver="latest")

                multiview = torch.from_numpy(self.multiview_data[pid][scene_id][valid_idxs])
                feat = torch.cat([feat,multiview],1)

            feats.append(feat)

            labels.append(torch.from_numpy(label))
            instance_labels.append(torch.from_numpy(instance_label))

            instance_infos.append(torch.from_numpy(inst_info))
            instance_pointnum.extend(inst_pointnum)

            lang_feats.append(torch.from_numpy(lang_feat).unsqueeze(0))
            lang_lens.append(lang_len)
            lang_ids.append(torch.from_numpy(self.lang_ids[scene_id][str(object_id)][ann_id]).unsqueeze(0))

        ### merge all the scenes in the batchd
        ann_ids = torch.tensor(ann_ids, dtype=torch.long)
        object_ids = torch.tensor(object_ids, dtype=torch.long)
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)

        locs = torch.cat(locs, 0)                                # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
        locs_float = torch.cat(locs_float, 0).to(torch.float32)  # float (N, 3)
        feats = torch.cat(feats, 0)                              # float (N, C)

        labels = torch.cat(labels, 0).long()                     # long (N)
        instance_labels = torch.cat(instance_labels, 0).long()   # long (N)

        instance_infos = torch.cat(instance_infos, 0).to(torch.float32)       # float (N, 9) (meanxyz, minxyz, maxxyz)
        instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int)  # int (total_nInst)

        lang_feats = torch.cat(lang_feats,0).to(torch.float32)
        lang_lens = torch.tensor(lang_lens,dtype=torch.long)
        lang_ids = torch.cat(lang_ids,0).to(torch.long)

        spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)     # long (3)

        ### voxelize
        voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, self.batch_size, self.mode)

        return {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
                'locs_float': locs_float, 'feats': feats, 'labels': labels, 'instance_labels': instance_labels,
                'instance_info': instance_infos, 'instance_pointnum': instance_pointnum,
                'id': id, 'offsets': batch_offsets, 'spatial_shape': spatial_shape, 
                'lang_feat': lang_feats, 'lang_len': lang_lens, 'lang_ids': lang_ids, 
                'ann_id': ann_ids, 'object_id': object_ids }


    def valMerge(self, id):
        locs = []
        locs_float = []
        feats = []
        labels = []
        instance_labels = []

        instance_infos = []  # (N, 9)
        instance_pointnum = []  # (total_nInst), int

        lang_feats = []
        lang_lens = []
        lang_ids = []

        ann_ids = []
        object_ids = []
        object_classes = []

        batch_offsets = [0]

        total_inst_num = 0
        for i, idx in enumerate(id):

            #get object 
            scene_id = self.train_data[idx]["scene_id"]
            object_id = int(self.train_data[idx]["object_id"])
            object_name = " ".join(self.train_data[idx]["object_name"].split("_"))
            ann_id = self.train_data[idx]["ann_id"]

            #get language features
            lang_feat = self.lang[scene_id][str(object_id)][ann_id]
            lang_len = len(self.scanrefer[idx]["token"]) + 2
            lang_len = lang_len if lang_len <= cfg.TRAIN_MAX_DES_LEN + 2 else cfg.TRAIN_MAX_DES_LEN + 2

            #get scene data
            data_file = os.path.join(self.data_root,self.dataset,'{}_pointgroup.pth'.format(scene_id))
            xyz_origin, rgb, label, instance_label = torch.load(data_file)

            ### flip x / rotation
            xyz_middle = self.dataAugment(xyz_origin, False, True, True)

            ### scale
            xyz = xyz_middle * self.scale

            ### offset
            xyz -= xyz.min(0)

            ### crop
            xyz, valid_idxs = self.crop(xyz)

            xyz_middle = xyz_middle[valid_idxs]
            xyz = xyz[valid_idxs]
            rgb = rgb[valid_idxs]
            label = label[valid_idxs]
            instance_label = self.getCroppedInstLabel(instance_label, valid_idxs)

            ### get instance information
            inst_num, inst_infos = self.getInstanceInfo(xyz_middle, instance_label.astype(np.int32))
            inst_info = inst_infos["instance_info"]  # (n, 9), (cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
            inst_pointnum = inst_infos["instance_pointnum"]  # (nInst), list

            instance_label[np.where(instance_label != -100)] += total_inst_num
            total_inst_num += inst_num

            ### merge the scene to the batch
            ann_ids.append(int(ann_id))
            object_ids.append(int(object_id))
            batch_offsets.append(batch_offsets[-1] + xyz.shape[0])

            locs.append(torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(i), torch.from_numpy(xyz).long()], 1))
            locs_float.append(torch.from_numpy(xyz_middle))

            feat = torch.from_numpy(rgb) 
            if self.use_multiview:
                pid = mp.current_process().pid
                if pid not in self.multiview_data:
                    self.multiview_data[pid] = h5py.File(os.path.join(self.data_root,self.dataset,'enet_feats_maxpool.hdf5'), "r", libver="latest")

                multiview = torch.from_numpy(self.multiview_data[pid][scene_id][valid_idxs])
                feat = torch.cat([feat,multiview],1)

            feats.append(feat)
            
            labels.append(torch.from_numpy(label))
            instance_labels.append(torch.from_numpy(instance_label))

            instance_infos.append(torch.from_numpy(inst_info))
            instance_pointnum.extend(inst_pointnum)

            lang_feats.append(torch.from_numpy(lang_feat).unsqueeze(0))
            lang_lens.append(lang_len)
            lang_ids.append(torch.from_numpy(self.lang_ids[scene_id][str(object_id)][ann_id]).unsqueeze(0))


        ### merge all the scenes in the batch
        ann_ids = torch.tensor(ann_ids, dtype=torch.long)
        object_ids = torch.tensor(object_ids, dtype=torch.long)
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)

        locs = torch.cat(locs, 0)                                  # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
        locs_float = torch.cat(locs_float, 0).to(torch.float32)    # float (N, 3)
        feats = torch.cat(feats, 0)                                # float (N, C)
        labels = torch.cat(labels, 0).long()                       # long (N)
        instance_labels = torch.cat(instance_labels, 0).long()     # long (N)

        instance_infos = torch.cat(instance_infos, 0).to(torch.float32)               # float (N, 9) (meanxyz, minxyz, maxxyz)
        instance_pointnum = torch.tensor(instance_pointnum, dtype=torch.int)          # int (total_nInst)

        lang_feats = torch.cat(lang_feats,0).to(torch.float32)
        lang_lens = torch.tensor(lang_lens,dtype=torch.long)
        lang_ids = torch.cat(lang_ids,0).to(torch.long)

        spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)  # long (3)

        ### voxelize
        voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, self.batch_size, self.mode)

        return {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
                'locs_float': locs_float, 'feats': feats, 'labels': labels, 'instance_labels': instance_labels,
                'instance_info': instance_infos, 'instance_pointnum': instance_pointnum,
                'id': id, 'offsets': batch_offsets, 'spatial_shape': spatial_shape, 
                'lang_feat': lang_feats, 'lang_len': lang_lens, 'lang_ids': lang_ids, 
                'ann_id': ann_ids, 'object_id': object_ids }


    def testMerge(self, id):
        locs = []
        locs_float = []
        feats = []

        batch_offsets = [0]

        for i, idx in enumerate(id):
            if self.test_split == 'val':
                xyz_origin, rgb, label, instance_label = self.test_files[idx]
            elif self.test_split == 'test':
                xyz_origin, rgb = self.test_files[idx]
            else:
                print("Wrong test split: {}!".format(self.test_split))
                exit(0)

            ### flip x / rotation
            xyz_middle = self.dataAugment(xyz_origin, False, True, True)

            ### scale
            xyz = xyz_middle * self.scale

            ### offset
            xyz -= xyz.min(0)

            ### merge the scene to the batch
            batch_offsets.append(batch_offsets[-1] + xyz.shape[0])

            locs.append(torch.cat([torch.LongTensor(xyz.shape[0], 1).fill_(i), torch.from_numpy(xyz).long()], 1))
            locs_float.append(torch.from_numpy(xyz_middle))
            feats.append(torch.from_numpy(rgb))

        ### merge all the scenes in the batch
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)

        locs = torch.cat(locs, 0)                                         # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
        locs_float = torch.cat(locs_float, 0).to(torch.float32)           # float (N, 3)
        feats = torch.cat(feats, 0)                                       # float (N, C)

        spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)  # long (3)

        ### voxelize
        voxel_locs, p2v_map, v2p_map = pointgroup_ops.voxelization_idx(locs, self.batch_size, self.mode)

        return {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
                'locs_float': locs_float, 'feats': feats,
                'id': id, 'offsets': batch_offsets, 'spatial_shape': spatial_shape}