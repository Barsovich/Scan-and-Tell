'''
Generate instance groundtruth .txt files (for evaluation)
'''

import numpy as np
import glob
import torch
import os
import sys
sys.path.append('../../') # HACK add the root folder

semantic_label_idxs = [3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39]
semantic_label_names = ['cabinet','bed','chair','sofa','table','door','window','bookshelf','picture','counter','desk','curtain','refrigerator','shower curtain','toilet','sink','bathtub','others']

if __name__ == '__main__':
    split = 'val'
    dataset = 'scannet_data'
    test_scan_names = sorted([line.rstrip() for line in open(os.path.join('meta_data/scannetv2_val.txt'))])
    files = list(map(lambda name: os.path.join(dataset,'{}_pointgroup.pth'.format(name)), test_scan_names))
    rooms = [torch.load(i) for i in files]

    if not os.path.exists(os.path.join(dataset, split + '_gt')):
        os.mkdir(os.path.join(dataset, split + '_gt'))

    for i in range(len(rooms)):
        xyz, rgb, label, instance_label = rooms[i]   # label 0~19 -100;  instance_label 0~instance_num-1 -100
        scene_name = files[i].split('/')[-1][:12]
        print('{}/{} {}'.format(i + 1, len(rooms), scene_name))

        instance_label_new = np.zeros(instance_label.shape, dtype=np.int32)  # 0 for unannotated, xx00y: x for semantic_label, y for inst_id (1~instance_num)

        instance_num = int(instance_label.max()) + 1
        for inst_id in range(instance_num):
            instance_mask = np.where(instance_label == inst_id)[0]
            sem_id = int(label[instance_mask[0]])
            if(sem_id == -100): sem_id = 0
            semantic_label = semantic_label_idxs[sem_id]
            instance_label_new[instance_mask] = semantic_label * 1000 + inst_id + 1

        np.savetxt(os.path.join(dataset, split + '_gt', scene_name + '.txt'), instance_label_new, fmt='%d')