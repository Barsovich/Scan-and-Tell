""" 

Usage example: python ./batch_load_scannet_data.py
"""

import os
import sys
import datetime
import numpy as np
from load_scannet_data import export, read_segmentation
import pdb
import argparse 
import scannet_utils
import torch

import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='3D backbone (votenet / pointgroup)', default='votenet')
parser.add_argument('--delete_data',action='store_true',help='Delete raw data after processing')
opt = parser.parse_args()

model = opt.model

SCANNET_DIR = 'scans'
SCAN_NAMES = sorted([line.rstrip() for line in open('meta_data/scannetv2.txt')])
LABEL_MAP_FILE = 'meta_data/scannetv2-labels.combined.tsv'
DONOTCARE_CLASS_IDS = np.array([])
OBJ_CLASS_IDS = np.array([3,4,5,6,7,8,9,10,11,12,14,16,24,28,33,34,36,39]) #np.array([3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]) # exclude wall (1), floor (2), ceiling (22)
MAX_NUM_POINT = 50000
OUTPUT_FOLDER = './scannet_data'

def export_one_scan(model, scan_name, output_filename_prefix):    
    mesh_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '_vh_clean_2.ply')
    seg_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '_vh_clean_2.0.010000.segs.json')
     

    if (model == 'votenet'):

        meta_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '.txt') # includes axisAlignment info for the train set scans.  
        agg_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '_vh_clean.aggregation.json')
        mesh_vertices, aligned_vertices, semantic_labels, instance_labels, instance_bboxes, aligned_instance_bboxes = export(mesh_file, agg_file, seg_file, meta_file, LABEL_MAP_FILE, None)

        mask = np.logical_not(np.in1d(semantic_labels, DONOTCARE_CLASS_IDS))
        mesh_vertices = mesh_vertices[mask,:]
        aligned_vertices = aligned_vertices[mask,:]
        semantic_labels = semantic_labels[mask]
        instance_labels = instance_labels[mask]

        if instance_bboxes.shape[0] > 1:
            num_instances = len(np.unique(instance_labels))
            print('Num of instances: ', num_instances)

            # bbox_mask = np.in1d(instance_bboxes[:,-1], OBJ_CLASS_IDS)
            bbox_mask = np.in1d(instance_bboxes[:,-2], OBJ_CLASS_IDS) # match the mesh2cap
            instance_bboxes = instance_bboxes[bbox_mask,:]
            aligned_instance_bboxes = aligned_instance_bboxes[bbox_mask,:]
            print('Num of care instances: ', instance_bboxes.shape[0])
        else:
            print("No semantic/instance annotation for test scenes")

        N = mesh_vertices.shape[0]
        if N > MAX_NUM_POINT:
            choices = np.random.choice(N, MAX_NUM_POINT, replace=False)
            mesh_vertices = mesh_vertices[choices, :]
            aligned_vertices = aligned_vertices[choices, :]
            semantic_labels = semantic_labels[choices]
            instance_labels = instance_labels[choices]

        print("Shape of points: {}".format(mesh_vertices.shape))

        np.save(output_filename_prefix+'_vert.npy', mesh_vertices)
        np.save(output_filename_prefix+'_aligned_vert.npy', aligned_vertices)
        np.save(output_filename_prefix+'_sem_label.npy', semantic_labels)
        np.save(output_filename_prefix+'_ins_label.npy', instance_labels)
        np.save(output_filename_prefix+'_bbox.npy', instance_bboxes)
        np.save(output_filename_prefix+'_aligned_bbox.npy', aligned_instance_bboxes)

    elif (model == 'pointgroup'):

        TEST_SCAN_NAMES = sorted([line.rstrip() for line in open('meta_data/scannetv2_test.txt')])

        if (scan_name in TEST_SCAN_NAMES):

            vertices = scannet_utils.read_mesh_vertices_rgb(mesh_file)
            coords = np.ascontiguousarray(vertices[:, :3] - vertices[:, :3].mean(0))
            colors = np.ascontiguousarray(vertices[:, 3:6]) / 127.5 - 1
            
            torch.save((coords, colors), output_filename_prefix + '_pointgroup.pth')
            

        else:
            labels_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '_vh_clean_2.labels.ply')
            agg_file = os.path.join(SCANNET_DIR, scan_name, scan_name + '.aggregation.json')

            vertices = scannet_utils.read_mesh_vertices_rgb(mesh_file)
            coords = np.ascontiguousarray(vertices[:, :3] - vertices[:, :3].mean(0))
            colors = np.ascontiguousarray(vertices[:, 3:6]) / 127.5 - 1

            sem_labels = scannet_utils.get_labels(labels_file,OBJ_CLASS_IDS)

            segid_to_pointid, _ = read_segmentation(seg_file)

            instance_segids = scannet_utils.get_instance_segids(scan_name, agg_file)

            instance_labels = np.ones(sem_labels.shape[0]) * -100
            for i in range(len(instance_segids)):
                segids = instance_segids[i]
                pointids = []
                for segid in segids:
                    pointids += segid_to_pointid[segid]
                instance_labels[pointids] = i
                assert(len(np.unique(sem_labels[pointids])) == 1)
            N = len(sem_labels)
            if N > MAX_NUM_POINT:
                choices = np.random.choice(N, MAX_NUM_POINT, replace=False)
                coords = coords[choices, :]
                colors = colors[choices, :]
                sem_labels = sem_labels[choices]
                instance_labels = instance_labels[choices]
            torch.save((coords, colors, sem_labels, instance_labels), output_filename_prefix + '_pointgroup.pth')


def handler(func,path,exc_info):
    print('Removing directory failed with error:')
    print(exc_info)


def delete_raw(scan_name,handler_func):
    print('Deleting data...')
    path = os.path.join(SCANNET_DIR,scan_name)
    shutil.rmtree(path,onerror=handler_func)
    print('Raw data deleted')


def batch_export():
    np.random.seed(0)
    if not os.path.exists(OUTPUT_FOLDER):
        print('Creating new data folder: {}'.format(OUTPUT_FOLDER))                
        os.mkdir(OUTPUT_FOLDER)        
        
    for scan_name in SCAN_NAMES:
        output_filename_prefix = os.path.join(OUTPUT_FOLDER, scan_name)
        # if os.path.exists(output_filename_prefix + '_vert.npy'): continue
        
        print('-'*20+'begin')
        print(datetime.datetime.now())
        print(scan_name)
              
        export_one_scan(model, scan_name, output_filename_prefix)
             
        print('-'*20+'done')

        if (opt.delete_data):
            delete_raw(scan_name,handler)


if __name__=='__main__':    
    batch_export()
