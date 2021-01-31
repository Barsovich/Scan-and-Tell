import os
import sys
import json 

sys.path.append(os.path.join(os.getcwd(),"../")) # HACK add the root folder
from config.config_votenet import CONF

SCANNET_TRAIN_SMALL = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META,"scannetv2_train_small.txt"))])
SCANNET_VAL_SMALL = sorted([line.rstrip() for line in open(os.path.join(CONF.PATH.SCANNET_META,"scannetv2_val_small.txt"))])

SCANREFER = json.load(open(os.path.join(CONF.PATH.DATA,"ScanRefer_filtered.json")))

train_small = []
val_small = []

for data in SCANREFER:
    scene_id = data["scene_id"]
    if scene_id in SCANNET_TRAIN_SMALL:
        train_small.append(data)
    elif scene_id in SCANNET_VAL_SMALL:
        val_small.append(data)
    else:
        continue

with open(os.path.join(CONF.PATH.DATA,"ScanRefer_filtered_train.json"), "w") as train_file:
    json.dump(train_small, train_file, indent=4)

with open(os.path.join(CONF.PATH.DATA,"ScanRefer_filtered_val.json"), "w") as val_file:
    json.dump(val_small, val_file, indent=4)
