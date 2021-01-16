import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
from data.scannet.model_util_scannet import ScannetDatasetConfig
from config.config_votenet import CONF
from utils.box_util import get_3d_box, get_3d_box_batch, box3d_iou, box3d_iou_batch_tensor
from lib.pointgroup_ops.functions import pointgroup_ops

# constants
DC = ScannetDatasetConfig()

def select_target(data_dict,backbone='votenet'):
    
    if backbone == 'votenet':

        # predicted bbox
        pred_bbox = data_dict["bbox_corner"] # batch_size, num_proposals, 8, 3
        batch_size, num_proposals, _, _ = pred_bbox.shape

        # ground truth bbox
        gt_bbox = data_dict["ref_box_corner_label"] # batch_size, 8, 3

        target_ids = []
        target_ious = []
        for i in range(batch_size):
            # convert the bbox parameters to bbox corners
            pred_bbox_batch = pred_bbox[i] # num_proposals, 8, 3
            gt_bbox_batch = gt_bbox[i].unsqueeze(0).repeat(num_proposals, 1, 1) # num_proposals, 8, 3
            ious = box3d_iou_batch_tensor(pred_bbox_batch, gt_bbox_batch)
            target_id = ious.argmax().item() # 0 ~ num_proposals - 1
            target_ids.append(target_id)
            target_ious.append(ious[target_id])

        target_ids = torch.LongTensor(target_ids).cuda() # batch_size
        target_ious = torch.FloatTensor(target_ious).cuda() # batch_size

    elif backbone == 'pointgroup':

        #predictions
        scores, proposals_idx, proposals_offset = data_dict['proposal_scores']

        #ground truth info
        target_instance_labels = data_dict['target_instance_labels']
        target_instance_pointnum = data_dict['target_instance_pointnum']

        ious = pointgroup_ops.get_iou(proposals_idx[:, 1].cuda(), proposals_offset.cuda(),
            target_instance_labels.cuda(), target_instance_pointnum.cuda()) # shape: [num_proposals, batch_size]

        target_ious, target_ids = ious.max(0)


    return target_ids, target_ious


class PlainCapModule(nn.Module):

	def __init__(self,num_features=128,embedding_size=300,hidden_size=512):
		"""
		"""
		super().__init__()
		self.num_features = num_features
		self.embedding_size = embedding_size
		self.hidden_size = hidden_size


	def forward(self):

		pass






class AttentionModule(nn.Module):

	def __init__(self,num_features,hidden_size,relational_graph=True):
		"""

		"""

		super().__init__()
		self.num_features = num_features
		self.hidden_size = hidden_size
		self.RG = relational_graph

		self.W_v = nn.Linear(self.num_features,self.num_features,bias=False)
		self.W_h = nn.Linear(self.hidden_size,self.num_features,bias=False)
		self.W_a = nn.Linear(self.num_features,1,bias=False)

		


	def forward(self,h,V_r=None):
		"""
		Arguments: 
			V_r: [num_objects,num_features]
			h: [1,1,hidden_size]

		Output:
			attention: [num_features]
		"""

		num_objects = V_r.shape[0]

		ones_attention = torch.ones((1,self.num_features))
		H = h.repeat((num_objects,1,1))

		attention = F.softmax(self.W_a(self.W_v(V_r) + self.W_h(H)),dim=0) 

		context_vect = V_r * attention
		aggr_context_vect = context_vect.sum(0)





		return aggr_context_vect



class AttentionCapModule(nn.Module):

	def __init__(self,num_features=128,embedding_size=300,hidden_size=512,relational_graph=True):
		"""
		Context Aware Captioning Module

		"""

		super().__init__()
		self.num_features = num_features
		self.embedding_size = embedding_size
		self.hidden_size = hidden_size
		self.RG = relational_graph

		self.W_e = nn.Linear(self.embedding_size,self.embedding_size,bias=False)
		self.FC1 = nn.Linear(self.hidden_size + self.num_features + self.embedding_size,self.hidden_size)
		self.fusionGRU = nn.GRU(self.hidden_size,self.hidden_size)
		self.attention_module = AttentionModule(self.num_features,self.hidden_size,self.RG)
		self.FC2 = nn.Linear(self.hidden_size+self.num_features,self.hidden_size)
		self.languageGRU = nn.GRU(self.hidden_size,self.hidden_size)
		self.FC3 = nn.Linear(self.hidden_size,self.embedding_size)

		#initalize states
		self.token = torch.zeros((1,1,self.embedding_size))
		self.hidden1 = torch.zeros((1,1,self.hidden_size))
		self.hidden2 = torch.zeros((1,1,self.hidden_size))



	def forward(self,V,E):
		"""
		Arguments:
			V: [num_objects,num_features] enhanced object features
			E: [num_objects,num_neighbors,num_features] object relation features
		"""

		tokens = []
		for i,object_feature in enumerate(V):

			V_r = object_feature + E[i]

			x_t = self.W_e(self.token)

			input = torch.cat([self.hidden2,object_feature.view(1,1,-1),x_t],dim=2)
			_, self.hidden1 = self.fusionGRU(self.FC1(input),self.hidden1)

			aggr_context_vect = self.attention_module(V_r,self.hidden1)
			fused_features = torch.cat([aggr_context_vect,self.hidden1],dim=2)

			_, self.hidden2 = self.languageGRU(self.FC2(fused_features),self.hidden2)

			self.token = self.FC3(self.hidden2)

			tokens.append(self.token.view(-1))

		return tokens

		

class SceneCaptionModule(nn.Module):
    def __init__(self, vocabulary, embeddings, emb_size=300, feat_size=128, hidden_size=512, num_proposals=256):
        super().__init__() 

        self.vocabulary = vocabulary
        self.embeddings = embeddings
        self.num_vocabs = len(vocabulary["word2idx"])

        self.emb_size = emb_size
        self.feat_size = feat_size
        self.hidden_size = hidden_size
        self.num_proposals = num_proposals

        # transform the visual signals
        self.map_feat = nn.Sequential(
            nn.Linear(feat_size, emb_size),
            nn.ReLU()
        )

        # captioning core
        self.recurrent_cell = nn.GRUCell(
            input_size=emb_size,
            hidden_size=emb_size
        )
        self.classifier = nn.Linear(emb_size, self.num_vocabs)

    def step(self, step_input, hidden):
        hidden = self.recurrent_cell(step_input, hidden) # num_proposals, emb_size

        return hidden, hidden

    def forward(self, data_dict, backbone='votenet', use_tf=True, is_eval=False, max_len=CONF.TRAIN.MAX_DES_LEN):
        if not is_eval:
            data_dict = self.forward_sample_batch(data_dict, backbone, max_len)
        else:
            data_dict = self.forward_scene_batch(data_dict, use_tf, max_len)

        return data_dict

    def forward_sample_batch(self, data_dict, backbone='votenet',
        max_len=CONF.TRAIN.MAX_DES_LEN, min_iou=CONF.TRAIN.MIN_IOU_THRESHOLD):
        """
        generate descriptions based on input tokens and object features
        """

        # unpack
        word_embs = data_dict["lang_feat"] # batch_size, max_len, emb_size
        des_lens = data_dict["lang_len"] # batch_size
        obj_feats = data_dict["proposal_feature"] # batch_size, num_proposals, feat_size for votenet
                                                  # total_num_proposals, feat_size for pointgroup
        
        num_words = des_lens[0]
        batch_size = des_lens.shape[0]

        # transform the features
        obj_feats = self.map_feat(obj_feats) # batch_size, num_proposals, emb_size OR total_num_proposals, feat_size 

        # find the target object ids
        target_ids, target_ious = select_target(data_dict,backbone)

        # select object features
        if backbone == 'votenet':
            target_feats = torch.gather(
                obj_feats, 1, target_ids.view(batch_size, 1, 1).repeat(1, 1, self.emb_size)).squeeze(1) # batch_size, emb_size
        elif backbone == 'pointgroup':
            target_feats = obj_feats[target_ids] #batch_size, emb_size

        # recurrent from 0 to max_len - 2
        outputs = []
        hidden = target_feats # batch_size, emb_size
        step_id = 0
        step_input = word_embs[:, step_id] # batch_size, emb_size
        while True:
            # feed
            step_output, hidden = self.step(step_input, hidden)
            step_output = self.classifier(step_output) # batch_size, num_vocabs
            
            # store
            step_output = step_output.unsqueeze(1) # batch_size, 1, num_vocabs 
            outputs.append(step_output)

            # next step
            step_id += 1
            if step_id == num_words - 1: break # exit for train mode
            step_input = word_embs[:, step_id] # batch_size, emb_size

        outputs = torch.cat(outputs, dim=1) # batch_size, num_words - 1/max_len, num_vocabs

        # NOTE when the IoU of best matching predicted boxes (proposals) and the GT boxes 
        # are smaller than the threshold, the corresponding predicted captions
        # should be filtered out in case the model learns wrong things
        good_bbox_masks = target_ious > min_iou # batch_size
        # good_bbox_masks = target_ious != 0 # batch_size

        num_good_bboxes = good_bbox_masks.sum()
        mean_target_ious = target_ious[good_bbox_masks].mean() if num_good_bboxes > 0 else torch.zeros(1)[0].cuda()

        # store
        data_dict["lang_cap"] = outputs
        data_dict["pred_ious"] = mean_target_ious
        data_dict["good_proposal_masks"] = good_bbox_masks

        return data_dict


    def forward_scene_batch(self, data_dict, use_tf=False, max_len=CONF.TRAIN.MAX_DES_LEN):
        """
        generate descriptions based on input tokens and object features
        """

        # unpack
        word_embs = data_dict["lang_feat"] # batch_size, max_len, emb_size
        des_lens = data_dict["lang_len"] # batch_size
        obj_feats = data_dict["proposal_feature"] # batch_size, num_proposals, feat_size
        
        num_words = des_lens[0]
        batch_size = des_lens.shape[0]

        # transform the features
        obj_feats = self.map_feat(obj_feats) # batch_size, num_proposals, emb_size

        # recurrent from 0 to max_len - 2
        outputs = []
        for prop_id in range(self.num_proposals):
            # select object features
            target_feats = obj_feats[:, prop_id] # batch_size, emb_size

            # start recurrence
            prop_outputs = []
            hidden = target_feats # batch_size, emb_size
            step_id = 0
            step_input = word_embs[:, step_id] # batch_size, emb_size
            while True:
                # feed
                step_output, hidden = self.step(step_input, hidden)
                step_output = self.classifier(step_output) # batch_size, num_vocabs
                
                # predicted word
                step_preds = []
                for batch_id in range(batch_size):
                    idx = step_output[batch_id].argmax() # 0 ~ num_vocabs
                    word = self.vocabulary["idx2word"][str(idx.item())]
                    emb = torch.FloatTensor(self.embeddings[word]).unsqueeze(0).cuda() # 1, emb_size
                    step_preds.append(emb)

                step_preds = torch.cat(step_preds, dim=0) # batch_size, emb_size

                # store
                step_output = step_output.unsqueeze(1) # batch_size, 1, num_vocabs 
                prop_outputs.append(step_output)

                # next step
                step_id += 1
                if not use_tf and step_id == max_len - 1: break # exit for eval mode
                if use_tf and step_id == num_words - 1: break # exit for train mode
                step_input = step_preds if not use_tf else word_embs[:, step_id] # batch_size, emb_size

            prop_outputs = torch.cat(prop_outputs, dim=1).unsqueeze(1) # batch_size, 1, num_words - 1/max_len, num_vocabs
            outputs.append(prop_outputs)

        outputs = torch.cat(outputs, dim=1) # batch_size, num_proposals, num_words - 1/max_len, num_vocabs

        # store
        data_dict["lang_cap"] = outputs

        return data_dict







