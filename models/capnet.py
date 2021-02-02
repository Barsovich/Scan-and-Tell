import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder
# from models.backbone_module import Pointnet2Backbone
from models.voting_module import VotingModule
# from models.proposal_module import ProposalModule
from models.pointgroup import PointGroup
from models.graph_module import GraphModule
from models.captioning_module import PlainCapModule, AttentionCapModule, SceneCaptionModule

from data.scannet.model_util_scannet import ScannetDatasetConfig
DC = ScannetDatasetConfig()


class VoteNetBackbone(nn.Module):
    def __init__(self,num_class,num_heading_bin, num_size_cluster, mean_size_arr, 
    input_feature_dim=0, num_proposal=256, num_locals=-1, vote_factor=1, sampling="vote_fps"):
        
        super().__init__()
        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert(mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling = sampling

        # --------- PROPOSAL GENERATION ---------
        # Backbone point feature learning
        # self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim)

        # Hough voting
        self.vgen = VotingModule(self.vote_factor, 256)

        # Vote aggregation and object proposal
        # self.proposal = ProposalModule(num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling)

    def forward(self,data_dict,epoch=None):

        # --------- HOUGH VOTING ---------
        data_dict = self.backbone_net(data_dict)
                
        # --------- HOUGH VOTING ---------
        xyz = data_dict["fp2_xyz"]
        features = data_dict["fp2_features"]
        data_dict["seed_inds"] = data_dict["fp2_inds"]
        data_dict["seed_xyz"] = xyz
        data_dict["seed_features"] = features
        
        xyz, features = self.vgen(xyz, features)
        features_norm = torch.norm(features, p=2, dim=1)
        features = features.div(features_norm.unsqueeze(1))
        data_dict["vote_xyz"] = xyz
        data_dict["vote_features"] = features

        # --------- PROPOSAL GENERATION ---------
        data_dict = self.proposal(xyz, features, data_dict)

        return data_dict



class CapNet(nn.Module):
    def __init__(self, vocabulary, embeddings, cfg=None, detection_backbone = 'votenet', num_class = DC.num_class, 
    num_heading_bin = DC.num_heading_bin, num_size_cluster = DC.num_size_cluster, mean_size_arr = DC.mean_size_arr, 
    input_feature_dim=0, num_proposal=256, num_locals=5, vote_factor=1, sampling="vote_fps",
    no_caption=True, use_topdown=False, query_mode="corner", 
    graph_mode="graph_conv", num_graph_steps=2, use_relation=False, graph_aggr="add",
    use_orientation=False, num_bins=6, use_distance=False, use_new=False, prepare_epochs=0,
    emb_size=300, hidden_size=512):
        super().__init__()

        self.detection_backbone = detection_backbone
        self.pointgroup_cfg = cfg
        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert(mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.vote_factor = vote_factor
        self.sampling = sampling
        self.no_caption = no_caption
        self.num_graph_steps = num_graph_steps
        self.prepare_epochs = prepare_epochs

        # --------- PROPOSAL GENERATION ---------
        if self.detection_backbone == 'votenet':
            self.detection = VoteNetBackbone(num_class, num_heading_bin, num_size_cluster, mean_size_arr,
                input_feature_dim, num_proposal, num_locals, vote_factor, sampling)
        elif self.detection_backbone == 'pointgroup':
            self.detection = PointGroup(self.pointgroup_cfg)
        else:
            print("Unknown backbone. Exiting...")
            exit(0)

        # Relation graph
        if use_relation: assert use_topdown # only enable use_relation in topdown captioning module

        if num_graph_steps > 0:
            self.graph = GraphModule(128, 128, num_graph_steps, num_proposal, 128, num_locals,
                query_mode, graph_mode, return_edge=use_relation, graph_aggr=graph_aggr,
                return_orientation=use_orientation, num_bins=num_bins, return_distance=use_distance, backbone=self.detection_backbone)

        # Caption generation
        if not no_caption:
            if use_topdown:
                self.caption = AttentionCapModule(128,emb_size,hidden_size,use_relation) #(vocabulary, embeddings, emb_size, 128, 
                    #hidden_size, num_proposal, num_locals, query_mode, use_relation)
            else:
                #self.caption = PlainCapModule(128,emb_size,hidden_size)  
                self.caption = SceneCaptionModule(vocabulary, embeddings, emb_size, 128, hidden_size, num_proposal)

    def forward(self, data_dict, epoch=1, use_tf=True, is_eval=False):
        """ Forward pass of the network

        Args:
            data_dict: dict
                {
                    point_clouds, 
                    lang_feat
                }

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """

        #######################################
        #                                     #
        #           DETECTION BRANCH          #
        #                                     #
        #######################################

        data_dict = self.detection(data_dict,epoch)

        #######################################
        #                                     #
        #           GRAPH ENHANCEMENT         #
        #                                     #
        #######################################

        if self.num_graph_steps > 0 and epoch > self.prepare_epochs: 
            data_dict = self.graph(data_dict)

        #######################################
        #                                     #
        #            CAPTION BRANCH           #
        #                                     #
        #######################################

        # --------- CAPTION GENERATION ---------
        if not self.no_caption and epoch > self.prepare_epochs: 
            data_dict = self.caption(data_dict, self.detection_backbone, use_tf, is_eval)

        return data_dict
