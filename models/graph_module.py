import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys


class MessagePassingLayer(nn.Module):

    def __init__(self,input_dim,output_dim):

        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.model = nn.Sequential(
            nn.Linear(self.input_dim,self.output_dim),
            nn.ReLU(),
            nn.Linear(self.output_dim,self.output_dim)
            )

     def forward(self,raw_E):
        """
        Arguments:
            raw_E: [num_objects,num_objects,2 * num_features] object relations before message passing

        """
        E = self.model(raw_E)

        return E 


class GraphModule(nn.Module):

    def __init__(self,num_features,num_layers=2):

        super().__init__()
        self.num_features = num_features
        self.num_layers = num_layers

        #message passing layers
        self._layers = nn.ModuleList([
            MessagePassingLayer(2 * self.num_features, self.num_features) for i in range(num_layers + 1)
            ])


    def forward(self,A,features):

        V = features

        for layer in self._layers[:-1]:
            raw_relations = self._feature_differences(V)
            E = layer(raw_relations)
            V = self._aggregate(A,E)

        raw_relations = self._feature_differences(V)
        E = self._layers[-1](raw_relations)


        return V,E


    def _feature_differences(self,V):
        """
        Find the difference between each feature vector and the others,
        concatenate them

        -----------
        Arguments:
            V: [num_objects,num_features] Object features 

        Output:
            raw_relations: [num_objects,num_objects,2 * num_features] 

        """

        differences = V.unsqueeze(dim=1) - V # [M,M,F]
        V_expanded = V.unsqueeze(dim=1).repeat((1,V.shape[0],1)) # [M,M,F]
        raw_relations = torch.cat([V_expanded,differences],dim=2)

        return raw_relations


    def _aggregate(self,A,E):
        """
        Take the graph in form of node indices to build edges with per node and the object relation 
        features to aggregate node features by summing the edge features
        -----------------
        Arguments:
            A: [num_objects,num_edges] Graph representation
            E: [num_objects,num_objects,features] Object relation features
        """
        # A.type(E.dtype)
        # V = torch.zeros(E.shape[0],E.shape[2])

        # for idx,mask in enumerate(A):
        #     feature = torch.matmul(mask,E[idx])
        #     V[idx] = feature

        num_objects = E.shape[0]
        V = E[torch.arange(num_objects).unsqueeze(-1),A].sum(dim=1)
        
        return V 




def build_graph(centers,num_neighbors,drop_self_edge=True):
    """
    Build the scene graph from the object centers
    by taking K closest objects

    -------------
    Arguments:
       centers: object centers
       num_neighbors: number of objects to build edges with; node rank 

    """
    l2_distances = torch.norm(centers.unsqueeze(dim=1) - centers,dim=2)
    edge_idx = torch.topk(l2_distances,k=num_neighbors + 1,dim=1,largest=False)[1] #get k + 1 closest centers

    if drop_self_edge:
        edge_idx = edge_idx[:,:1] #drop the self-edge A[i,i]

    return edge_idx








