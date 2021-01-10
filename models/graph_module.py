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
            raw_E: [batch_size,num_objects,num_objects,2 * num_features] object relations before message passing

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
            raw_relations = self._feature_differences(A,V)
            E = layer(raw_relations)
            #aggregation
            V = E.sum(dim=2)

        raw_relations = self._feature_differences(A,V)
        E = self._layers[-1](raw_relations)


        return V,E


    def _feature_differences(self,A,V):
        """
        Find the difference between each feature vector and its neighboring objects,
        concatenate them

        -----------
        Arguments:
        	A: [batch_size,num_objects,num_edges] Graph representation
            V: [batch_size,num_objects,num_features] Object features 

        Output:
            raw_relations: [batch_size,num_objects,num_neighbors,2 * num_features] 

        """
        batch_size,num_objects,num_neighbors = A.shape
        neighbors = V[torch.arange(batch_size).view(-1,1,1),A]

        differences = V.unsqueeze(dim=2) - neighbors # [B,M,K,F]
        V_expanded = V.unsqueeze(dim=2).repeat((1,1,num_neighbors,1)) # [B,M,K,F]
        raw_relations = torch.cat([V_expanded,differences],dim=3)

        return raw_relations


    def _aggregate(self,A,E):
        """
        Take the graph in form of node indices to build edges with per node and the object relation 
        features to aggregate features by summing the edge features
        -----------------
        Arguments:
            A: [batch_size,num_objects,num_edges] Graph representation
            E: [batch_size,num_objects,num_objects,num_features] Object relation features

        Output:
        	V: [batch_size,num_objects,num_features]
        """

        batch_size,num_objects,_,num_features = E.shape
        V = torch.zeros((batch_size,num_objects,num_features))

        #V = E[torch.arange(num_objects).unsqueeze(-1),A].sum(dim=1) #implementation without batch

        for batch_idx,graph in enumerate(A):
            V[batch_idx] = E[batch_idx,torch.arange(num_objects).unsqueeze(-1),graph].sum(dim=1)

        
        return V 


def build_graph(features,centers,num_neighbors,scores,model='votenet',drop_self_edge=True):
    """
    Build the scene graph from the object centers
    by taking K closest objects

    Proposed objects need to be filtered by the scores to find the valid ones. The graph must
    be constructed from valid objects.

	Features for invalid objects are set to zero.
    -------------
    Arguments:
       centers: [batch_size,num_objects,3] object centers
       num_neighbors: int, number of objects to build edges with; node rank 
       scores: [batch_size,num_objects,2]

    """

    #centers = centers.cpu()
    #scores = scores.cpu()

    if model == 'votenet':
        
    	#TODO: how to select valid objects? score > some threshold?
        objectness_masks = (torch.argmax(scores,2) == 1)

        # set invalid centers to infinite, so that they don't affect the closest neighbor search
        filtered_centers = torch.where(objectness_masks,centers,torch.ones_like(centers) * float('Inf'))  
        l2_distances = torch.norm(centers.unsqueeze(dim=2) - filtered_centers.unsqueeze(dim=1),dim=3)
        
        edge_idx = torch.topk(l2_distances,k=num_neighbors + 1,dim=2,largest=False)[1] #get k + 1 closest centers
        
        edge_idx = edge_idx * objectness_masks.int() # delete the edges of invalid objects
        features = features * objectness_masks.int() # zero-out features

    elif model == 'pointgroup':
        pass

    if drop_self_edge:
        edge_idx = edge_idx[:,:,1:] #drop the self-edge A[i,i]


    return edge_idx








