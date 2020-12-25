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
 			raw_E: [num_objects,num_objects,2 * num_features] 

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
		Arguments:
			A: [num_objects,num_objects] Adjacency matrix
			E: [num_objects,num_objects,features] Object relation features
		"""



		return V 
