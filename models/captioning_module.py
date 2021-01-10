import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os



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

		if not self.RG:


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









