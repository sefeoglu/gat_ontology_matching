from itertools import count
import os, sys
from typing import ValuesView
import torch
import torch.nn as nn


import numpy as np

PACKAGE_PARENT = '.'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))


from graph_attention import GraphAttentionLayer as GATConv
from semantic_attention import SemanticAttention


class HANLayer(nn.Module):
 
    def __init__(self, in_size, out_size, dropout):
        super().__init__()
        self.alpha = 0.2
        self.gat = GATConv(in_size, out_size, dropout, alpha=self.alpha, concat=True)
        self.semantic_attention = SemanticAttention(in_size, out_size)
        
    def forward(self, hs, adj):
        
        semantic_embeddings = []
        for i, h in enumerate(hs):
            semantic_embeddings.append(self.gat(h, adj[i]))
        semantic_embeddings = torch.stack(semantic_embeddings)

        
        return  self.semantic_attention(semantic_embeddings)             

class SiamHAN(nn.Module):

    def __init__(self, emb_vals, max_types, max_paths, max_pathlen, threshold=0.9):
      
        super().__init__()
        self.dropout = 0.6
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.n_neighbours = max_types
        self.max_types = nn.Parameter(torch.DoubleTensor([max_types]))
        self.max_types.requires_grad = False
        self.max_paths = max_paths
        self.max_pathlen = max_pathlen
        self.embedding_dim = np.array(emb_vals).shape[1]
       
        self.threshold = nn.Parameter(torch.DoubleTensor([threshold]))
        self.threshold.requires_grad = False
        self.gnn = HANLayer(self.embedding_dim, self.embedding_dim, self.dropout)
        
        ### Embedding layer
        self.name_embedding = nn.Embedding(len(emb_vals), self.embedding_dim)
        self.name_embedding.load_state_dict({'weight': torch.from_numpy(np.array(emb_vals))})
        self.name_embedding.weight.requires_grad = False
        ## output layer
        self.output = nn.Linear(2*self.embedding_dim, 300)
        ## similarity score
        self.cosine_sim_layer = nn.CosineSimilarity(dim=1)
        
        self.weight = nn.Parameter(torch.DoubleTensor([1.0]))

        self.weight_prop = nn.Parameter(torch.DoubleTensor([0.33]))
        self.v = nn.Parameter(torch.DoubleTensor([1/(self.max_pathlen+1) for i in range(self.max_pathlen+1)]))
  
  
    def forward(self, nodes, features, prop_nodes, prop_features, max_prop_len):
        # define here again for ontologies
        # get node and feature embedding
        results = []
        
        self.max_prop_len = max_prop_len
        nodes = nodes.permute(1,0) # dim: (2, batch_size)
        features = features.permute(1,0,2,3,4) # dim: (2, batch_size, max_types, max_paths, max_pathlen)
 
        for i in range(2):
            node_emb = self.name_embedding(nodes[i]) # dim: (2, batch_size)
            feature_emb = self.name_embedding(features[i]) #  dim: (2, batch_size, max_types, max_paths, max_pathlen, 512)
         
            h_primes = []

            for j, _node_emb in enumerate(node_emb):
                # create graph structure
                h, adj = self.create_adjacency_and_homgraph(feature_emb[j], _node_emb)
                
                semantic_attention_out = self.gnn(h, adj)
                semantic_attention_out = torch.sum((self.v[None,:,None] * semantic_attention_out), dim=1)
                if self.n_neighbours == 2:
                    h_prime = self.weight*semantic_attention_out[0,:]\
                        + self.weight*semantic_attention_out[1,:]
                    h_primes.append(h_prime)

                if self.n_neighbours == 3:
                    h_prime = self.weight*semantic_attention_out[0,:]\
                        + self.weight*semantic_attention_out[1,:]\
                        +self.weight*semantic_attention_out[2,:]
                    h_primes.append(h_prime)
                if self.n_neighbours == 4:
                    h_prime = self.weight*semantic_attention_out[0,:]\
                        + self.weight*semantic_attention_out[1,:]\
                        +self.weight*semantic_attention_out[2,:]\
                        +self.weight*semantic_attention_out[3,:]
                    h_primes.append(h_prime)
                if self.n_neighbours == 5:
                    h_prime = self.weight*semantic_attention_out[0,:]\
                        + self.weight*semantic_attention_out[1,:]\
                        +self.weight*semantic_attention_out[2,:]\
                        +self.weight*semantic_attention_out[3,:]\
                        +self.weight*semantic_attention_out[4,:]
                    h_primes.append(h_prime)

            h_primes = torch.stack(h_primes)
            context = torch.cat((node_emb, h_primes), dim=1)
            contextual = self.output(context)
            results.append(contextual)
        ## consine similarity layer which computes the similarity score
                
        sim_ent = self.cosine_sim_layer(results[0], results[1])

        if prop_nodes.nelement() != 0:
            # Calculate prop sum
            aggregated_prop_sum = torch.sum(self.name_embedding(prop_features), dim=-2)
            
            sim_prop = self.weight_prop * self.cosine_sim_layer(aggregated_prop_sum[:,0,0], aggregated_prop_sum[:,1,0])
            sim_prop += self.weight_prop * self.cosine_sim_layer(aggregated_prop_sum[:,0,1], aggregated_prop_sum[:,1,1])
            sim_prop += (1.0-2*self.weight_prop) * self.cosine_sim_layer(aggregated_prop_sum[:,0,2], aggregated_prop_sum[:,1,2])
 
            

            return torch.cat((sim_ent, sim_prop))
            
        return sim_ent

    def create_adjacency_and_homgraph(self, features_list,  node):

        adj, hs = [], []

        for _, features in enumerate(features_list):
            h, g =  self.create_graph_freatures(features, node)
            hs.append(h)
            adj.append(g)
        return hs, adj

    def create_adj(self, size):

        adj = np.zeros(shape=(size, size))
        adj[0,:] = 1.0
        return adj


    def create_graph_freatures(self, features, c_node):

        h = []

        values = [feature.detach().to("cpu").numpy()[0,0] for feature in features]
        if not all(np.array(values) == 0):
            h.append(c_node.detach().to("cpu").numpy())
            
            for _, feature in enumerate(features):
                for _, feature_node in enumerate(feature):
                    h.append(feature_node.detach().to("cpu").numpy())
                break
        else:
            for _ in range(self.max_pathlen+1):
                h.append(np.zeros(self.embedding_dim))

        adj = self.create_adj(len(h))
        h_tensor = torch.DoubleTensor(h).to(self.device)
        return h_tensor, adj