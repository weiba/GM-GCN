import numpy as np
import scipy.sparse as sp
import torch
import torch_sparse
import torch.nn.functional as F
from torch.nn import Linear
from model_other import CNN, MLP

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sparse_to_tuple(matrix):
    if not sp.isspmatrix_coo(matrix):
        matrix=matrix.tocoo()
        coords = np.vstack((matrix.row, matrix.col)).transpose()
        values = matrix.data
        shape = matrix.shape
        return coords, values, shape

def sparse_to_matrix(matrix):
    tmp = sp.csr_matrix(matrix)
    coords, values, shape = sparse_to_tuple(tmp)
    coords = torch.LongTensor(coords.transpose())
    values = torch.FloatTensor(values)
    mat = torch.sparse.FloatTensor(coords, values, shape)
    mat = mat.to(device)

    coords_trans, values_trans = torch_sparse.transpose(coords, values, shape[0], shape[1])
    mat_trans = torch.sparse.FloatTensor(coords_trans, values_trans, (shape[1],shape[0]))
    mat_trans = mat_trans.to(device)

    return mat, mat_trans

def graph_inception_unit(network, Korder, weight):
    temp_Korder = Korder
    temp_Korder = torch.sparse.mm(network, temp_Korder)
    relfeature = weight(temp_Korder)

    return relfeature, temp_Korder

def Multiply(network, l_feat, r_feat, weight):
    l_feat = torch.mm(network, l_feat)
    mul = torch.mul(l_feat,r_feat)
    relfeature = weight(mul)

    return relfeature, mul

class layers(torch.nn.Module):
    def __init__(self, hop, inputdims, hiddendims, outputdims):
        super(layers,self).__init__()
        self.conv1 = graph_inception(hop, inputdims, hiddendims)
        self.conv2 = graph_inception(hop, hiddendims, outputdims)
    def forward(self, l_feat, r_feat, network):
        x1 = self.conv1(l_feat, r_feat, network)
        y1 = torch.relu(x1[0]) 
        z1 = torch.relu(x1[1])
        x2 = self.conv2(y1, z1, network)
        y2 = torch.relu(x2[0])
        z2 = torch.relu(x2[1]) 

        return y2

class graph_inception(torch.nn.Module):
    def __init__(self, hop, inputdims, outputdims):
        super(graph_inception, self).__init__()
        self.hop = hop
        self.lin1 = Linear(inputdims, outputdims)
        self.lin2 = Linear(inputdims, outputdims)
    def forward(self, l_feat, r_feat, network):
        Korder_list = [[l_feat],[r_feat]]
        convresults = [[] for _ in range(2)]
        l_index = 0
        r_index = 1
        network_l, network_r = sparse_to_matrix(network)
        for i in range(self.hop):
            temp_Korders = [[] for _ in range(2)]
            # l_node
            x0 = F.dropout(Korder_list[r_index][i],training=self.training)
            l_reFeat, l_Korder = graph_inception_unit(network_l, x0, self.lin1)
            convresults[l_index].append(l_reFeat)
            temp_Korders[l_index].append(l_Korder)

            # r_node
            x1 = F.dropout(Korder_list[l_index][i], training=self.training)
            r_reFeat, r_Korder = graph_inception_unit(network_r, x1, self.lin1)
            convresults[r_index].append(r_reFeat)
            temp_Korders[r_index].append(r_Korder)

            # Dot product feature with neighbors
            l_mul, l_Korder_mul = Multiply(network_l, x0, x1, self.lin2)
            r_mul, r_Korder_mul = Multiply(network_r, x1, x0, self.lin2)
            convresults[l_index].append(l_mul)
            convresults[r_index].append(r_mul)
            temp_Korders[l_index].append(l_Korder_mul)
            temp_Korders[r_index].append(r_Korder_mul)

            for j in range(len(temp_Korders)):
                if len(temp_Korders[j]) == 1:
                    temp = temp_Korders[j][0]
                else:
                    temp = temp_Korders[j][0]
                    for k in range(1, len(temp_Korders[j])):
                        temp = temp + temp_Korders[j][k]
                Korder_list[j].append(temp)

        final_convs = []
        for convresult in convresults:
            if len(convresult) == 1:
                final_convs.append(convresult[0])
            else:
                temp = convresult[0]
                for z in range(1, len(convresult)):
                    temp = temp + convresult[z]
                final_convs.append(temp)

        return final_convs

class Net(torch.nn.Module):
    def __init__(self,l_feat, r_feat, ori_feat, network, hop, inputdims, hiddendims, outputdims):
        super(Net, self).__init__()
        self.l_feat = l_feat
        self.r_feat = r_feat
        self.ori_feat = ori_feat
        self.network = network
        self.gcn = layers(hop, inputdims, hiddendims, outputdims)
        self.cnn = CNN()
        self.mlp = MLP()
    def forward(self):
        # gene-miRNA network
        x = self.gcn(self.l_feat, self.r_feat, self.network)

        # 1D convolution module
        w1 = self.cnn(x)

        # mlp module
        w0 = self.ori_feat
        w2 = self.mlp(w0)
        w = w1 + w2         # final feature

        # Logistic Regression Module input features
        w3 = torch.cat((w1, w2),dim=1)

        return w, w1, w3
