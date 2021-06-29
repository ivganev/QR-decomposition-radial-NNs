import numpy as np
from typing import List

import torch
import math
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torch.nn.functional as F

dev = 'cpu'
if torch.cuda.is_available():
    dev = 'cuda:0'
print("Running on:",dev)
device = torch.device(dev)


#################################
## Auxiliary functions needed in the representation class
#################################

# Embed n x n matrices into the bottom right of (n+1)x(n+1) matrices with a one in the top left.
def extend_matrix(q: np.array) -> np.array:
    num_rows, num_cols = q.shape
    assert q.shape[0] == q.shape[1], "Matrix must be square"
    q_ext = np.eye(num_rows + 1)
    q_ext[1:, 1:] = q
    return q_ext

# Pad a matrix with zeros.
def padzeros(M: np.array, newrows: int, newcols: int) -> np.array:
    oldrows, oldcols = M.shape
    return np.pad(M,((0,newrows-oldrows),(0,newcols-oldcols)),mode="constant")


#################################
## Class for representations of the neural quiver
#################################

class representation:
    def __init__(self, W: List[np.array]):
        
        # Check that W defines a representation of the neural quiver
        assert all([W[i].shape[0] +1 == W[i+1].shape[1] for i in range(len(W)-1)]), \
        "The input does not define a representation"
        
        self.W = W
        self.L = len(W)
        self.n = [W[0].shape[1]-1] + [w.shape[0] for w in W]
        
        # Compute the reduced dimension vector
        self.n_red = [self.n[0]]
        for i in range(1,self.L):
            previous = self.n_red[-1]
            self.n_red.append(min(self.n[i], previous + 1))
        self.n_red.append(self.n[-1])
        
        # Compute the QR decomposition and transformed representation
        self.QR_decomposition()        
        
    def QR_decomposition(self):
        
        # For readability
        n = self.n 
        n_red = self.n_red
        
        # Initiation
        Q = []
        R = []
        U = [np.zeros((n[1],n[0]+1))]
        W_cur = self.W[0]
        
        for s in range(self.L-1):
            # Compute the QR decomposition of W_cur
            Q_cur,R_cur = np.linalg.qr(W_cur, mode="complete")
            
            # Ensure that the diagonal entries of R_cur are positive
            sign_corrections = np.eye(Q_cur.shape[0])
            for i in range(min(len(sign_corrections),R_cur.shape[1])):
                sign_corrections[i][i] = np.diagonal(np.sign(R_cur))[i]
            Q_cur = Q_cur @ sign_corrections
            R_cur = sign_corrections @ R_cur
            
            if n_red[s+1] <= n[s+1]:
                
                # Update Q and R            
                Q.append(Q_cur)
                R.append(R_cur[:n_red[s]+1])

                # Compute the extended version of Q_cur and its transpose
                Q_ext = extend_matrix(Q_cur)
                Q_ext_t = extend_matrix(np.transpose(Q_cur))

                # Compute the next value of U
                QP = np.copy(Q_ext)
                QP[:,:n_red[s+1] +1] = np.zeros((n[s+1]+1,n_red[s+1]+1))
                U.append(self.W[s+1] @ QP @ Q_ext_t)

                # Update the next value of W_cur
                W_cur = (self.W[s+1] @ Q_ext)[:,:n_red[s+1]+1]
                
            else:

                # Update Q, R, and U            
                Q.append(Q_cur)
                R.append(R_cur)
                U.append(np.zeros((n[s+2], n[s+1]+1)))

                # Update the next value of W_cur
                Q_ext = extend_matrix(Q_cur)
                W_cur = self.W[s+1] @ Q_ext
                
        R.append(W_cur)
        
        self.Q = Q
        self.R = R
        self.U = U
                            
        return (Q,R,U)
    
    
    def reduced_representation(self, padded=False):
                            
        if padded:
            
            # Compute the padded version of R
            R_padded = [padzeros(self.R[0],n[1], n[0]+1)]
            for s in range(1, L-1):
                R_padded.append(padzeros(self.R[s],n[s+1],n[s]+1))
            R_padded.append(padzeros(self.R[L-1],n[L],n[L-1]+1))
            self.R_padded = R_padded
                                
            return representation(R_padded)
                                
        else:
            
            return representation(self.R)
    
        
    def transformed_representation(self):
        
        Q = self.Q
        W = self.W
        U = self.U
        L = self.L
                                        
        # Compute Q_inv acting on W
        Q_inv_W = [np.round(np.transpose(Q[0]) @ W[0],10)]
        for s in range(1, L-1):
            Q_inv_W.append(np.round(np.transpose(Q[s]) @ W[s] @ extend_matrix(Q[s-1]), 10)) 
        Q_inv_W.append(np.round(W[L-1]@ extend_matrix(Q[L-2]), 10))               
        self.Q_inv_W = Q_inv_W
        
        # For future reference: Compute Q_inv acting on U
        Q_inv_U = [np.round(np.transpose(Q[0]) @ U[0], 10)]
        for s in range(1, L-1):
            Q_inv_U.append(np.round(np.transpose(Q[s]) @ U[s] @ extend_matrix(Q[s-1]), 10)) 
        Q_inv_U.append(np.round(U[L-1]@ extend_matrix(Q[L-2]), 10))               
        self.Q_inv_U = Q_inv_U

        return representation(self.Q_inv_W)
        

    def test_decomposition(self):
        
        #For Formula Readability.  
        Q = self.Q
        R = self.R
        U = self.U
        L = self.L 
        n = self.n
                
        #Reconstructed W
        W_test = ([U[0] + Q[0] @ padzeros(R[0],n[1], n[0]+1)] + 
          [U[s] + Q[s] @ padzeros(R[s],n[s+1],n[s]+1) @ extend_matrix(np.transpose(Q[s-1])) for s in range(1,L-1)] + 
          [U[L-1] + padzeros(R[L-1],n[L],n[L-1]+1) @ extend_matrix(np.transpose(Q[L-2]))])
        
        return (max([np.max(np.abs(w1-w2)) for w1,w2 in zip(self.W,W_test)]) < 1e-10)
    
    
    def print_shapes(self):
        print("Q    shapes:",[q.shape for q in self.Q])
        print("R    shapes:",[r.shape for r in self.R])
        print("U    shapes:",[u.shape for u in self.U])
        return    


#################################
## Class for radial activation functions
#################################

'''
Possible etas:
    F.relu
    torch.exp
    torch.sigmoid
    lambda x : x**k
    torch.atan
    lambda x : x**2/(1+x**2)
'''
class RadAct(nn.Module):
    def __init__(self, eta = F.relu, has_bias=True):
        super().__init__()
        self.has_bias = has_bias
        if self.has_bias:
            bias = torch.rand(1)
            self.bias = torch.nn.parameter.Parameter(bias)
        else:
            self.bias = 0.
        self.eta = eta
        
    def forward(self,x):
        # x: [Batch x Channel]
        r = torch.linalg.norm(x, dim=-1) 
        if torch.min(r) < 1e-6:
            r += 1e-6
        scalar = self.eta(r + self.bias) / r
        return x * scalar.unsqueeze(-1)   


#################################
## Class for radial neural networks
#################################

class RadNet(nn.Module):
    
    def __init__(self, eta, dims=[1,2,3,4,1], has_bias=True):
        super().__init__()
        self.eta = eta
        self.dims = dims
        self.has_bias = has_bias
        
        # Reduced dimension vector
        self.dims_red = [self.dims[0]]
        for i in range(1,len(dims)-1):
            previous = self.dims_red[-1]
            self.dims_red.append(min(self.dims[i], previous + 1))
        self.dims_red.append(self.dims[-1])
        
        self.depth = len(self.dims)-1
        self.layers = nn.ModuleList([nn.Linear(self.dims[i], self.dims[i+1]) for i in range(self.depth)])
        
        self.output_layer = self.layers[-1]
        self.act_fns = nn.ModuleList([RadAct(self.eta, has_bias=has_bias) for _ in range(self.depth-1)])

    def forward(self, x):
        h = x
        for lin,act in zip(self.layers[:-1], self.act_fns):
            h = act(lin(h))
        return self.output_layer(h)
    
    def set_weights(self, new_weights: representation):
        assert type(new_weights) == representation, \
        "Input must be a representation object"
        assert new_weights.n == self.dims,\
        "The dimension vector of the imput does not match the widths of the network"
        with torch.no_grad():
            for layer,w in zip(self.layers, new_weights.W):
                layer.weight = torch.nn.parameter.Parameter(torch.tensor(w[:,1:], dtype=torch.float32))
                layer.bias = torch.nn.parameter.Parameter(torch.tensor(w[:,0], dtype=torch.float32))
        return None
    
    def set_activation_biases(self, new_biases: List[float]):    
        assert len(new_biases) == self.depth-1, \
         "Input must be a list of length equal to the number of layers."
        with torch.no_grad():
            for act, b in zip(self.act_fns, new_biases):
                if self.has_bias:
                    act.bias = torch.nn.parameter.Parameter(torch.tensor(b))
                else:
                    act.bias = b
        return None

    def export_weights(self) -> representation:
        W = []
        for layer in self.layers:
            A = layer.weight.detach().cpu().numpy()
            b = layer.bias.detach().cpu().numpy()
            W.append(np.concatenate((np.expand_dims(b, 1), A), axis =1))
        return representation(W)
    
    def export_activation_biases(self) -> List[float]:
        if not self.has_bias:
            return [0.]*(self.depth -1)
        act_biases = []
        for act in self.act_fns:
            b = act.bias.detach().cpu().numpy()
            act_biases.append(b)
        return act_biases
    
    def export_reduced_weights(self) -> representation:
        exported_rep = self.export_weights()
        #assert exported_rep.reducible, "Debugging required: Error in exporting weights"
        return exported_rep.reduced_representation()
    
    def transformed_network(self):
        exported_rep = self.export_weights()
        rep_transformed = exported_rep.transformed_representation()
        net_trans = RadNet(self.eta, rep_transformed.n, self.has_bias)
        net_trans.set_weights(rep_transformed)
        net_trans.set_activation_biases(self.export_activation_biases())
        return net_trans
        
    def reduced_network(self):
        exported_rep = self.export_weights()
        rep_red = exported_rep.reduced_representation()
        net_red = RadNet(self.eta, rep_red.n, self.has_bias)
        net_red.set_weights(rep_red)
        net_red.set_activation_biases(self.export_activation_biases())
        return net_red


#################################
## Loss function (mean square error)
#################################

def loss_fn(y_predict, y_train):
    squared_diffs = (y_predict - y_train)**2
    return squared_diffs.mean()


#################################
## Masks used in projected gradient descent
#################################

def single_mask(red_rows, red_cols, orig_rows, orig_cols):
    assert red_rows <= orig_rows 
    assert red_cols <= orig_cols
    mask = torch.ones((orig_rows, orig_cols))
    for j in range(red_rows, orig_rows):
        mask[j][:red_cols] = 0
    return mask

def masks(reduced_dims, orig_dims):
    N = len(reduced_dims)
    assert N == len(orig_dims), \
    "Lengths must match"
    assert all([reduced_dims[i] <= orig_dims[i] for i in range(N)]), \
        "Reduced dimension vector must not be larger in any coordinate"
    
    masks = []
    
    # Add masks in order corresponding to the parameters
    for i in range(N-1):
        masks.append(single_mask(reduced_dims[i+1], reduced_dims[i], orig_dims[i+1], orig_dims[i]))
        masks.append(torch.transpose(single_mask(reduced_dims[i+1], 1, orig_dims[i+1],1 ), 0,1).reshape(orig_dims[i+1]))
    
    return(masks)


#################################
## Training loop with usual gradient descent
## Optimizer excluded to remove randomness
#################################

def training_loop(n_epochs, learning_rate, model, params, x_train, y_train, verbose=True):
    for epoch in range(1, n_epochs + 1):
        for p in params:
            if p.grad is not None: 
                p.grad.zero_()
        
        y_pred = model(x_train) 
        loss = loss_fn(y_pred, y_train)            
        loss.backward()
        
        with torch.no_grad(): 
            for p in params:
                p -= learning_rate * p.grad
                
        if verbose:
            if epoch ==1 or epoch % 500 == 0:
                print('Epoch %d, Loss %f' % (epoch, float(loss)))
            
    return model


#################################
## Training loop with projected gradient descent
## Optimizer excluded to remove randomness
#################################

def training_loop_proj_GD(n_epochs, learning_rate, model, params, original_dimensions, reduced_dimensions, \
                          x_train, y_train, verbose=True):
    
    param_masks = masks(reduced_dimensions, original_dimensions)
    
    for epoch in range(1, n_epochs + 1):
        for p in params:
            if p.grad is not None: 
                p.grad.zero_()
        
        y_pred = model(x_train) 
        loss = loss_fn(y_pred, y_train)            
        loss.backward()
        
        with torch.no_grad(): 
            for p,m in zip(params, param_masks):
                assert p.shape == m.shape, "Parameter shape and mask shape don't match"
                
                # Use the mask to zero out gradients that will not be updated
                p -= learning_rate * (p.grad * m)
                
        if verbose:
            if epoch ==1 or epoch % 500 == 0:
                print('Epoch %d, Loss %f' % (epoch, float(loss)))
            
    return model


#################################
## Training loop with ordinary gradient descent
## and a stopping value for the loss
#################################

def training_loop_with_stop(n_epochs, learning_rate, model, params, \
                            x_train, y_train, stopping_value=0.001, verbose=False):
    for epoch in range(1, n_epochs + 1):
        for p in params:
            if p.grad is not None: 
                p.grad.zero_()
        
        y_pred = model(x_train) 
        loss = loss_fn(y_pred, y_train)            
        loss.backward()
        
        with torch.no_grad(): 
            for p in params:
                p -= learning_rate * p.grad
                
        if loss < stopping_value:
            print('Epoch %d, Loss %f' % (epoch, float(loss)))
            return model
        
        if verbose:
            if epoch ==1 or epoch % 500 == 0:
                print('Epoch %d, Loss %f' % (epoch, float(loss)))
            
    return model