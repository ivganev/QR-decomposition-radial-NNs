# The QR Decomposition for radial neural networks

This repository accompanies the paper "The QR decomposition for radial neural networks".

## The file source.py

### The representation class

In the file source.py, we create a class $\texttt{representation}$ for representations of the neural quiver <img src="https://render.githubusercontent.com/render/math?math=\mathscr{Q}_L>, which is the following quiver:
<img src="neural-quiver.png" alt="drawing" width="500"/>
with $L + 1$ vertices in the top row and a 'bias vertex' at the bottom. We only consider dimension vectors whose value at the bias vertex is equal to $1$, so a dimension vector for $\mathscr{Q}_L$ refers to a tuple $\mathbf{n} = (n_0, n_1, \dots, n_L)$. For any such dimension vector, the vector space of representations of the neural quiver $\mathscr{Q}_L$ can be identified with a direct sum of matrix spaces:
$$\mathsf{Rep}(\mathscr{Q}_L, \mathbf{n})  \simeq  \bigoplus_{i=1}^L \mathrm{Hom}(\mathbb{R}^{1 +n_{i-1}}, \mathbb{R}^{n_i})$$
where, $\mathrm{Hom}(\mathbb{R}^{1 +n_{i-1}}, \mathbb{R}^{n_i})$ denotes the space of $n_i \times (1+n_{i-1}$ matrices, i.e. the space of linear maps from $\mathbb{R}^{n_i}$ to $\mathbb{R}^{1 + n_{i-1}}$. 

Let  $\mathbf{W} = (W_i)_{i=1}^L$ be a representation of the neural quiver with dimension vector $\mathbf{n}$, so each $W_i$ is an $n_i \times (1 + n_{i-1})$ matrix. The class $\texttt{representation}$ includes methods to compute:

-- The QR decomposition of the representation $\mathbf{W}$, so $$\mathbf{W} = \mathbf{Q} \cdot \mathbf{R} + \mathbf{U}.$$

-- The reduced representation $\mathbf{R}$.

-- The transformed representation $\mathbf{Q}^{-1} \cdot \mathbf{W}$.

### Radial neural networks

Continuing in the $\texttt{source.py}$, we use PyTorch modules to implement radial activation functions in the class $\texttt{RadAct(nn.Module)}$, where there is an option to add a shift. We then implement radial neural networks in the class $\texttt{RadNet(nn.Module)}$, which has methods to:

-- Set the weights.

-- Export the weights $\mathbf{W}$.

-- Export the reduced network (with weights $\mathbf{R}$ from the QR decomposition for $\mathbf{W}$).

-- Export the transformed network (with weights $\mathbf{Q}^{-1} \cdot \mathbf{W}$ from the QR decomposition for $\mathbf{W}$).

### Training

For training models, we have three different types of training loops:

-- $\texttt{training_loop}$, which is the most basic training loop with usual gradient descent. There is no optimizer in order to remove randomness. 

-- $\texttt{training_loop_proj_GD}$, which uses projected gradient descent. We define the appropriate masks in order to implement this properly. 

-- $\texttt{training_loop_with_stop}$, which impelments usual gradient descent, but with a stopping value for the loss function

## The file $\texttt{script-experiment-1.py}$

In this experiment, we instantiate a radial neural network with weights $\mathbf{W}$ and show that projected gradient descent on the transformed network (with weights $\mathbf{Q}^{-1} \cdot \mathbf{W}$) matches usual gradient descent on the reduced network (with weights $\mathbf{R}$). Specifically, the values of the loss function are the same in both training regimes, epoch by epoch. 


## The file $\texttt{script-experiment-2.py}$

In this experiment, we instantiate a radial neural network with weights $\mathbf{W}$ and a somewhat large dimension vector. We train both the original model and the reduced model (with weights $\mathbf{R}$ coming from the QR decomposition of $\mathbf{W}$) with usual gradient descent using a stopping value for the loss function. We show that the reduced model achieves this low value for the loss function after less time (albeit after more epochs).

