# DEFINTIONS/CLASS FILE

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import grad
from torch.autograd import Variable
import matplotlib.pyplot as plt
import time
import copy
from scipy.integrate import odeint
import os
from tqdm import tqdm  # for progress bar
from IPython.display import clear_output

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Prevents crashes at plot stage
dtype = torch.float


# Define the sin() activation function
class mySin(torch.nn.Module):
    @staticmethod
    def forward(input):
        return torch.sin(input)


# Define some more general functions
def dfx(x, f):
    # Calculate the derivative with auto-differention
    return grad([f], [x], grad_outputs=torch.ones(x.shape, dtype=dtype), create_graph=True)[0]


def perturbPoints(grid, t0, tf, sig=0.5):
    #   stochastic perturbation of the evaluation points
    #   force t[0]=t0  & force points to be in the t-interval
    delta_t = grid[1] - grid[0]
    noise = delta_t * torch.randn_like(grid) * sig
    t = grid + noise
    # the .data method does
    t.data[2] = torch.ones(1, 1) * (-1)
    t.data[t < t0] = t0 - t.data[t < t0]
    t.data[t > tf] = 2 * tf - t.data[t > tf]
    t.data[0] = torch.ones(1, 1) * t0

    t.data[-1] = torch.ones(1, 1) * tf
    t.requires_grad = False
    # print(type(t.data))
    return t


def parametricSolutions(t, nn, t0, x1):
    # parametric solutions
    N1, N2 = nn(t)
    dt = t - t0
    #### THERE ARE TWO PARAMETRIC SOLUTIONS. Uncomment f=dt
    f = (1 - torch.exp(-dt)) * (1 - torch.exp(dt - 6))
    # f=dt
    psi_hat = x1 + f * N1
    return psi_hat


def hamEqs_Loss(t, psi, E, V):
    psi_dx = dfx(t, psi)
    psi_ddx = dfx(t, psi_dx)
    # f = -psi_ddx/2 + (-E+V)*psi
    f = -psi_ddx + (-E + V) * psi
    L = (f.pow(2)).mean();
    return L


def potential(Xs):
    # Gives the potential at each point
    # Takes in tensor of x points, gives back tensor of V at each point
    k0 = 1
    k1 = 1

    Xsnp = Xs.data.numpy()
    # Vnp = (k0*Xsnp**2)/2
    Vnp = k0 * Xsnp ** 2 + k1 * Xsnp ** 4
    Vtorch = torch.from_numpy(Vnp)
    return Vtorch


# Network class
class qNN1(torch.nn.Module):
    def __init__(self, D_hid=10):
        super(qNN1, self).__init__()

        # Define activation function
        # self.actF = torch.nn.Sigmoid()
        self.actF = mySin()

        # Looking at the net in the paper. the 1->\lambda stage contains a single arrow contributing 1X1 weight the
        # (x,\lambda) -> HL1 (Hidden layer one), has 2 nodes on left, D_hid on right, hence 2 X D_(hid)
        # arrows/parameters the HL1 -> HL2 contributes D_(hid) X D_(hid) parameters Finally, HL2 -> Output contributes
        # D_(hid) X 1 parameters.
        self.Ein = torch.nn.Linear(1, 1)
        self.Lin_1 = torch.nn.Linear(2, D_hid)
        self.Lin_2 = torch.nn.Linear(D_hid, D_hid)
        self.Lin_3 = torch.nn.Linear(D_hid, D_hid)
        self.out = torch.nn.Linear(D_hid, 1)

    def forward(self, t):
        In1 = self.Ein(torch.ones_like(t))
        L1 = self.Lin_1(
            torch.cat((t, In1), 1))  # cat joins the inputs of x (perturbations) and the 1 that is fed into the Ein
        # layer
        h1 = self.actF(L1)
        L2 = self.Lin_2(h1)
        h2 = self.actF(L2)
        # L3 = self.Lin_3(h2)
        # h3 = self.actF(L3)
        out = self.out(h2)
        return out, In1


def run_Scan_oscillator(t0, tf, x1, neurons, epochs, n_train, lr, minibatch_number=1):
    fc0 = qNN1(neurons)  # Creates a class member called fc0
    fc1 = 0
    betas = [0.999, 0.9999]
    optimizer = optim.Adam(fc0.parameters(), lr=lr, betas=betas)
    Loss_history = [];
    Llim = 1e+20
    En_loss_history = []
    boundary_loss_history = []
    nontriv_loss_history = []
    SE_loss_history = []
    Ennontriv_loss_history = []
    criteria_loss_history = []
    En_history = []
    EWall_history = []
    di = (None, 1e+20)
    # Ready to log the first 16 lots of "models". As in, save their weights/biases at special moments (when eigenvalues
    # are found). This is specifically related to loss changes.
    dic = {0: di, 1: di, 2: di, 3: di, 4: di, 5: di, 6: di, 7: di, 8: di, 9: di, 10: di, 11: di, 12: di, 13: di, 14: di,
           15: di, 16: di}

    grid = torch.linspace(t0, tf, n_train).reshape(-1, 1)

    ## TRAINING ITERATION
    TeP0 = time.time()
    walle = -2
    last_psi_L = 0
    # Note: tqdm wraps the iterable to provide a progress bar
    for tt in tqdm(range(epochs)):
        # adjusting learning rate at epoch 3e4
        # if tt == 3e4:
        #    optimizer = optim.Adam(fc0.parameters(), lr = 1e-2, betas = betas)
        # Perturbing the evaluation points & forcing t[0]=t0
        t = perturbPoints(grid, t0, tf, sig=.03 * tf)

        # BATCHING
        batch_size = int(n_train / minibatch_number)
        batch_start, batch_end = 0, batch_size

        idx = np.random.permutation(n_train)
        t_b = t[idx]
        t_b.requires_grad = True
        t_f = t[-1]
        t_f = t_f.reshape(-1, 1)
        t_f.requires_grad = True
        loss = 0.0

        for nbatch in range(minibatch_number):
            # batch time set
            t_mb = t_b[batch_start:batch_end]

            #  Network solutions
            nn, En = fc0(t_mb)

            En_history.append(En[0].data.tolist()[0])

            psi = parametricSolutions(t_mb, fc0, t0, x1)
            # - last_psi_L*torch.exp(-(torch.ones_like(t_mb)-1)**2/(2*(1/20)))
            # last_psi_L = parametricSolutions(torch.ones_like(t_mb),fc0,t0,x1).data.numpy()[0][0]
            # print(last_psi_L)
            Pot = potential(t_mb)
            Ltot = hamEqs_Loss(t_mb, psi, En, Pot)
            SE_loss_history.append(Ltot)

            criteria_loss = Ltot

            if tt % 1000 == 0:
                walle += 0.16
            # Adding regulation terms to loss
            Ltot += 1 / ((psi.pow(2)).mean() + 1e-6) + 1 / (En.pow(2).mean() + 1e-6) + torch.exp(-1 * En + walle).mean()
            En_loss_history.append(torch.exp(-1 * En + walle).mean())  #
            EWall_history.append(walle)

            nontriv_loss_history.append(1 / ((psi.pow(2)).mean() + 1e-6))  #
            Ennontriv_loss_history.append(1 / (En.pow(2).mean() + 1e-6))  #
            # OPTIMIZER
            Ltot.backward(retain_graph=False);  # True
            optimizer.step();
            loss += Ltot.data.numpy()
            optimizer.zero_grad()

            batch_start += batch_size
            batch_end += batch_size

        # keep the loss function history
        Loss_history.append(loss)

        # Keep the best model (lowest loss) by using a deep copy
        # Check what occurs at the next local loss minimum? If each successive eigenvalue incurs
        # a smaller loss, then this makes sense
        if criteria_loss < Llim:
            fc1 = copy.deepcopy(fc0)
            Llim = criteria_loss

        # This part retains the overall wavefunction when an eigenvalue plateau is found
        E_bin = abs(En[0].data.tolist()[0] // 2)
        if criteria_loss < dic[E_bin][1]:
            dic[E_bin] = (copy.deepcopy(fc0), criteria_loss)

    TePf = time.time()
    runTime = TePf - TeP0
    loss_histories = (
        Loss_history, boundary_loss_history, nontriv_loss_history, SE_loss_history, Ennontriv_loss_history,
        En_loss_history,
        criteria_loss_history, fc0, En_history, EWall_history, dic)
    return fc1, loss_histories, runTime
