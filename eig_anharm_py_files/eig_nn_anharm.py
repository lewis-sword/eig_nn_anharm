# NETWORK TRAIN AND PREDICT FILE
# Importing all network definitions: loss, activation, derivative, random x data creation e.t.c. from the
# defs_eig_nn_anharm.py script
from defs_eig_nn_anharm import *
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Prevents crashes at plot stage
dtype = torch.float

# for the plots
plt.rc('xtick', labelsize=16)
plt.rcParams.update({'font.size': 16})

# Checking network paramaters
netTest = qNN1(5)
for name, param in netTest.named_parameters():
    if param.requires_grad:
        print(name, param.data)

# Training the model

# Note: altering the range of x, i.e. values of t0 and tf, requires the associated change to the
# parametricSolutions function in def_eig_nn_anharm
t0 = -3
tf = 3
xBC1 = 0.

# Parameter values
n_train, neurons, epochs, lr, mb = 600, 50, int(120000), 9e-3, 1
model1, loss_hists1, runTime1 = run_Scan_oscillator(t0, tf, xBC1, neurons, epochs, n_train, lr, mb)

# Loss function plot
print('Training time (minutes):', runTime1 / 60)
plt.figure(figsize=(8, 6))
plt.plot(loss_hists1[0][:], '-b')
plt.tight_layout()
plt.ylabel('Total Loss')
plt.xlabel('Epochs')
plt.show()

# Test predicted solutions
nTest = n_train
tTest = torch.linspace(t0 - .1, tf + .1, nTest)
tTest = tTest.reshape(-1, 1)
tTest.requires_grad = True
t_net = tTest.detach().numpy()
psi = parametricSolutions(tTest, model1, t0, xBC1)
psi = psi.data.numpy()

plt.figure(figsize=(8, 6))
plt.plot(loss_hists1[8][:])
plt.tight_layout()
plt.ylabel('Model Energy History')
plt.xlabel('Epochs')
plt.show()

# Stored eigenvalues

# log_hists1[10][i] access dic[i] = (copy.deepcopy(fc0), criteria_loss), hence calling [0] on it, get the copy of
# fc0, at that time. This must contain the weights of the model at that point, and calling forward(tTest) on it,
# does a forward pass through the network to generate the network output from the forward definition, which is tuple
# (out, In1). Taking the [1] accesses the eigenvalue at all x points in the range, while taking [0] would evaluate
# the wavefunction at all points in the range. We finally take the [0] component of log_hists1[10][i], as the weights
# that generate the eigenvalue are invariant under x value (x isn't involved) Note: that we have chosen i's that go
# in twos in log_hists1[10][i]. This seems to be some degeneracy in the recording of the eigenvalues and eigenstates.
print(loss_hists1[10][0][0].forward(tTest)[1][0])
print(loss_hists1[10][2][0].forward(tTest)[1][0])
print(loss_hists1[10][4][0].forward(tTest)[1][0])
print(loss_hists1[10][6][0].forward(tTest)[1][0])

# Plotting wavefunctions of associated eigenvalues found
plt.figure(figsize=(8, 6))
psi_0to10 = parametricSolutions(tTest, loss_hists1[10][0][0], t0, xBC1)
psi_10to20 = parametricSolutions(tTest, loss_hists1[10][2][0], t0, xBC1)
psi_20to30 = parametricSolutions(tTest, loss_hists1[10][4][0], t0, xBC1)
psi_30to40 = parametricSolutions(tTest, loss_hists1[10][6][0], t0, xBC1)

plt.plot(t_net, 1 * psi_0to10.data.numpy() / np.max(np.abs(psi_0to10.data.numpy())), '-b', linewidth=1, label='n = 1')
plt.plot(t_net, 1 * psi_10to20.data.numpy() / np.max(np.abs(psi_10to20.data.numpy())), '-y', linewidth=1, label='n = 2')
plt.plot(t_net, 1 * psi_20to30.data.numpy() / np.max(np.abs(psi_20to30.data.numpy())), '-g', linewidth=1, label='n = 3')
plt.plot(t_net, 1 * psi_30to40.data.numpy() / np.max(np.abs(psi_30to40.data.numpy())), '-r', linewidth=1, label='n = 4')
plt.legend()
plt.ylabel('$\\psi(x)$')
plt.xlabel('x')
plt.show()
