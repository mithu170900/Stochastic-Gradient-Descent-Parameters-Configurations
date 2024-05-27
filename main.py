#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 22:26:26 2023

@author: minhthubui
"""
import numpy as np
from autograd import grad
import autograd.numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp


seed = 42   # seed for pseudorandom number generator
N = 10000    # number of obervations
D = 10       # features dimensionality 

def _robust_loss(psi, beta, nu, Y, Z):
    scaled_sq_errors = np.exp(-2*psi)  * (np.dot(Z, beta) - Y)**2
    if nu == np.inf:
        return scaled_sq_errors/2 + psi
    return (nu + 1)/2 * np.log(1 + scaled_sq_errors / nu) + psi


def make_sgd_robust_loss(Y, Z, nu):
    N = Y.size
    loss = lambda param: np.mean(_robust_loss(param[0], param[1:], nu, Y, Z)) + np.sum(param**2)/(2*N)
    sgd_loss = lambda param, inds: np.mean(_robust_loss(param[0], param[1:], nu, Y[inds], Z[inds])) + np.sum(param**2)/(2*N)
    grad_sgd_loss = grad(sgd_loss)
    return loss, sgd_loss, grad_sgd_loss


def generate_data(N, D, seed):
    rng = np.random.default_rng(seed)
    # generate multivariate t covariates with 10 degrees
    # of freedom and non-diagonal covariance
    t_dof = 10
    locs = np.arange(D).reshape((D,1))
    cov = (t_dof - 2) / t_dof * np.exp(-(locs - locs.T)**2/4)
    Z = rng.multivariate_normal(np.zeros(D), cov, size=N)
    Z *= np.sqrt(t_dof / rng.chisquare(t_dof, size=(N, 1)))
    # generate responses using regression coefficients beta = (1, 2, ..., D)
    # and t-distributed noise
    true_beta = np.arange(1, D+1)
    Y = Z.dot(true_beta) + rng.standard_t(t_dof, size=N)
    # for simplicity, center responses
    Y = Y - np.mean(Y)
    return true_beta, Y, Z


def run_SGD(grad_loss, epochs, init_param, init_stepsize, stepsize_decayrate, batchsize, N):
    K = (epochs * N) // batchsize # Count the number of iterations so N //batchsize is the number of iterations per epoch
    D = len(init_param)
    paramiters = np.zeros((K+1,D))
    paramiters[0] = init_param
    for k in range(K):
        inds = np.random.choice(N, batchsize)
        stepsize = init_stepsize / (k+1)**stepsize_decayrate
        paramiters[k+1] = paramiters[k] - stepsize*grad_loss(paramiters[k], inds)
    return paramiters


def plot_iterates_and_squared_errors(paramiters, true_beta, opt_param, skip_epochs, epochs, N, batchsize, include_psi=True):
    D = true_beta.size
    param_names = [r'$\beta_{{{}}}$'.format(i) for i in range(D)]
    if include_psi:
        param_names = [r'$\psi$'] + param_names
    else:
        paramiters = paramiters[:,1:]
        opt_param = opt_param[1:]
    skip_epochs = 0
    skip_iters = int(skip_epochs*N//batchsize)
    xs = np.linspace(skip_epochs, epochs, paramiters.shape[0] - skip_iters)
    plt.plot(xs, paramiters[skip_iters:]);
    plt.plot(np.array(D*[[xs[0], xs[-1]]]).T, np.array([true_beta,true_beta]), ':')
    plt.xlabel('epoch')
    plt.ylabel('parameter value')
    plt.legend(param_names, bbox_to_anchor=(0,1.02,1,0.2), loc='lower left',
               mode='expand', borderaxespad=0, ncol=4, frameon=False)
    sns.despine()
    plt.show()
    plt.plot(xs, np.linalg.norm(paramiters - opt_param[np.newaxis,:], axis=1)**2)
    plt.xlabel('epoch')
    plt.ylabel(r'$\|x_k - x_{\star}\|_2^2$')
    plt.yscale('log')
    sns.despine()
    plt.show()

################################################
## (A) The effect of the number of iterations:##
################################################
batchsize = 10       # batch size
init_param = np.zeros(D+1)  
init_stepsize = 0.2
stepsize_node = 5
k = 0.5
alpha = 0.51
stepsize_decayrate = 0
stepsize_decayrate_3 = alpha
nu = 5
true_beta, Y, Z = generate_data(N, D, seed)
loss, sgd_loss, grad_sgd_loss = make_sgd_robust_loss(Y, Z, nu)
true_val = np.concatenate(([0.0], true_beta)) 
opt_param = sp.optimize.minimize(loss, true_val).x
epochs = 100

def plot_K_and_squared_error(paramiters_last_iterate, paramiters_iterate_averaged, paramiters_last_iterate_decreasing, true_val, opt_param, skip_epochs, epochs, N, batchsize, include_psi):
    K = (epochs * N) // batchsize
    skip_iters = int(skip_epochs*N//batchsize)
    xs = np.linspace(0, K, paramiters_last_iterate.shape[0] - skip_iters)
    plt.plot(xs,np.linalg.norm(paramiters_last_iterate - opt_param[np.newaxis,:], axis=1)**2, label='SGD Last Iterate')
    plt.plot(xs,np.linalg.norm(paramiters_iterate_averaged - opt_param[np.newaxis,:], axis=1)**2, label='SGD Iterate Averaged')
    plt.plot(xs,np.linalg.norm(paramiters_last_iterate_decreasing - opt_param[np.newaxis,:], axis=1)**2, label='SGD Iterate Decreasing')
    plt.xlabel('Number of iterations')
    plt.ylabel(r'$\|x_k - x_{\star}\|_2^2$')
    plt.yscale('log')
    plt.legend(loc='upper right')
    plt.title("Number of Iterations and Squared Error, Epochs = " +str(epochs))
    sns.despine()
    plt.show()


paramiters_last_iterate = run_SGD(grad_sgd_loss, epochs, init_param, init_stepsize, stepsize_decayrate, batchsize, N)
paramiters_iterate_averaged = np.array([np.mean(paramiters_last_iterate[i//2:i+1], axis=0) for i in range(paramiters_last_iterate.shape[0])])
paramiters_last_iterate_decreasing = run_SGD(grad_sgd_loss, epochs, init_param, stepsize_node, stepsize_decayrate_3, batchsize, N)

plot_K_and_squared_error(paramiters_last_iterate, paramiters_iterate_averaged, paramiters_last_iterate_decreasing, true_beta, opt_param, skip_epochs = 0, epochs=epochs, N=N, batchsize=batchsize, include_psi=True)
plot_iterates_and_squared_errors(paramiters_last_iterate, true_beta, opt_param, skip_epochs = 0, epochs = epochs, N = N, batchsize = batchsize, include_psi=True)
plot_iterates_and_squared_errors(paramiters_last_iterate_decreasing, true_beta, opt_param, skip_epochs = 0, epochs = epochs, N = N, batchsize = batchsize, include_psi=True)



################################################
##    (B) The effect of initialization:       ##
################################################
epochs = 10 #number of epochs
batchsize = 10    # batch size

change_updates = [np.linspace(0, 10, 11), np.linspace(0, 50, 11), np.linspace(0, 100, 11)]
x0_lists = []
for change_update in change_updates:
    x0_list = opt_param + change_update**2
    x0_lists.append(x0_list)

init_params_list = x0_lists


paramiters_last_iterate = run_SGD(grad_sgd_loss, epochs, init_param, init_stepsize, stepsize_decayrate, batchsize, N)
paramiters_iterate_averaged = np.array([np.mean(paramiters_last_iterate[i//2:i+1], axis=0) for i in range(paramiters_last_iterate.shape[0])])
paramiters_last_iterate_decreasing = run_SGD(grad_sgd_loss, epochs, init_param, stepsize_node, stepsize_decayrate_3, batchsize, N)

s=[np.linalg.norm(paramiters_last_iterate_decreasing[i,:] - opt_param[np.newaxis,:], axis=1)**2 for i in range(paramiters_last_iterate_decreasing.shape[0]//2, paramiters_last_iterate_decreasing.shape[0])]
s_mean=np.mean(s)

def plot_K_and_squared_error_diff_init_param(paramiters_last_iterate, paramiters_iterate_averaged, paramiters_last_iterate_decreasing, true_beta, opt_param, skip_epochs, epochs, N, batchsize, include_psi):
    K = (epochs * N) // batchsize
    skip_iters = int(skip_epochs*N//batchsize)
    xs = np.linspace(0, K, paramiters_last_iterate.shape[0] - skip_iters)
    plt.plot(xs, np.linalg.norm(paramiters_last_iterate - opt_param[np.newaxis,:], axis=1)**2, label = "Last Iterate")
    plt.plot(xs, np.linalg.norm(paramiters_iterate_averaged - opt_param[np.newaxis,:], axis=1)**2, label = "Iterate Averaged")
    plt.plot(xs, np.linalg.norm(paramiters_last_iterate_decreasing - opt_param[np.newaxis,:], axis=1)**2, label = "Iterate Decreasing")
    plt.xlabel('Number of iterations')
    plt.ylabel(r'$\|x_k - x_{\star}\|_2^2$')
    plt.yscale('log')
    plt.legend(loc='upper right')
    #plt.title("Different Initializations and Squared Error")
    sns.despine()
    plt.show()

for init_param in init_params_list:
    plot_K_and_squared_error_diff_init_param(paramiters_last_iterate, paramiters_iterate_averaged, paramiters_last_iterate_decreasing, true_beta, opt_param, skip_epochs = 0, epochs=epochs, N=N, batchsize=batchsize, include_psi=True)


################################################
##    (C) The effect of step sizes:           ##
################################################   
init_stepsize = 0.2
epochs = 10 #number of epochs
batchsize = 10   # batch size
stepsize_node = 5
k = 0.5
alpha = 0.51
stepsize_decayrate = 0
stepsize_decayrate_3 = alpha
paramiters_last_iterate_list = []
paramiters_iterate_averaged_list = []
skip_epochs = 0
skip_iters = int(skip_epochs*N//batchsize)

# Changing initial stepsize for constant case:
init_stepsizes_constant = [0.2, 0.9]
for init_stepsize in init_stepsizes_constant:
    paramiters_last_iterate = run_SGD(grad_sgd_loss, epochs, init_param, init_stepsize, stepsize_decayrate, batchsize, N)
    paramiters_iterate_averaged = np.array([np.mean(paramiters_last_iterate[i//2:i+1], axis=0) for i in range(paramiters_last_iterate.shape[0])])   
    xs = np.linspace(skip_epochs, epochs, paramiters_last_iterate.shape[0] - skip_iters)
    plt.plot(xs, np.linalg.norm(paramiters_last_iterate - opt_param[np.newaxis,:], axis=1)**2,label = "Stepsize =" + str(init_stepsize))
    #plt.plot(xs, np.linalg.norm(paramiters_iterate_averaged - opt_param[np.newaxis,:], axis=1)**2,label = "Stepsize =" + str(init_stepsize))
    plt.xlabel('Epochs')
    plt.ylabel(r'$\|x_k - x_{\star}\|_2^2$')
    plt.yscale('log')
    plt.legend(loc='upper right')
    #plt.title("Different stepsizes and Squared Error for Accuracy")
    sns.despine()
    plt.show()

for init_stepsize in init_stepsizes_constant:
    paramiters_last_iterate = run_SGD(grad_sgd_loss, epochs, init_param, init_stepsize, stepsize_decayrate, batchsize, N)
    paramiters_iterate_averaged = np.array([np.mean(paramiters_last_iterate[i//2:i+1], axis=0) for i in range(paramiters_last_iterate.shape[0])])   
    xs = np.linspace(skip_epochs, epochs, paramiters_last_iterate.shape[0] - skip_iters)
    #plt.plot(xs, np.linalg.norm(paramiters_last_iterate - opt_param[np.newaxis,:], axis=1)**2,label = "Stepsize =" + str(init_stepsize))
    plt.plot(xs, np.linalg.norm(paramiters_iterate_averaged - opt_param[np.newaxis,:], axis=1)**2,label = "Stepsize =" + str(init_stepsize))
plt.xlabel('Epochs')
plt.ylabel(r'$\|x_k - x_{\star}\|_2^2$')
plt.yscale('log')
plt.legend(loc='upper right')
#plt.title("Different stepsizes for Iterate Averaged")
sns.despine()
plt.show()

# Decreasing stepsize for constant size:
    # 1) Alpha constant, stepsize_node changes
stepsize_nodes = [5, 10, 100]
k = 0.5
alpha = 0.51
stepsize_decayrate = 0
stepsize_decayrate_3 = alpha

for stepsize_node in stepsize_nodes:
    paramiters_last_iterate_decreasing = run_SGD(grad_sgd_loss, epochs, init_param, stepsize_node, stepsize_decayrate_3, batchsize, N)
    xs = np.linspace(skip_epochs, epochs, paramiters_last_iterate.shape[0] - skip_iters)
    plt.plot(xs, np.linalg.norm(paramiters_last_iterate_decreasing - opt_param[np.newaxis,:], axis=1)**2,label = "Eta node =" + str(stepsize_node))
plt.xlabel('Epochs')
plt.ylabel(r'$\|x_k - x_{\star}\|_2^2$')
plt.yscale('log')
plt.legend(loc='upper right')
plt.title("Different stepsizes and Squared Error for Accuracy of SGD Decreasing Stepsize")
sns.despine()
plt.show()

    # 2) Alpha changes, stepsize_node remains
stepsize_nodes = 5
k = 0.5
alphas = [0.01, 0.2, 0.51, 0.7, 1]
stepsize_decayrate = 0
stepsize_decayrate_3 = alpha

for alpha in alphas:
    stepsize_decayrate_3 = alpha
    paramiters_last_iterate_decreasing = run_SGD(grad_sgd_loss, epochs, init_param, stepsize_node, stepsize_decayrate_3, batchsize, N)
    xs = np.linspace(skip_epochs, epochs, paramiters_last_iterate.shape[0] - skip_iters)
    plt.plot(xs, np.linalg.norm(paramiters_last_iterate_decreasing - opt_param[np.newaxis,:], axis=1)**2,label = "Alpha=" + str(alpha))
    plt.xlabel('Epochs')
    plt.ylabel(r'$\|x_k - x_{\star}\|_2^2$')
    plt.yscale('log')
    plt.legend(loc='upper right')
    #plt.title("Different Alphas and Squared Error for Accuracy of SGD Decreasing Stepsize")
    sns.despine()
    plt.show()



################################################
##    (D) The effect of gradient noises:      ##
################################################
batchsizes = [5, 10, 100]       # batch size
init_param = np.zeros(D+1)  
init_stepsize = 0.2
stepsize_node = 5
k = 0.5
alpha = 0.51
stepsize_decayrate = 0
stepsize_decayrate_3 = alpha
nu = 5
epochs = 100

def plot_batchsize_and_squared_error(paramiters_last_iterate, paramiters_iterate_averaged, paramiters_last_iterate_decreasing, true_beta, opt_param, skip_epochs, epochs, N, batchsize, include_psi):
    skip_iters = int(skip_epochs*N//batchsize)
    xs = np.linspace(skip_epochs, epochs, paramiters_last_iterate.shape[0] - skip_iters)
    plt.plot(xs, np.linalg.norm(paramiters_last_iterate - opt_param[np.newaxis,:], axis=1)**2, label = "Last Iterate")
    plt.plot(xs, np.linalg.norm(paramiters_iterate_averaged - opt_param[np.newaxis,:], axis=1)**2,label = "Iterate Averaged")
    plt.plot(xs, np.linalg.norm(paramiters_last_iterate_decreasing - opt_param[np.newaxis,:], axis=1)**2, label = "Iterate Decreasing")
    plt.xlabel('epoch')
    plt.ylabel(r'$\|x_k - x_{\star}\|_2^2$')
    plt.yscale('log')
    plt.legend(loc='upper right')
    #plt.title("Different batchsizes and Squared Error, Batchsize = "+str(batchsize))
    sns.despine()
    plt.show()
    
for batchsize in batchsizes:
    paramiters_last_iterate = run_SGD(grad_sgd_loss, epochs, init_param, init_stepsize, stepsize_decayrate, batchsize, N)
    paramiters_iterate_averaged = np.array([np.mean(paramiters_last_iterate[i//2:i+1], axis=0) for i in range(paramiters_last_iterate.shape[0])])
    paramiters_last_iterate_decreasing = run_SGD(grad_sgd_loss, epochs, init_param, stepsize_node, stepsize_decayrate_3, batchsize, N)

    plot_batchsize_and_squared_error(paramiters_last_iterate, paramiters_iterate_averaged, paramiters_last_iterate_decreasing, true_beta, opt_param, skip_epochs = 0, epochs = epochs, N = N, batchsize = batchsize, include_psi = True)



################################################
##        (E) The effect of loss:             ##
################################################  
nus = [5, 500, np.inf]
for nu in nus:
    loss, sgd_loss, grad_sgd_loss = make_sgd_robust_loss(Y, Z, nu)
    paramiters_last_iterate = run_SGD(grad_sgd_loss, epochs, init_param, init_stepsize, stepsize_decayrate, batchsize, N)
    paramiters_iterate_averaged = np.array([np.mean(paramiters_last_iterate[i//2:i+1], axis=0) for i in range(paramiters_last_iterate.shape[0])])
    paramiters_last_iterate_decreasing = run_SGD(grad_sgd_loss, epochs, init_param, stepsize_node, stepsize_decayrate_3, batchsize, N)

    skip_epochs = 0
    skip_iters = int(skip_epochs*N//batchsize)
    xs = np.linspace(skip_epochs, epochs, paramiters_last_iterate.shape[0] - skip_iters)

    plt.plot(xs, np.linalg.norm(paramiters_last_iterate - opt_param[np.newaxis,:], axis=1)**2,label = "Last Iterate")
    plt.plot(xs, np.linalg.norm(paramiters_iterate_averaged - opt_param[np.newaxis,:], axis=1)**2,label = "Iterate Averaged")
    plt.plot(xs, np.linalg.norm( paramiters_last_iterate_decreasing - opt_param[np.newaxis,:], axis=1)**2,label = "Iterate Decreasing")
    plt.xlabel('epoch')
    plt.ylabel(r'$\|x_k - x_{\star}\|_2^2$')
    plt.yscale('log')
    plt.legend(loc='upper right')
    #plt.title("nu = " + str(nu) + " and Squared Error for Accuracy")
    sns.despine()
    plt.show()