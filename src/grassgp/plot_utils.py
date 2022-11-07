import jax.numpy as np
from jax import vmap
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# import matplotlib.colors as colors
# import matplotlib.colorbar as colorbar
from scipy.interpolate import griddata

from grassgp.grassmann import grass_dist

# def plot_projected_data(X_projs, s, Ys, fig_size: tuple = (10, 6)):
#     n_locs = s.shape[0]
#     for i in range(n_locs):
#         fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)
#         ax.scatter(X_projs[i,:,0], Ys[:,i], alpha=0.5)
#         ax.set_xlabel('projected input')
#         if len(s.shape) == 1:
#             ax.set_title(f'Dataset at t = {s[i]:.2f}')
#         else:
#             ax.set_title(f'Dataset at spatial point = ({s[i, 0]:.2f}, {s[i, 1]:.2f})')
#     plt.show()

def plot_projected_data(X_projs, s, Ys, base_fig_size: tuple = (10, 6), cols:int = 2, ex=0.75, fontsize=15, axisfontsize=15):
    n_locs = s.shape[0]
    rows = n_locs // cols
    if n_locs % cols != 0:
        rows += 1
    
    w, h = base_fig_size
    fig_size = (w * cols + ex*w*(cols-1), h * rows + ex*h*(rows-1))
    fig, axs = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=fig_size, constrained_layout=True)
    # fig, axs = plt.subplots(rows, cols, sharex=True, sharey=True, constrained_layout=True)
    axs = axs.reshape(rows,cols)
    for i in range(n_locs):
        id_x, id_y = i // cols, i % cols
        # fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)
        axs[id_x,id_y].scatter(X_projs[i,:,0], Ys[:,i], alpha=0.5)
        axs[id_x,id_y].set_xlabel('projected input', fontdict={'fontsize': fontsize})
        axs[id_x,id_y].tick_params(axis='x', labelsize=axisfontsize)
        axs[id_x,id_y].tick_params(axis='y', labelsize=axisfontsize)
        if len(s.shape) == 1:
            axs[id_x,id_y].set_title(f'Dataset at t = {s[i]:.2f}', fontdict={'fontsize': fontsize})
        else:
            axs[id_x,id_y].set_title(f'Dataset at spatial point = ({s[i, 0]:.2f}, {s[i, 1]:.2f})', fontdict={'fontsize': fontsize})
    plt.show()


def flatten_samples(samples: dict, ignore: list = ['L_factor', 'sigmas', 'Omega', 'mat']):
    flattened_samples = {}
    for key, value in samples.items():
        if any([name in key for name in ignore]):
            continue
        
        key_shape = value.shape
        if len(key_shape) == 1:
            flattened_samples[f'{key}'] = value
        elif len(key_shape) == 2:
            for i in range(key_shape[1]):
                flattened_samples[f'{key}[{i}]'] = value[:, i]
        elif len(key_shape) == 3:
            for i in range(key_shape[1]):
                for j in range(key_shape[2]):
                    flattened_samples[f'{key}[{i},{j}]'] = value[:, i, j]
        elif len(key_shape) == 4:
            for i in range(key_shape[1]):
                for j in range(key_shape[2]):
                    for k in range(key_shape[3]):
                        flattened_samples[f'{key}[{i},{j},{k}]'] = value[:, i, j, k]
        else:
            raise ValueError(f'samples[{key}] contains values of dim > 3')
        
    my_samples = pd.DataFrame(flattened_samples)
    return my_samples

def pairplots(my_samples):
    fig_pair = sns.pairplot(my_samples, diag_kind='hist', corner='True')
    plt.show()


# def traceplots(my_samples, a=0.5):
#     for name in my_samples.keys():
#         plt.plot(my_samples[name], alpha=a)
#         plt.title(name)
#         plt.show()

def traceplots(my_samples, a=1.0, base_fig_size: tuple = (10, 6), cols:int = 3, ex=0.75, fontsize=20, axisfontsize=20):
    tot = len(my_samples.keys())
    rows = tot // cols
    if tot % cols != 0:
        rows += 1
    
    w, h = base_fig_size
    fig_size = (w * cols + ex*w*(cols-1), h * rows + ex*h*(rows-1))
    fig, axs = plt.subplots(rows, cols, sharex=True, figsize=fig_size, constrained_layout=True)
    axs = axs.reshape(rows,cols)
    for i, name in enumerate(my_samples.keys()):
        id_x, id_y = i // cols, i % cols
        axs[id_x,id_y].plot(my_samples[name], alpha=a)
        axs[id_x,id_y].set_title(name, fontdict={'fontsize': fontsize})
        # axs[id_x,id_y].tick_params(axis='x', labelsize=axisfontsize)
        axs[id_x,id_y].tick_params(axis='y', labelsize=axisfontsize)
    plt.show()
    

def plot_grids(X_fine, X, a=0.25):
    plt.scatter(X_fine[:,0],X_fine[:,1],c='b',alpha=0.25,label='fine')
    plt.scatter(X[:,0],X[:,1],c='r',label='coarse')
    plt.legend()
    plt.show()


## function to plot predictions at train locs against true data
# def plot_preds_train_locs(means, predictions, X, X_test, s, Ys, Ps, percentile_levels, fig_size=(8, 6)):
#     # compute average of means and percentiles for predictions
#     means_avged = np.mean(means, axis=0)
#     percentiles = np.percentile(predictions, np.array(percentile_levels), axis=0)
#     lower = percentiles[0,:,:]
#     upper = percentiles[1,:,:]

#     # Project the train and test data
#     X_projs = np.einsum('ij,ljk->lik', X, Ps)
#     X_test_projs = np.einsum('ij,ljk->lik', X_test, Ps)

#     # sort X_projs (and Ys according to this) and X_test_projs
#     assert X_projs.shape[-1] == 1
#     indices = X_projs.argsort(axis=1)
#     X_projs_sorted = np.take_along_axis(X_projs, indices, axis=1)
#     assert vmap(lambda sorted: (np.diff(sorted.flatten()) >= 0).all())(X_projs_sorted).all()
#     assert X_projs_sorted.shape == X_projs.shape

#     assert Ys.T.shape == X_projs.shape[:-1]
#     Ys_sorted = np.take_along_axis(Ys.T,indices[:,:,0],axis=1).T
#     assert Ys.shape == Ys_sorted.shape

#     assert X_test_projs.shape[-1] == 1
#     indices_test = X_test_projs.argsort(axis=1)
#     X_test_projs_sorted = np.take_along_axis(X_test_projs, indices_test, axis=1)
#     assert vmap(lambda sorted: (np.diff(sorted) >= 0).all())(X_test_projs_sorted).all()
#     assert X_test_projs_sorted.shape == X_test_projs.shape

#     # use sorting for test data to sort means_avged and percentiles
#     assert means_avged.T.shape == X_test_projs.shape[:-1]
#     means_avged_sorted = np.take_along_axis(means_avged.T, indices_test[:,:,0],axis=1).T
#     assert means_avged_sorted.shape == means_avged.shape
    
#     assert lower.T.shape == X_test_projs.shape[:-1]
#     assert upper.T.shape == X_test_projs.shape[:-1]
#     lower_sorted = np.take_along_axis(lower.T, indices_test[:,:,0], axis=1).T
#     upper_sorted = np.take_along_axis(upper.T, indices_test[:,:,0], axis=1).T

#     # plot the data
#     for i in range(s.shape[0]):
#         # make figure
#         fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)
        
#         # plot training data
#         ax.scatter(X_projs_sorted[i,:,0], Ys_sorted[:,i], color='red', label='data')
        
#         # plot confidence level for predictions
#         conf_size = percentile_levels[1] - percentile_levels[0]
#         ax.fill_between(X_test_projs_sorted[i,:,0], lower_sorted[:,i], upper_sorted[:,i], color='lightblue', alpha=0.75, label=f'{conf_size:.1f}% credible interval')
        
#         # plot mean prediction
#         ax.scatter(X_test_projs_sorted[i,:,0], means_avged_sorted[:,i],color='green',alpha=0.5, label='predictions')

#         if len(s.shape) == 1:
#             ax.set_title(f'Predictions at train time: t = {s[i]:.2f}')
#         else:
#             ax.set_title(f'Predictions at spatial train point: s = ({s[i,0]:.2f}, {s[i,1]:.2f})')
        
#         ax.legend()
        
#         plt.show()

def plot_preds_train_locs(means, predictions, X, X_test, s, Ys, Ps, percentile_levels, base_fig_size=(8, 6), cols:int = 2, ex=0.75, fontsize=15, axisfontsize=15, legendfontsize=12):
    # compute average of means and percentiles for predictions
    means_avged = np.mean(means, axis=0)
    percentiles = np.percentile(predictions, np.array(percentile_levels), axis=0)
    lower = percentiles[0,:,:]
    upper = percentiles[1,:,:]

    # Project the train and test data
    X_projs = np.einsum('ij,ljk->lik', X, Ps)
    X_test_projs = np.einsum('ij,ljk->lik', X_test, Ps)

    # sort X_projs (and Ys according to this) and X_test_projs
    assert X_projs.shape[-1] == 1
    indices = X_projs.argsort(axis=1)
    X_projs_sorted = np.take_along_axis(X_projs, indices, axis=1)
    assert vmap(lambda sorted: (np.diff(sorted.flatten()) >= 0).all())(X_projs_sorted).all()
    assert X_projs_sorted.shape == X_projs.shape

    assert Ys.T.shape == X_projs.shape[:-1]
    Ys_sorted = np.take_along_axis(Ys.T,indices[:,:,0],axis=1).T
    assert Ys.shape == Ys_sorted.shape

    assert X_test_projs.shape[-1] == 1
    indices_test = X_test_projs.argsort(axis=1)
    X_test_projs_sorted = np.take_along_axis(X_test_projs, indices_test, axis=1)
    assert vmap(lambda sorted: (np.diff(sorted) >= 0).all())(X_test_projs_sorted).all()
    assert X_test_projs_sorted.shape == X_test_projs.shape

    # use sorting for test data to sort means_avged and percentiles
    assert means_avged.T.shape == X_test_projs.shape[:-1]
    means_avged_sorted = np.take_along_axis(means_avged.T, indices_test[:,:,0],axis=1).T
    assert means_avged_sorted.shape == means_avged.shape
    
    assert lower.T.shape == X_test_projs.shape[:-1]
    assert upper.T.shape == X_test_projs.shape[:-1]
    lower_sorted = np.take_along_axis(lower.T, indices_test[:,:,0], axis=1).T
    upper_sorted = np.take_along_axis(upper.T, indices_test[:,:,0], axis=1).T

    # plot the data
    tot = s.shape[0]
    rows = tot // cols
    if tot % cols != 0:
        rows += 1
    
    w, h = base_fig_size
    fig_size = (w * cols + ex*w*(cols-1), h * rows + ex*h*(rows-1))
    fig, axs = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=fig_size, constrained_layout=True)
    axs = axs.reshape(rows,cols)
    for i in range(tot):
        id_x, id_y = i // cols, i % cols
        
        # plot training data
        axs[id_x,id_y].scatter(X_projs_sorted[i,:,0], Ys_sorted[:,i], color='red', label='data')
        
        # plot confidence level for predictions
        conf_size = percentile_levels[1] - percentile_levels[0]
        axs[id_x,id_y].fill_between(X_test_projs_sorted[i,:,0], lower_sorted[:,i], upper_sorted[:,i], color='lightblue', alpha=0.75, label=f'{conf_size:.1f}% credible interval')
        
        # plot mean prediction
        axs[id_x,id_y].scatter(X_test_projs_sorted[i,:,0], means_avged_sorted[:,i],color='green',alpha=0.5, label='predictions')

        if len(s.shape) == 1:
            axs[id_x,id_y].set_title(f'Predictions at train time: t = {s[i]:.2f}', fontdict = {'fontsize': fontsize})
        else:
            axs[id_x,id_y].set_title(f'Predictions at spatial train point: s = ({s[i,0]:.2f}, {s[i,1]:.2f})', fontdict={'fontsize': fontsize})
        
        axs[id_x,id_y].legend(prop = {'size' : legendfontsize})
        axs[id_x,id_y].tick_params(axis='x', labelsize=axisfontsize)
        axs[id_x,id_y].tick_params(axis='y', labelsize=axisfontsize)
        
    plt.show()


# def plot_grass_dists(Ps_samples, Ps, s, fig_size=(8, 6), a=0.75, quantity = 'Grass dists'):
#     assert Ps_samples.shape[1:] == Ps.shape
#     # create function to compute grass_dist at specific time for each sample
#     compute_dists_at_single_time = lambda i: vmap(lambda proj: grass_dist(Ps[i,:,:], proj[i,:,:]))(Ps_samples)

#     # use vmap to run this for all times
#     dists = vmap(compute_dists_at_single_time)(np.arange(s.shape[0]))

#     # plot dists
#     for i in range(s.shape[0]):
#         # make figure
#         fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)

#         # plot dists
#         ax.plot(dists[i,:], alpha=a)
        
#         if len(s.shape) == 1:
#             ax.set_title(f'{quantity} at time: t = {s[i]:.2f}')
#         else:
#             ax.set_title(f'{quantity} at spatial point: s = ({s[i,0]:.2f}, {s[i,1]:.2f})')

#         plt.show()
def plot_grass_dists(Ps_samples, Ps, s, base_fig_size=(8, 6), a=1.0, quantity = 'Grass dists', cols:int = 2, ex=0.75, fontsize=15, axisfontsize=15):
    assert Ps_samples.shape[1:] == Ps.shape
    # create function to compute grass_dist at specific time for each sample
    compute_dists_at_single_time = lambda i: vmap(lambda proj: grass_dist(Ps[i,:,:], proj[i,:,:]))(Ps_samples)

    # use vmap to run this for all times
    dists = vmap(compute_dists_at_single_time)(np.arange(s.shape[0]))

    # plot dists
    tot = s.shape[0]
    rows = tot // cols
    if tot % cols != 0:
        rows += 1
    
    w, h = base_fig_size
    fig_size = (w * cols + ex*w*(cols-1), h * rows + ex*h*(rows-1))
    fig, axs = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=fig_size, constrained_layout=True)
    axs = axs.reshape(rows,cols)
    for i in range(tot):
        # plot dists
        id_x, id_y = i // cols, i % cols
        
        axs[id_x, id_y].plot(dists[i,:], alpha=a)
        
        if len(s.shape) == 1:
            axs[id_x, id_y].set_title(f'{quantity} at time: t = {s[i]:.2f}', fontdict={'fontsize': fontsize})
        else:
            axs[id_x, id_y].set_title(f'{quantity} at spatial point: s = ({s[i,0]:.2f}, {s[i,1]:.2f})', fontdict={'fontsize': fontsize})
        
        axs[id_x,id_y].tick_params(axis='x', labelsize=axisfontsize)
        axs[id_x,id_y].tick_params(axis='y', labelsize=axisfontsize)

    plt.show()


def mat_heatmap(M, title = None, cbar = True, annot = False, cmap = cm.viridis):
    sns.heatmap(M,cbar=cbar,
                annot=annot,
                xticklabels=False,
                yticklabels=False,
                cmap=cmap)
    if title:
        plt.title(title)
    plt.show()

def plot_contourf(X, Y, P, cmap='plasma', P_est = None, P_est_name = 'MAP'):
    n = int(np.sqrt(X.shape[0]))
    X1 = X[:,0].reshape((n,n))
    X2 = X[:,1].reshape((n,n))
    Z = Y.reshape((n,n))
    plt.contourf(X1,X2,Z,cmap=cmap)
    dx = float(P[0])
    dy = float(P[1])
    plt.arrow(0,1.0,dx,dy,shape='full', label='true active subspace', lw=10,length_includes_head=True, head_width=.05, color='blue')
    if P_est is not None:
        dx_est = float(P_est[0])
        dy_est = float(P_est[1])
        plt.arrow(0.5,0,dx_est,dy_est,shape='full', label=f'{P_est_name} estimate of active subspace', lw=10,length_includes_head=True, head_width=.05, color='green')
    plt.legend()
    plt.colorbar()
    plt.show()

def plot_contourf_on_ax(fig, ax, X, Y, P, cmap='plasma', P_est = None, P_est_name = 'MAP', levels=None, require_color_bar = True):
    n = int(np.sqrt(X.shape[0]))
    X1 = X[:,0].reshape((n,n))
    X2 = X[:,1].reshape((n,n))
    Z = Y.reshape((n,n))
    if levels is not None:
        im = ax.contourf(X1, X2, Z, levels=levels, cmap=cmap)
    else:
        im = ax.contourf(X1, X2, Z, cmap=cmap)
    dx = float(P[0])
    dy = float(P[1])
    ax.arrow(0,1.0,dx,dy,shape='full', label='true active subspace', lw=10,length_includes_head=True, head_width=.05, color='blue')
    if P_est is not None:
        dx_est = float(P_est[0])
        dy_est = float(P_est[1])
        ax.arrow(0.5,0,dx_est,dy_est,shape='full', label=f'{P_est_name} estimate of active subspace', lw=10,length_includes_head=True, head_width=.05, color='green')
    ax.legend()
    if require_color_bar:
        fig.colorbar(im, ax=ax)

def compare_contours(s_fine, X_fine, X, Ys_fine, Ps_fine, Ps_est, means, predictions, cmap='plasma', fig_size=(20,30), n_levels=10):
    N_fine_sqrt = int(np.sqrt(X_fine.shape[0]))
    n_s_fine = s_fine.shape[0]
    n_s = Ps_est.shape[0]
    assert n_s_fine % n_s == 0
    s_gap = n_s_fine // n_s
    fig, axs = plt.subplots(n_s, 2, figsize=fig_size, constrained_layout=True)

    X1 = X_fine[:,0].reshape((N_fine_sqrt,N_fine_sqrt))
    X2 = X_fine[:,1].reshape((N_fine_sqrt,N_fine_sqrt))
    
    for i in range(0, n_s_fine, s_gap):
        interpolated_mean = np.array(griddata(X, means[:,i//2], (X1, X2), method='cubic'))
        inter = interpolated_mean[~np.isnan(interpolated_mean)]
        min_val = min(inter.min(), Ys_fine[:,i].min())
        max_val = max(inter.max(), Ys_fine[:,i].max())
        levels = np.linspace(min_val, max_val, n_levels)
        plot_contourf_on_ax(fig, axs[i//2, 0], X_fine, Ys_fine[:,i], Ps_fine[i,:,:], levels=levels, require_color_bar=False)
        axs[i//2, 0].set_title(f'Contour plot of training data at s = {s_fine[i]:.2f}')
        plot_contourf_on_ax(fig, axs[i//2, 1], X_fine, interpolated_mean, Ps_fine[i,:,:], P_est=Ps_est[i//2,:,:], levels=levels)
        axs[i//2, 1].set_title(f'Contour plot of mean MAP prediction at s = {s_fine[i]:.2f}')
    plt.show()
    
## function to plot grass predictions
# def plot_grass_preds(s, s_test, Ps_mean_samples, Ps_test_preds, Ps_orig, percentile_levels, fig_size=(8, 6), a_points=0.5, a_conf_int = 0.5):
#     assert len(s_test.shape) == 1
#     Ps_test_means_avged = np.mean(Ps_mean_samples, axis=0)
#     Ps_percentiles = np.percentile(Ps_test_preds, np.array(percentile_levels), axis=0)

#     # create plots
#     for ind in range(Ps_orig.shape[1]):
#         # make figure
#         fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)
        
#         # plot actual projection data
#         ax.scatter(s_test, Ps_orig[:, ind, 0], color='red', alpha=a_points)
#         ax.plot(s_test, Ps_orig[:, ind, 0], color='red', label='actual', alpha=a_points)
        
#         # conf interval
#         conf_size = percentile_levels[1] - percentile_levels[0]
#         ax.fill_between(s_test, Ps_percentiles[0, :, ind, 0], Ps_percentiles[1, :, ind, 0], color='lightblue', alpha=a_conf_int, label=f'{conf_size:.1f}% credible interval')
        
#         # plot mean pred
#         ax.scatter(s_test, Ps_test_means_avged[:, ind, 0], color='green', alpha=a_points)
#         ax.plot(s_test, Ps_test_means_avged[:, ind, 0], label='pred', color='green', alpha=a_points)

#         # add vertical lines at train times
#         lower_min = np.min(Ps_percentiles[0, :, ind, 0])
#         upper_max = np.max(Ps_percentiles[1, :, ind, 0])
#         # ax.vlines(s, -1, 1, colors='green', linestyles='dashed')
#         ax.vlines(s, lower_min, upper_max, colors='green', linestyles='dashed')

#         ax.set_title(f'{ind+1}th component of projection')
#         ax.legend()
#         plt.show()
def plot_grass_preds(s, s_test, Ps_mean_samples, Ps_test_preds, Ps_orig, percentile_levels, fig_size=(22, 8), a_points=0.5, a_conf_int = 0.5):
    assert len(s_test.shape) == 1
    Ps_test_means_avged = np.mean(Ps_mean_samples, axis=0)
    Ps_percentiles = np.percentile(Ps_test_preds, np.array(percentile_levels), axis=0)

    # create plots
    fig, axs = plt.subplots(1,2,figsize=fig_size, constrained_layout=True)
    for ind in range(Ps_orig.shape[1]):
        # make figure
        # plot actual projection data
        axs[ind].scatter(s_test, Ps_orig[:, ind, 0], color='red', alpha=a_points)
        axs[ind].plot(s_test, Ps_orig[:, ind, 0], color='red', label='actual', alpha=a_points)
        
        # conf interval
        conf_size = percentile_levels[1] - percentile_levels[0]
        axs[ind].fill_between(s_test, Ps_percentiles[0, :, ind, 0], Ps_percentiles[1, :, ind, 0], color='lightblue', alpha=a_conf_int, label=f'{conf_size:.1f}% credible interval')
        
        # plot mean pred
        axs[ind].scatter(s_test, Ps_test_means_avged[:, ind, 0], color='green', alpha=a_points)
        axs[ind].plot(s_test, Ps_test_means_avged[:, ind, 0], label='pred', color='green', alpha=a_points)

        # add vertical lines at train times
        lower_min = np.min(Ps_percentiles[0, :, ind, 0])
        upper_max = np.max(Ps_percentiles[1, :, ind, 0])
        # ax.vlines(s, -1, 1, colors='green', linestyles='dashed')
        axs[ind].vlines(s, lower_min, upper_max, colors='green', linestyles='dashed')

        axs[ind].set_title(f'{ind+1}th component of projection')
        axs[ind].legend()
    plt.show()



# def my_plot(X, X_test, s, Ys, Projs, Preds, percentile_levels, fig_size=(8, 6)):
#     # using all the projections in Projs, project the train and test data
#     X_projs = vmap(lambda proj: np.einsum('ij,ljk->lik', X, proj))(Projs)
#     X_test_projs = vmap(lambda proj: np.einsum('ij,ljk->lik', X_test, proj))(Projs)
#
#     N = Projs.shape[0]
#     n_s = s.shape[0]
#     fig, axs = plt.subplots(n_s, 1, figsize=fig_size, constrained_layout=True)
#     for i in range(N):
#         # get the current projected train/test data
#         train = X_projs[i]
#         test = X_test_projs[i]
#         pred = Preds[i]
#         # sort
#         indices = test.argsort(axis=1)
#         test_sorted = np.take_along_axis(test, indices, axis=1)
#         assert pred.T.shape == test.shape[:-1]
#         pred_sorted = np.take_along_axis(pred.T, indices[:,:,0], axis=1).T
#         for j in range(n_s):
#             # plot true data
#             if i == 0:
#                 axs[j].scatter(train[j].flatten(), Ys[:,j], color='red', label='data')
#             else:
#                 axs[j].scatter()
#
#             # plot current preds
#             if i == 0:
#                 axs[j].plot(test_sorted[j].flatten(), pred_sorted[:,j], color='blue', label='prediction')
#             else:
#                 axs[j].plot(test_sorted[j].flatten(), pred_sorted[:,j], color='blue')
#
#             if len(s.shape) == 1:
#                 axs[j].set_title(f'Predictions at train time: t = {s[j]:.2f}')
#             else:
#                 axs[j].set_title(f'Predictions at spatial train point: s = ({s[j,0]:.2f}, {s[j,1]:.2f})')
#             
#             axs[j].legend()
#
#     plt.show()


## function to plot predictions
# def plot_preds(means, predictions, X, X_test, s_test, Ys_test, Ps_fixed_test, percentile_levels, fig_size=(8, 6)):
#     # compute average of means and percentiles for predictions
#     means_avged = np.mean(means, axis=0)
#     percentiles = np.percentile(predictions, np.array(percentile_levels), axis=0)
#     lower = percentiles[0,:,:]
#     upper = percentiles[1,:,:]

#     # Project the train and test data
#     X_projs = np.einsum('ij,ljk->lik', X, Ps_fixed_test)
#     X_test_projs = np.einsum('ij,ljk->lik', X_test, Ps_fixed_test)

#     # sort X_projs (and Ys according to this) and X_test_projs
#     assert X_projs.shape[-1] == 1
#     indices = X_projs.argsort(axis=1)
#     X_projs_sorted = np.take_along_axis(X_projs, indices, axis=1)
#     assert vmap(lambda sorted: (np.diff(sorted.flatten()) >= 0).all())(X_projs_sorted).all()
#     assert X_projs_sorted.shape == X_projs.shape

#     # assert Ys.T.shape == X_projs.shape[:-1]
#     # Ys_sorted = np.take_along_axis(Ys.T,indices[:,:,0],axis=1).T
#     # assert Ys.shape == Ys_sorted.shape

#     assert X_test_projs.shape[-1] == 1
#     indices_test = X_test_projs.argsort(axis=1)
#     X_test_projs_sorted = np.take_along_axis(X_test_projs, indices_test, axis=1)
#     assert vmap(lambda sorted: (np.diff(sorted) >= 0).all())(X_test_projs_sorted).all()
#     assert X_test_projs_sorted.shape == X_test_projs.shape

#     # use sorting for test data to sort Ys_test, means_avged and percentiles
#     assert Ys_test.T.shape == X_test_projs.shape[:-1]
#     Ys_test_sorted = np.take_along_axis(Ys_test.T, indices_test[:,:,0],axis=1).T
#     assert Ys_test_sorted.shape == Ys_test.shape

#     assert means_avged.T.shape == X_test_projs.shape[:-1]
#     means_avged_sorted = np.take_along_axis(means_avged.T, indices_test[:,:,0],axis=1).T
#     assert means_avged_sorted.shape == means_avged.shape

#     assert lower.T.shape == X_test_projs.shape[:-1]
#     assert upper.T.shape == X_test_projs.shape[:-1]
#     lower_sorted = np.take_along_axis(lower.T, indices_test[:,:,0], axis=1).T
#     upper_sorted = np.take_along_axis(upper.T, indices_test[:,:,0], axis=1).T

#     # plot the data
#     for i in range(s_test.shape[0]):
#         # make figure
#         fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)

#         # plot training data
#         ax.scatter(X_projs_sorted[i,:,0], Ys_test_sorted[:,i], color='red', label='data')

#         # plot confidence level for predictions
#         conf_size = percentile_levels[1] - percentile_levels[0]
#         ax.fill_between(X_test_projs_sorted[i,:,0], lower_sorted[:,i], upper_sorted[:,i], color='lightblue', alpha=0.75, label=f'{conf_size:.1f}% credible interval')

#         # plot mean prediction
#         ax.scatter(X_test_projs_sorted[i,:,0], means_avged_sorted[:,i],color='green',alpha=0.5, label='predictions')

#         if len(s_test.shape) == 1:
#             ax.set_title(f'Predictions at train time: t = {s_test[i]:.2f}')
#         else:
#             ax.set_title(f'Predictions at spatial train point: s = ({s_test[i,0]:.2f}, {s_test[i,1]:.2f})')

#         ax.legend()

#         plt.show()

## function to plot predictions
def plot_preds(means, predictions, X, X_test, s_test, Ys_test, Ps_fixed_test, s, percentile_levels, base_fig_size=(8, 6), cols:int = 2, ex=0.75, fontsize=20, axisfontsize=20, legendfontsize=15):
    # compute average of means and percentiles for predictions
    means_avged = np.mean(means, axis=0)
    percentiles = np.percentile(predictions, np.array(percentile_levels), axis=0)
    lower = percentiles[0,:,:]
    upper = percentiles[1,:,:]

    # Project the train and test data
    X_projs = np.einsum('ij,ljk->lik', X, Ps_fixed_test)
    X_test_projs = np.einsum('ij,ljk->lik', X_test, Ps_fixed_test)

    # sort X_projs (and Ys according to this) and X_test_projs
    assert X_projs.shape[-1] == 1
    indices = X_projs.argsort(axis=1)
    X_projs_sorted = np.take_along_axis(X_projs, indices, axis=1)
    assert vmap(lambda sorted: (np.diff(sorted.flatten()) >= 0).all())(X_projs_sorted).all()
    assert X_projs_sorted.shape == X_projs.shape

    # assert Ys.T.shape == X_projs.shape[:-1]
    # Ys_sorted = np.take_along_axis(Ys.T,indices[:,:,0],axis=1).T
    # assert Ys.shape == Ys_sorted.shape

    assert X_test_projs.shape[-1] == 1
    indices_test = X_test_projs.argsort(axis=1)
    X_test_projs_sorted = np.take_along_axis(X_test_projs, indices_test, axis=1)
    assert vmap(lambda sorted: (np.diff(sorted) >= 0).all())(X_test_projs_sorted).all()
    assert X_test_projs_sorted.shape == X_test_projs.shape

    # use sorting for test data to sort Ys_test, means_avged and percentiles
    assert Ys_test.T.shape == X_test_projs.shape[:-1]
    Ys_test_sorted = np.take_along_axis(Ys_test.T, indices_test[:,:,0],axis=1).T
    assert Ys_test_sorted.shape == Ys_test.shape

    assert means_avged.T.shape == X_test_projs.shape[:-1]
    means_avged_sorted = np.take_along_axis(means_avged.T, indices_test[:,:,0],axis=1).T
    assert means_avged_sorted.shape == means_avged.shape

    assert lower.T.shape == X_test_projs.shape[:-1]
    assert upper.T.shape == X_test_projs.shape[:-1]
    lower_sorted = np.take_along_axis(lower.T, indices_test[:,:,0], axis=1).T
    upper_sorted = np.take_along_axis(upper.T, indices_test[:,:,0], axis=1).T

    # plot the data
    tot = s_test.shape[0]
    rows = tot // cols
    if tot % cols != 0:
        rows += 1
    
    w, h = base_fig_size
    fig_size = (w * cols + ex*w*(cols-1), h * rows + ex*h*(rows-1))
    fig, axs = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=fig_size, constrained_layout=True)
    axs = axs.reshape(rows,cols)
    for i in range(tot):
        id_x, id_y = i // cols, i % cols

        # plot training data
        axs[id_x, id_y].scatter(X_projs_sorted[i,:,0], Ys_test_sorted[:,i], color='red', label='data')

        # plot confidence level for predictions
        conf_size = percentile_levels[1] - percentile_levels[0]
        axs[id_x, id_y].fill_between(X_test_projs_sorted[i,:,0], lower_sorted[:,i], upper_sorted[:,i], color='lightblue', alpha=0.75, label=f'{conf_size:.1f}% credible interval')

        # plot mean prediction
        axs[id_x, id_y].scatter(X_test_projs_sorted[i,:,0], means_avged_sorted[:,i],color='green',alpha=0.5, label='predictions')

        if len(s_test.shape) == 1:
            if s_test[i] in s:
                axs[id_x, id_y].set_title(f'Predictions at train time: t = {s_test[i]:.2f}', fontdict={'fontsize': fontsize})
            else:
                axs[id_x, id_y].set_title(f'Predictions at test time: t = {s_test[i]:.2f}', fontdict={'fontsize': fontsize})   
        else:
            if s_test[i] in s:
                axs[id_x, id_y].set_title(f'Predictions at spatial train point: s = ({s_test[i,0]:.2f}, {s_test[i,1]:.2f})', fontdict={'fontsize': fontsize})
            else:
                axs[id_x, id_y].set_title(f'Predictions at spatial test point: s = ({s_test[i,0]:.2f}, {s_test[i,1]:.2f})', fontdict={'fontsize': fontsize})

        axs[id_x, id_y].legend(prop = {'size': legendfontsize})
        axs[id_x,id_y].tick_params(axis='x', labelsize=axisfontsize)
        axs[id_x,id_y].tick_params(axis='y', labelsize=axisfontsize)

    plt.show()

def plot_density(d, a: float = -10.0, b: float = 10.0, N: int = 100, fig_size=(8, 6)):
    x = np.linspace(a, b, N)
    y = np.exp(d.log_prob(x))
    fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)
    ax.plot(x, y)
    ax.grid()
    ax.set_xlabel('x')
    ax.set_ylabel('density')
    return fig, ax


def plot_densities(d_dict: dict, a: float = -10.0, b: float = 10.0, N: int = 100, fig_size=(8, 6)):
    x = np.linspace(a, b, N)
    fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)
    for title, d in d_dict.items():
        ax.plot(x, np.exp(d.log_prob(x)), label=title)

    ax.grid()
    ax.set_xlabel('x')
    ax.set_ylabel('density')
    ax.legend()
    return fig, ax
    

def plot_AS_dir_preds(Ps_preds, Ps_test, s_test, s, a=0.05, base_fig_size=(8,6), cols:int = 2, ex=0.75, fontsize=20, axisfontsize=20):
    tot = s_test.shape[0]
    rows = tot // cols
    if tot % cols != 0:
        rows += 1
    
    w, h = base_fig_size
    fig_size = (w * cols + ex*w*(cols-1), h * rows + ex*h*(rows-1))
    fig, axs = plt.subplots(rows, cols, sharex=True, sharey=True, figsize=fig_size, constrained_layout=True)
    axs = axs.reshape(rows,cols)
    for i in range(tot):
        id_x, id_y = i // cols, i % cols
        axs[id_x,id_y].set_xlim(-1.5,1.5)
        axs[id_x,id_y].set_ylim(-1.5,1.5)
        dx = float(Ps_test[i][0])
        dy = float(Ps_test[i][1])
        axs[id_x,id_y].arrow(0, 0,dx,dy,shape='full', label='true active subspace', lw=4, length_includes_head=True, head_width=.05, color='blue')
        for j in range(Ps_preds.shape[0]):
            dx_sample = float(Ps_preds[j, i][0])
            dy_sample = float(Ps_preds[j, i][1])
            axs[id_x, id_y].arrow(0, 0, dx_sample,dy_sample, shape='full', lw=1, length_includes_head=True, head_width=.05, color='red', alpha=a)
        
        axs[id_x,id_y].tick_params(axis='x', labelsize=axisfontsize)
        axs[id_x,id_y].tick_params(axis='y', labelsize=axisfontsize)
        
        if s_test[i] in s:
            axs[id_x, id_y].set_title(f"Predictions for active subspace direction at train time: t = {s_test[i]:.2f}")
        else:
            axs[id_x, id_y].set_title(f"Predictions for active subspace direction at test time: t = {s_test[i]:.2f}")

    plt.show()
    

def plot_fixed_x_preds_vs_time(means, preds, s_grid, s, x, Ys_fixed, percentile_levels = [2.5, 97.5], fig_size=(10,6)):
    means_avged = np.mean(means, axis=0)
    percentiles = np.percentile(preds, np.array(percentile_levels), axis=0)
    lower = percentiles[0,:,:]
    upper = percentiles[1,:,:]
    conf_size = percentile_levels[1] - percentile_levels[0]
    fig, ax = plt.subplots(figsize=fig_size, constrained_layout=True)
    ax.scatter(s, Ys_fixed[0,:], c='r', label='data')
    ax.plot(s_grid, means_avged[0,:], c='g', label='predicted mean')
    ax.fill_between(s_grid, lower[0,:], upper[0,:], color='lightblue', alpha=0.5,label=f'{conf_size:.1f}% credible interval')
    ax.set_title(f"Predictions for x = ({x.flatten()[0]:.2f},{x.flatten()[1]:.2f}) vs time")
    ax.legend()
    plt.show()

    
if __name__ == "__main__":
    import numpyro.distributions as dist
    bs = [0.1, 0.5, 1.0, 1.5, 2.0, 5.0]
    d_dict = {f"LogNormal(0.0,{b:0.1f})":dist.LogNormal(0.0, b) for b in bs}
    fig, ax = plot_densities(d_dict, N=1000)
    plt.show()
