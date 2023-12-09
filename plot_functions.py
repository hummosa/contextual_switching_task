import matplotlib.pyplot as plt
import numpy as np
from utils.plot_utils import *
import torch
import matplotlib.transforms as mtransforms
from collections import defaultdict

obs_color = 'tab:grey'
preds_color = 'tab:red'

figure_dpi = 600

def plot_task_and_hierarchies_illustration(testing_memory_buffer, testing_env, testing_losses, config, x1=50, x2=np.inf, latent_dim = 0 ):
    # get the obs from the memory buffer
    logger = testing_memory_buffer
    env = testing_env
    obs_color = 'tab:grey'
    obs = testing_memory_buffer.timestep_data['obs']
    obs = np.array(obs)

    env_key = testing_memory_buffer.timestep_data['context_names']
    means = []
    for env_key in testing_memory_buffer.timestep_data['context_names']:
        mean = testing_env.envs[env_key]['kwargs']['mean']
        means.append(mean)
    means = np.array(means)

    max_trials = config.training_phases[0]['config']['max_trials_per_block']
    context2 = get_context2_vector(testing_memory_buffer, config)

    fig, axes = plt.subplot_mosaic([['A'], ['B'], ['B']], sharex=True,
                                    constrained_layout=False, figsize = [16/2.53, 5/2.53], dpi=300)

    for label, ax in axes.items():
        # label physical distance to the left and up: (left, up) raise up to move label up
        trans = mtransforms.ScaledTranslation(-23/72, 2/72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize='large', va='bottom', fontfamily='arial',weight='bold')

    ax = axes['B'] 
    obs = np.stack(logger.timestep_data['obs']).squeeze()
    preds = np.stack(logger.timestep_data['predictions']).squeeze()
    ax.plot(obs, 'o', label='obs', markersize=0.5, color=obs_color)
    # ax.plot(preds, 'o', label='preds', markersize=0.5, color=preds_color)
    switches_ts_padded = env.env_logger['switches_ts'] +[logger.timestep_data['timestep_i'][-1]]
    # ax.legend(loc='upper right', fontsize=6, ncol=2)
    ax.legend(fontsize=6,)# loc='upper left')#loc=(.3,1.2))
    axes_labels(ax,'Time steps','Observations')
    # ax.set_ylim([-0.1, 1.1])
    # ax.set_xticklabels([])


    for i, switch in enumerate(switches_ts_padded[:-1]):
        if i%2 == 0 and  switches_ts_padded[i+1] < x2:
            ax.axvspan(switches_ts_padded[i], switches_ts_padded[i+1], alpha=0.1, color='grey')

    ax = axes['A']
    for c in range(20, len(obs)):
        if (c-20) % (3*max_trials) ==0: # if context changes
            color = 'b' if c//(3*max_trials) % 2 == 0 else 'g'
            ax.axvline(c, .6, .9 , color=color, linestyle='-', alpha=0.6, linewidth=3)

    # Define the colormap
    cmap = plt.get_cmap('magma')

    # Normalize the means vector to range between 0 and 1
    norm = plt.Normalize(vmin=0, vmax=1)

    # Convert the means vector to a color vector using the colormap
    color_vector = cmap((means))

    for b in range(20,len(means)):
        if (b-20) % (max_trials) ==0: # if context changes
            color = color_vector[b]
            ax.axvline(b, 0.0,0.4,  alpha=0.6, color=color, linewidth=3)
    # ax.axhline(1, color=color_vector, linestyle='-', linewidth=3, alpha=0.5)
    ax.set_axis_off()


def get_context2_vector(logger, config):
    obs = np.stack(logger.timestep_data['obs'])
    max_trials = config.training_phases[0]['config']['max_trials_per_block']

    context2 = np.zeros(len(obs))
    # context2 is 1 for indices from 0 to 3*max_trials trials and the -1 from 3*max_trials trials to 6*max_trials trials and so on
    for i in range(len(context2)):
        if i//(3*max_trials) % 2 == 0:
            context2[i] = 1
        else:
            context2[i] = -1
    # the very first block is a special case, choosen always to be 20 and does not belong to any of the two sequences
    # append 20 -1s to the beginning of the context2 and then remove the last 20 elements
    context2 = np.concatenate([np.ones(20)*-1, context2])
    context2 = context2[:-20]
    # print('context2' , context2)
    return context2

def get_level_2_values(env, testing_memory_buffer):
    switches_ts = env.env_logger['switches_ts']
    level_2_event_type = env.env_logger['level_2_values']
    # print(level_2_event_type)
    current_level_2_vector = []
    current_level_2 = 0

    # get the hiest timestep in switches ts
    max_ts = testing_memory_buffer.timestep_data['trial_i'][-1]

    for ts in range(max_ts+1):
        if ts in switches_ts:
                    current_level_2 = level_2_event_type[switches_ts.index(ts)]
        current_level_2_vector.append(current_level_2)

    current_level_2_vector = np.array(current_level_2_vector)
    return current_level_2_vector


def plot_modulations(testing_memory_buffer, testing_env, testing_losses, config, x1=50, x2=np.inf, latent_dim = 0, replace_context2_with_level2 = False):
    # now get the gradients from the memory buffer
    grads = np.stack(testing_memory_buffer.timestep_data['thalamus_grad'])
    grads = grads.squeeze()[:, latent_dim] # pick one unit grads
    grads[0] = 0 # first value is padded with a nan
    centered_grads = grads - np.mean(grads)

    # get the obs from the memory buffer
    obs = testing_memory_buffer.timestep_data['obs']
    obs = np.array(obs)

    env_key = testing_memory_buffer.timestep_data['context_names']
    means = []
    for env_key in testing_memory_buffer.timestep_data['context_names']:
        mean = testing_env.envs[env_key]['kwargs']['mean']
        means.append(mean)
    means = np.array(means)

    max_trials = config.training_phases[0]['config']['max_trials_per_block']

    if not replace_context2_with_level2:
        context2 = np.zeros(len(obs))
        # context2 is 1 for indices from 0 to 3*max_trials trials and the -1 from 3*max_trials trials to 6*max_trials trials and so on
        for i in range(len(context2)):
            if i//(3*max_trials) % 2 == 0:
                context2[i] = 1
            else:
                context2[i] = -1
        # the very first block is a special case, choosen always to be 20 and does not belong to any of the two sequences
        # append 20 -1s to the beginning of the context2 and then remove the last 20 elements
        context2 = np.concatenate([np.ones(20)*-1, context2])
        context2 = context2[:-20]
    else:
        context2 = get_level_2_values(testing_env, testing_memory_buffer)

    # Now that I have the grads in a vector, I want to check their modulation by context2 array vs means array, using something similar to the analysis below:

    centered_grads = grads - np.mean(grads)
    centered_means = means-np.mean(means)
    grads_modulation_by_context2 = np.dot(context2-np.mean(context2), centered_grads)/ len(context2) # 
    grads_modulation_by_context2 = np.abs(grads_modulation_by_context2)

    grads_modulation_by_means = np.dot(centered_means, centered_grads)/ len(means) #
    grads_modulation_by_means = np.abs(grads_modulation_by_means)

    grads_corr_by_context2 = np.abs(np.corrcoef(grads, context2)[0][1])
    grads_corr_by_means = np.abs(np.corrcoef(grads, means)[0][1])

    fig, axes = plt.subplot_mosaic([['A','B','B','B','B'], ['C', 'D', 'D', 'D', 'D']],
                            constrained_layout=False, figsize = [12/2.53, 7/2.53])
    import matplotlib.transforms as mtransforms
    for label, ax in axes.items():
        # label physical distance to the left and up: (left, up) raise up to move label up
        trans = mtransforms.ScaledTranslation(-23/72, 2/72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
                fontsize='large', va='bottom', fontfamily='arial',weight='bold')

    # share x-axis between axes A and B
    ax = axes['A']
    ax2 = axes['C']
    ax.get_shared_y_axes().join(ax, ax2)

    # bar plot the modulation
    # fig, axes = plt.subplots(3,1, figsize=(8,6))
    ax = axes['A']
    ax.bar([-1,0,1], [0,grads_corr_by_means,0], color='tab:blue', width=0.95)
    ax.set_xticks([0])
    ax.set_xticklabels([ 'Means'])
    ax.set_ylabel('Correlation')
    ax.set_xlabel('Variable')

    ax = axes['C']
    ax.bar([-1,0,1], [0,grads_corr_by_context2,0], color='tab:green', width=0.9)
    ax.set_xticks([0])
    ax.set_xticklabels(['Context2'])
    ax.set_ylabel('Correlation')
    ax.set_xlabel('Variable')

    # plot centered grads overlaid on centered means
    # ax.plot(centered_grads, label='grads')
    ax = axes['B']
    ax.plot(centered_means, label='means', color='tab:blue')
    ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
    # ax.set_ylabel('Context1')
    ax.set_xlabel('Trials')
    # centered_grads have a very different scale, plot them on a second axis
    ax2 = ax.twinx()
    ax2.plot(centered_grads, label='grads', color='tab:orange', linewidth=1, alpha=0.7)
    ax2.set_ylabel('Grads')
    ax2.spines['right'].set_color('orange')
    ax.legend()

    # plot centered grads overlaid on centered context2
    ax = axes['D']
    ax.plot(context2-np.mean(context2), label='context2', color='tab:green')
    ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
    # ax.set_ylabel('Context2')
    ax.set_xlabel('Trials')
    # centered_grads have a very different scale, plot them on a second axis
    ax2 = ax.twinx()
    ax2.plot(centered_grads, label='grads', color='tab:orange', linewidth=1, alpha=0.7)
    ax2.set_ylabel('Grads')
    ax2.spines['right'].set_color('orange')
    ax2.legend()
    ax.legend()
    fig.tight_layout()

def plot_dual_modulations(testing_memory_buffer, testing_env, testing_losses, config, x1=50, x2=np.inf):
    # now get the gradients from the memory buffer
    grads = np.stack(testing_memory_buffer.timestep_data['thalamus_grad'])
    grads = grads.squeeze()
    grads[0,:] = grads[1,:] # first value is padded with a nan

    # get the obs from the memory buffer
    obs = testing_memory_buffer.timestep_data['obs']
    obs = np.array(obs)

    env_key = testing_memory_buffer.timestep_data['context_names']
    means = []
    for env_key in testing_memory_buffer.timestep_data['context_names']:
        mean = testing_env.envs[env_key]['kwargs']['mean']
        means.append(mean)
    means = np.array(means)

    max_trials = config.training_phases[0]['config']['max_trials_per_block']

    context2 = np.zeros(len(obs))
    # context2 is 1 for indices from 0 to 3*max_trials trials and the -1 from 3*max_trials trials to 6*max_trials trials and so on
    for i in range(len(context2)):
        if i//(3*max_trials) % 2 == 0:
            context2[i] = 1
        else:
            context2[i] = -1
    # the very first block is a special case, choosen always to be 20 and does not belong to any of the two sequences
    # append 20 -1s to the beginning of the context2 and then remove the last 20 elements
    context2 = np.concatenate([np.ones(20)*-1, context2])
    context2 = context2[:-20]

    centered_grads = (grads - np.mean(grads, axis=0)) * 1/np.std(grads, axis=0)
    centered_means = means-np.mean(means, axis=0)

    grads_corr_by_context2 = np.abs(np.corrcoef(centered_grads[:, 0], context2)[0][1])
    grads_corr_by_means = np.abs(np.corrcoef(centered_grads[:, 0], centered_means)[0][1])
    grads_corr_by_context2_2 = np.abs(np.corrcoef(centered_grads[:, 2], context2)[0][1])
    grads_corr_by_means_2 = np.abs(np.corrcoef(centered_grads[:, 2], centered_means)[0][1])

    fig, axes = plt.subplot_mosaic([['A','B','B','B','B'], ['C', 'D', 'D', 'D', 'D']],
                            constrained_layout=False, figsize = [12/2.53, 7/2.53])
    import matplotlib.transforms as mtransforms
    for label, ax in axes.items():
        # label physical distance to the left and up: (left, up) raise up to move label up
        trans = mtransforms.ScaledTranslation(-23/72, 2/72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
                fontsize='large', va='bottom', fontfamily='arial',weight='bold')

    # share x-axis between axes A and B
    ax = axes['A']
    ax2 = axes['C']
    ax.get_shared_y_axes().join(ax, ax2)

    # bar plot the modulation
    # fig, axes = plt.subplots(3,1, figsize=(8,6))
    ax = axes['A']
    ax.bar([-1,0,1], [grads_corr_by_means,0,grads_corr_by_means_2], color='tab:blue', width=0.95)
    ax.set_xticks([-1,1])
    ax.set_xticklabels([ 'z1', 'z2'])
    ax.set_ylabel('Correlation')
    ax.set_xlabel('Means')

    ax = axes['C']
    ax.bar([-1,0,1], [grads_corr_by_context2,0,grads_corr_by_context2_2], color='tab:green', width=0.9)
    ax.set_xticks([-1,1])
    ax.set_xticklabels(['Z1', 'Z2'])
    ax.set_ylabel('Correlation')
    ax.set_xlabel('Context2')

    # plot centered grads overlaid on centered means
    # ax.plot(centered_grads, label='grads')
    ax = axes['B']
    ax.plot(centered_means, label='means', linewidth=1, color='tab:blue')
    # ax.plot(centered_means, label='means', color='tab:orange', alpha=0.5)
    ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
    # ax.set_ylabel('Context1')
    ax.set_xlabel('Trials')
    # centered_grads have a very different scale, plot them on a second axis
    ax2 = ax.twinx()
    ax2.plot(centered_grads[:, 0], '-', label='Z1 grads', color='tab:orange', alpha=0.8, linewidth=1)
    ax2.plot(centered_grads[:, 2], label='Z2 grads', color='tab:red', linewidth=1)
    ax2.set_ylabel('Grads')
    ax2.spines['right'].set_color('tab:blue')
    ax.legend()

    # plot centered grads overlaid on centered context2
    ax = axes['D']
    ax.plot(context2-np.mean(context2), label='context2', color='tab:green')
    ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
    # ax.set_ylabel('Context2')
    ax.set_xlabel('Trials')
    # centered_grads have a very different scale, plot them on a second axis
    ax2 = ax.twinx()
    # ax2.plot(centered_grads[:, 0], label='grads', color='tab:blue')
    # ax2.plot(centered_grads[:, 1], label='grads', color='tab:orange', alpha=0.5)
    ax2.plot(centered_grads[:, 0], '-', label='Z1 grads', color='tab:orange', alpha=0.8, linewidth=1)
    ax2.plot(centered_grads[:, 2], label='Z2 grads', color='tab:red', linewidth=1)
    ax2.set_ylabel('Grads')
    ax2.spines['right'].set_color('tab:blue')
    ax2.legend()
    ax.legend()
    fig.tight_layout()

# plot_modulations(testing_memory_buffer, testing_env, testing_losses, config, x1=50, x2=np.inf)
def get_correlations(testing_memory_buffer, testing_env, config, use_grads=True, latent_dim=0):
    # get the thalamus activity from the memory buffer
    thalamus = np.stack(testing_memory_buffer.timestep_data['thalamus'])
    thalamus = thalamus.squeeze()[:, latent_dim] # take only one unit 

    grads = np.stack(testing_memory_buffer.timestep_data['thalamus_grad'])
    grads = grads.squeeze()[:, latent_dim] # take only one unit grads
    grads[0] = 0

    # get the obs from the memory buffer
    obs = testing_memory_buffer.timestep_data['obs']
    obs = np.array(obs)

    env_key = testing_memory_buffer.timestep_data['context_names']
    means = []
    for env_key in testing_memory_buffer.timestep_data['context_names']:
        mean = testing_env.envs[env_key]['kwargs']['mean']
        means.append(mean)
    means = np.array(means)

    max_trials = config.training_phases[0]['config']['max_trials_per_block']

    context2 = np.zeros(len(obs))
    # context2 is 1 for indices from 0 to 3*max_trials trials and the -1 from 3*max_trials trials to 6*max_trials trials and so on
    for i in range(len(context2)):
        if i//(3*max_trials) % 2 == 0:
            context2[i] = 1
        else:
            context2[i] = -1
    # the very first block is a special case, choosen always to be 20 and does not belong to any of the two sequences
    # append 20 -1s to the beginning of the context2 and then remove the last 20 elements
    context2 = np.concatenate([np.ones(20)*-1, context2])
    context2 = context2[:-20]

    if not use_grads:
        corr_context2 = np.abs(np.corrcoef(thalamus, context2)[0][1])
        corr_means = np.abs(np.corrcoef(thalamus, means)[0][1])
    else: # use grads
        corr_context2 = np.abs(np.corrcoef(grads, context2)[0][1])
        corr_means = np.abs(np.corrcoef(grads, means)[0][1])
        
    return corr_context2, corr_means

def extract_gen_performance(logger, env, ts_before=20, ts_after=0):
    obs = np.stack(logger.timestep_data['obs']).squeeze()
    preds = np.stack(logger.timestep_data['predictions']).squeeze()
    if preds.shape[-1] == 2:
        preds = preds[:,:1] # take only the mean pred, not variance
# if you get logger.timestep_data['context_name'] at [switch] you get the name of the upcoming context after the switch 
    switches_ts = env.env_logger['switches_ts']
    switches_ts = np.array(switches_ts)
    mean_diff = []
    distances_from_mean = defaultdict(list)
    inferred_means = defaultdict(list)
    mean_abs_errors = defaultdict(list)
    for i, switch in enumerate(switches_ts[1:]): # ignore the first and last switch
        if len (obs[switch-ts_before:switch+ts_after]) == ts_before+ts_after:
            mean_diff.append(np.abs((obs[switch-ts_before:switch+ts_after] - preds[switch-ts_before:switch+ts_after])))
        # so to get the id of the context before the switch you need to get the context name at the switch -1
            env_key = logger.timestep_data['context_names'][switch-1]
            mean = env.envs[env_key]['kwargs']['mean']
            distances_from_mean[mean].append(np.mean(np.abs((preds[switch-ts_before:switch+ts_after] - mean))))
            inferred_means[mean].append(np.mean(preds[switch-ts_before:switch+ts_after]))
            mean_abs_errors[mean].append(np.mean(np.abs((obs[switch-ts_before:switch+ts_after] - preds[switch-ts_before:switch+ts_after]))))
        # calc MSE instead
        # mean_abs_errors[mean].append(np.square((obs[switch-ts_before:switch+ts_after] - preds[switch-ts_before:switch+ts_after])))
        
    return distances_from_mean,inferred_means,mean_abs_errors

def extract_per_std_distance_from_mean(logger, env, ts_before=20, ts_after=0, abs_dist=False):
    obs = np.stack(logger.timestep_data['obs']).squeeze()
    preds = np.stack(logger.timestep_data['predictions']).squeeze()
    if preds.shape[-1] == 2:
        preds = preds[:,:1]
    # if you get logger.timestep_data['context_name'] at [switch] you get the name of the upcoming context after the switch 
    switches_ts = env.env_logger['switches_ts']
    switches_ts = np.array(switches_ts)
    mean_diff = []
    distances_from_mean = defaultdict(list)
    inferred_means = defaultdict(list)
    mean_abs_errors = defaultdict(list)
    switch_triggered_distance_from_mean = defaultdict(list)
    for i, switch in enumerate(switches_ts[1:]): # ignore the first and last switch
        if len (obs[switch-ts_before:switch+ts_after]) == ts_before+ts_after:
            # mean_diff.append(np.abs((obs[switch-ts_before:switch+ts_after] - preds[switch-ts_before:switch+ts_after])))
        # so to get the id of the context before the switch you need to get the context name at the switch -1
            switch_to_context_name = logger.timestep_data['context_names'][switch]
            switch_from_context_name = logger.timestep_data['context_names'][switch-1]
            if switch_to_context_name == 'base_gauss' and switch_from_context_name != 'base_gauss':
                next_mean = env.envs[switch_to_context_name]['kwargs']['mean']
                previous_mean = env.envs[switch_from_context_name]['kwargs']['mean']
                previous_std = env.envs[switch_from_context_name]['kwargs']['std']

                if abs_dist:
                    measure_previous = np.abs((preds[switch-ts_before:switch] - previous_mean))
                    measure_next = np.abs((preds[switch:switch+ts_after] - next_mean))
                else:
                    measure_previous = -((preds[switch-ts_before:switch] - previous_mean))
                    measure_next = ((preds[switch:switch+ts_after] - next_mean))
                measure = np.concatenate((measure_previous, measure_next))
                switch_triggered_distance_from_mean[previous_std].append(measure)
    
            # distances_from_mean[mean].append(np.mean(np.abs((preds[switch-ts_before:switch+ts_after] - mean))))
            # inferred_means[mean].append(np.mean(preds[switch-ts_before:switch+ts_after]))
            # mean_abs_errors[mean].append(np.mean(np.abs((obs[switch-ts_before:switch+ts_after] - preds[switch-ts_before:switch+ts_after]))))
        # calc MSE instead
        # mean_abs_errors[mean].append(np.square((obs[switch-ts_before:switch+ts_after] - preds[switch-ts_before:switch+ts_after])))
        
    # return distances_from_mean,inferred_means,mean_abs_errors
    return switch_triggered_distance_from_mean

def extract_per_std_distance_from_mean_v2(logger, env, ts_before=20, ts_after=0, abs_dist=True):
    ''' flipping the experiment to be from base Gauss to variable STD block '''
    obs = np.stack(logger.timestep_data['obs']).squeeze()
    preds = np.stack(logger.timestep_data['predictions']).squeeze()
    if preds.shape[-1] == 2:
        preds = preds[:,:1]
    # if you get logger.timestep_data['context_name'] at [switch] you get the name of the upcoming context after the switch 
    switches_ts = env.env_logger['switches_ts']
    switches_ts = np.array(switches_ts)
    mean_diff = []
    switch_triggered_distance_from_mean = defaultdict(list)
    for i, switch in enumerate(switches_ts): 
        if len (obs[switch-ts_before:switch+ts_after]) == ts_before+ts_after:
            # mean_diff.append(np.abs((obs[switch-ts_before:switch+ts_after] - preds[switch-ts_before:switch+ts_after])))
            # But using error is not helpful becuase it is not normalized by the std of the data. 
            # So instead get the true mean before and after the switch and measure the distance from that mean
            switch_to_context_name = logger.timestep_data['context_names'][switch]
            switch_from_context_name = logger.timestep_data['context_names'][switch-1]

            if switch_from_context_name == 'base_gauss' and switch_to_context_name != 'base_gauss':
                from_mean = env.envs[switch_from_context_name]['kwargs']['mean']
                to_mean = env.envs[switch_to_context_name]['kwargs']['mean']
                # from_std = env.envs[switch_from_context_name]['kwargs']['std']
                to_std = env.envs[switch_to_context_name]['kwargs']['std']
                if abs_dist:
                    distance_from_mean_before_switch = np.abs((preds[switch-ts_before:switch] - from_mean))
                    distance_from_mean_after_switch = np.abs((preds[switch:switch+ts_after] - to_mean))
                else:
                    distance_from_mean_before_switch = ((preds[switch-ts_before:switch] - from_mean))
                    distance_from_mean_after_switch = -((preds[switch:switch+ts_after] - to_mean))

                distance_from_mean = np.concatenate((distance_from_mean_before_switch, distance_from_mean_after_switch))
                switch_triggered_distance_from_mean[to_std].append(distance_from_mean)

                
    return switch_triggered_distance_from_mean



def plot_switch_triggered_per_std_distance_from_mean(switch_triggered_distance_from_mean, ax, params= None, cmap=plt.cm.viridis, ts_before=20, ts_after=40):
    colors = cmap(np.linspace(0, 1, 5+2))#+3)) 
    for i, (std, distances) in enumerate(switch_triggered_distance_from_mean.items()):
        if std != 0.0:
            color = colors[i]
            ax.plot(range(-ts_before, np.stack(distances).shape[1]-ts_before) , np.mean(np.stack(distances), axis=0), label=f'STD {std}', linewidth=0.5, color=color)
    # ax.legend(loc='lower right', fontsize=6, ncol=1)
    # ax.set_xlim([-5,20])
    # ax.set_xticklabels([-5,0,5,10,15,20])
    # ax.set_yticklabels([0.0,0.2,0.4,0.6, 0.8][::-1])
    ax.axvline(0, linewidth=0.5, color='k', linestyle='--')
    if params is not None:
        ax.set_title(params, fontsize=6)
    axes_labels(ax,'Time steps from switch','Preds distance from mean')
    # ax.grid(color='k', linestyle='--', linewidth=0.1, alpha=0.5)
    ax.set_ylim([.5,0])


    # plot inferred means
def plot_mean_std_lines(title, axes, means, all_distances_from_mean_RNN, all_inferred_means_RNN, all_mean_abs_errors_RNN, stds=[0.1, 0.2, 0.3, 0.4, 0.5]):
    # change color scheme to viridis
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, 3+len(stds)))
    for si,std_ in enumerate(stds):
        distances_from_mean_RNN = all_distances_from_mean_RNN[si]
        inferred_means_RNN = all_inferred_means_RNN[si]
        mean_abs_errors_RNN = all_mean_abs_errors_RNN[si]

        ax = axes['A']
        inferred_RNN = [np.array(inferred_means_RNN[mean]).mean() for mean in means]
        stds_RNN = [np.array(inferred_means_RNN[mean]).std() for mean in means]
        ax.plot( means, inferred_RNN, label=f'std_' + str(std_), linewidth=1, color=colors[si])# color= no_context_color)
    # ax.fill_between(means, np.array(inferred_RNN) - np.array(stds_RNN), np.array(inferred_RNN) + np.array(stds_RNN), alpha=0.2, color= no_context_color)
        ax.plot([-.2, 1.2], [-0.2, 1.2], color='k', linestyle='--', linewidth=1, alpha=0.5)
        axes_labels(ax, 'Gaussian mean', 'Preds mean')
    # ax.legend(fontsize=6)
        ax.set_title(title)

    # ax.set_title('inferred means')
    axes_labels(ax, 'Obs mean', 'Preds mean')
    # ax.legend(fontsize=6)
    # draw a diagonal line
    ax.plot([0, 1], [0, 1], color='k', linestyle='--', linewidth=1, alpha=0.5)
    ax.plot([-.2, 1.2], [-0.2, 1.2], color='k', linestyle='--', linewidth=1, alpha=0.5)


def plot_all_generalization_lines(memory_buffers, envs, ax, title='', stds=[0.1, 0.2, 0.3, 0.4, 0.5], means=np.array(list(range(-2, 13)))/10):
    with_context_color = 'tab:blue'
    no_context_color = 'tab:orange'

    # change color scheme to viridis
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, 3+len(stds)))

    # plot inferred means
    for si,std_ in enumerate(stds):
        distances_from_mean, inferred_means, mean_abs_errors = extract_gen_performance(memory_buffers[si],envs[si])

        inferred_RNN = [np.array(inferred_means[mean]).mean() for mean in means]
        stds_RNN = [np.array(inferred_means[mean]).std() for mean in means]
        ax.plot( means, inferred_RNN, label=f'STD ' + str(std_), linewidth=1, color=colors[si])# color= no_context_color)
        # ax.fill_between(means, np.array(inferred_RNN) - np.array(stds_RNN), np.array(inferred_RNN) + np.array(stds_RNN), alpha=0.2, color= no_context_color)
        ax.plot([-.2, 1.2], [-0.2, 1.2], color='k', linestyle='--', linewidth=0.5, alpha=0.5)
        axes_labels(ax, 'Gaussian mean', 'Preds mean')
        # ax.legend(fontsize=6)
        # set xticks 0, to 1 with 0.2 steps
        # ax.set_yticks(np.arange(0, 1.2, 0.2))
        ax.grid(color='k', linestyle='--', linewidth=0.1, alpha=0.5)
        ax.set_title(title)


        # ax.set_title('inferred means')
        axes_labels(ax, 'Gaussian mean', 'Preds mean')
        # ax.legend(fontsize=6)
        # draw a diagonal line
        ax.plot([0, 1], [0, 1], color='k', linestyle='--', linewidth=1, alpha=0.5)
        ax.plot([-.2, 1.2], [-0.2, 1.2], color='k', linestyle='--', linewidth=1, alpha=0.5)
        # zoom in to 0,1
        # ax.set_xlim([0, 1])
        # ax.set_ylim([0, 1])
    # fig.savefig('./exports/RNN_only_generalization.PDF', dpi=300)



# BEGIN: 8f7d6a9zj3k1
def plot_combined_behavior_and_histograms(logger, env, fig_height=4, ylim=1000):
    fig, axes = plt.subplots(2, 2, figsize=(2.2, fig_height), dpi=100, sharey='col')

    # Plot the 4th subplot from plot_behavior_novel_contexts
    sid1 = 4
    sid2 = 5
    ax = axes[0, 0]
    obs = np.stack(logger.timestep_data['obs']).squeeze()
    preds = np.stack(logger.timestep_data['predictions']).squeeze()
    if preds.shape[-1] == 2:
        preds = preds[..., :1]
    ax.plot(obs, 'o', label='obs', markersize=0.5, color=obs_color)
    ax.plot(preds, 'o', label='preds', markersize=0.5, color=preds_color)
    switches_ts_padded = env.env_logger['switches_ts'] + [logger.timestep_data['timestep_i'][-1]]
    axes_labels(ax, '', 'Observations')
    ax.set_xticklabels([])
    # ax.set_xlim(switches_ts_padded[sid1] - 100, switches_ts_padded[sid1])
    ax.set_xlim(switches_ts_padded[sid1] - 100, switches_ts_padded[sid1] + 20)
    ax.axvspan(switches_ts_padded[sid1] -100, switches_ts_padded[sid1], alpha=0.1, color='grey')

    # Plot the 5th subplot from plot_behavior_novel_contexts
    ax = axes[1, 0]
    ax.plot(obs, 'o', label='obs', markersize=0.5, color=obs_color)
    ax.plot(preds, 'o', label='preds', markersize=0.5, color=preds_color)
    axes_labels(ax, 'Time steps', 'Observations')
    ax.set_xticklabels([])
    # ax.set_xlim(switches_ts_padded[sid2]-100, switches_ts_padded[sid2] )
    ax.set_xlim(switches_ts_padded[sid2]-100, switches_ts_padded[sid2]+20 )
    ax.axvspan(switches_ts_padded[sid2] , switches_ts_padded[sid2]+20, alpha=0.1, color='grey') 

    ax.set_ylim(-.2, 2.)

    # Plot the 4th subplot from plot_histograms_novel_contexts
    switches_ts = env.env_logger['switches_ts']
    ax = axes[0, 1]
    preds = np.stack(logger.timestep_data['predictions']).squeeze()
    obs = np.stack(logger.timestep_data['obs']).squeeze()
    if preds.shape[-1] == 2:
        preds = preds[..., :1]
    bins = np.linspace(-.6, 1.6, 80)
    ax.hist(preds[switches_ts[3] + 100:switches_ts[4]], bins=bins, alpha=0.75, label='preds', color=preds_color)
    ax.hist(obs[switches_ts[3]:switches_ts[4]], bins=bins, alpha=0.65, label='obs', color=obs_color)
    axes_labels(ax, '', 'Count')
    switch_to_context_name = logger.timestep_data['context_names'][switches_ts[3]]
    context_mean = env.envs[switch_to_context_name]['kwargs']['mean']
    context_std = env.envs[switch_to_context_name]['kwargs']['std']
    # ax.set_xlabel(f'{context_mean:.2f}±{context_std:.2f}')
    ax.axvspan(ax.get_xlim()[0], ax.get_xlim()[1],alpha=0.1, color='grey')
    # Plot the 5th subplot from plot_histograms_novel_contexts
    ax = axes[1, 1]
    preds = np.stack(logger.timestep_data['predictions']).squeeze()
    obs = np.stack(logger.timestep_data['obs']).squeeze()
    if preds.shape[-1] == 2:
        preds = preds[..., :1]
    bins = np.linspace(-.6, 1.6, 80)
    ax.hist(preds[switches_ts[4] + 100:switches_ts[5]], bins=bins, alpha=0.75, label='preds', color=preds_color)
    ax.hist(obs[switches_ts[4]:switches_ts[5]], bins=bins, alpha=0.65, label='obs', color=obs_color)
    axes_labels(ax, 'Predictions', 'Count')
    switch_to_context_name = logger.timestep_data['context_names'][switches_ts[4]]
    context_mean = env.envs[switch_to_context_name]['kwargs']['mean']
    context_std = env.envs[switch_to_context_name]['kwargs']['std']
    # ax.set_xlabel(f'{context_mean:.2f}±{context_std:.2f}')
    ax.set_ylim([0, ylim])
    ax.autoscale() 

    axes[0,0].set_yticks([0,1])
    axes[1,0].set_yticks([0,1])
    axes[0,0].set_xticklabels(['', '25', '75'])
    axes[1,0].set_xticklabels(['', '25', '75'])
    fig.tight_layout()
# END: 8f7d6a9zj3k1
        

def plot_behavior_novel_contexts(logger, env,fig_height = 1.0):
    fig, axes = plt.subplots(1,2, figsize=(2.,1), dpi=100, sharey=True)
    ax = axes[0]
    obs = np.stack(logger.timestep_data['obs']).squeeze()
    preds = np.stack(logger.timestep_data['predictions']).squeeze()
    if preds.shape[-1] == 2:
        preds = preds [...,:1]
    ax.plot(obs, 'o', label='obs', markersize=0.5, color=obs_color)
    ax.plot(preds, 'o', label='preds', markersize=0.5, color=preds_color)
    switches_ts_padded = env.env_logger['switches_ts'] +[logger.timestep_data['timestep_i'][-1]]
    # ax.legend(loc='upper right', fontsize=6, ncol=2)
    # ax.legend(fontsize=6,)# loc='upper left')#loc=(.3,1.2))
    
    axes_labels(ax,'Time steps','Observations')
    ax.set_xticklabels([])
    # ax.set_xlim(switches_ts_padded[7]-100, switches_ts_padded[7] )

    ax.set_xlim(switches_ts_padded[6]-10, switches_ts_padded[6]+100 )
    ax.axvspan(switches_ts_padded[6]-10, switches_ts_padded[6], alpha=0.1, color='grey')

    ax = axes[1]
    ax.plot(obs, 'o', label='obs', markersize=0.5, color=obs_color)
    ax.plot(preds, 'o', label='preds', markersize=0.5, color=preds_color)
    # ax.legend(loc='upper right', fontsize=6, ncol=2)
    # ax.legend(fontsize=6,)# loc='upper left')#loc=(.3,1.2))
    axes_labels(ax,'Time steps','Observations')
    ax.set_xticklabels([])
    # ax.set_xlim(switches_ts_padded[7], switches_ts_padded[7] + 150)
    ax.set_xlim(switches_ts_padded[7]-10, switches_ts_padded[7] + 140)
    ax.axvspan(switches_ts_padded[7]-10, switches_ts_padded[7], alpha=0.1, color='grey')

    ax.set_ylim(-.5,2.5)
    plt.tight_layout()

    fig, axes = plt.subplots(1,8 , figsize=(7, fig_height), dpi=100, sharey=True)
    switches_ts = env.env_logger['switches_ts']
    for i in range(1, min(len(switches_ts)-1, 8), 1):
        ax = axes[i-1]
        ax.plot(obs, 'o', label='obs', markersize=0.4, color=obs_color, alpha=0.7)
        ax.plot(preds, 'o', label='preds', markersize=0.4, color=preds_color, alpha=0.7)
        # ax.legend(loc='upper right', fontsize=6, ncol=2)
        # ax.legend(fontsize=6,)# loc='upper left')#loc=(.3,1.2))
        axes_labels(ax,'Time steps','Observations')
        ax.set_yticks([0,1])
        ax.set_xticklabels(['', '25', '75'])
        ax.set_xlim(switches_ts[i]-100, switches_ts[i] )

        switch_to_context_name = logger.timestep_data['context_names'][switches_ts[i-1]]
        context_mean = env.envs[switch_to_context_name]['kwargs']['mean']
        context_std = env.envs[switch_to_context_name]['kwargs']['std'] 
        ax.set_xlabel(f'{context_mean:.2f}±{context_std:.2f}')
    fig.tight_layout()

def plot_histograms_novel_contexts(logger, env, fig_height = 0.9):
    preds = np.stack(logger.timestep_data['predictions']).squeeze()
    obs = np.stack(logger.timestep_data['obs']).squeeze()
    # remove var output if exists
    if preds.shape[-1] == 2:
        preds = preds [...,:1]

    obs_color = 'tab:grey'
    preds_color = 'tab:red'

    fig, axes = plt.subplots(1, 8, figsize=(9, fig_height), dpi=100)
    mixing_trials = 100
    # now plot the histogram for predictions from every other block according to the env.env_logger['switches_ts'] switch times
    switches_ts = env.env_logger['switches_ts']
    for i in range(0, min(len(switches_ts)-1, 8), 1):
        ax = axes[i]
        bins = np.linspace(-.6, 1.6, 80)
        ax.hist(preds[switches_ts[i]+mixing_trials:switches_ts[i+1]], bins= bins, alpha=0.75, label='preds', color=preds_color)
        ax.set_title(f'mean: {np.mean(preds[switches_ts[i]:switches_ts[i+1]]):.2f}', fontsize=6)
        ax.hist(obs[switches_ts[i]:switches_ts[i+1]], bins= bins, alpha=0.65, label='obs', color=obs_color)
        axes_labels(ax, 'Predictions', 'Count')
        ax.set_ylim(0, 1000)
        # for the first 2 axes resize the plot to 2/3 its height.
        switch_to_context_name = logger.timestep_data['context_names'][switches_ts[i]]
        context_mean = env.envs[switch_to_context_name]['kwargs']['mean']
        context_std = env.envs[switch_to_context_name]['kwargs']['std']
        ax.set_xlabel(f'{context_mean:.2f}±{context_std:.2f}')

        

def plot_only_behavior(logger, env, config, x1=50, x2=150):

    fig, axes = plt.subplot_mosaic([['A']], sharex=True,
                                    constrained_layout=False, figsize = [6/2.53, 2/2.53])

    for label, ax in axes.items():
        # label physical distance to the left and up: (left, up) raise up to move label up
        trans = mtransforms.ScaledTranslation(-23/72, 2/72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize='large', va='bottom', fontfamily='arial',weight='bold')

    ax = axes['A'] 
    obs = np.stack(logger.timestep_data['obs']).squeeze()
    preds = np.stack(logger.timestep_data['predictions']).squeeze()
    ax.plot(obs, 'o', label='obs', markersize=0.5, color=obs_color)
    ax.plot(preds, 'o', label='preds', markersize=0.5, color=preds_color)
    switches_ts_padded = env.env_logger['switches_ts'] +[logger.timestep_data['timestep_i'][-1]]
    # ax.legend(loc='upper right', fontsize=6, ncol=2)
    ax.legend(fontsize=6,)# loc='upper left')#loc=(.3,1.2))
    axes_labels(ax,'Time steps','Observations')
    # ax.set_xticklabels([])

    for ax in axes:
        for i, switch in enumerate(switches_ts_padded[:-1]):
            if i%2 == 0 and  switches_ts_padded[i+1] < x2:
                axes[ax].axvspan(switches_ts_padded[i], switches_ts_padded[i+1], alpha=0.1, color='grey')
        axes[ax].set_xlim([x1, x2]) # zoomed in for clarity in k99 plots

def plot_behavior(logger, env, losses, config, _use_oracle = False):
    fig, axes = plt.subplot_mosaic([['A','A','A',], ['B', 'C', 'D']],
                                constrained_layout=False, figsize = [12/2.53, 7/2.53])
    import matplotlib.transforms as mtransforms
    for label, ax in axes.items():
        # label physical distance to the left and up: (left, up) raise up to move label up
        trans = mtransforms.ScaledTranslation(-23/72, 2/72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize='large', va='bottom', fontfamily='arial',weight='bold')

    # pltu.axes_labels(ax,'','Mean FR')
    # pltu.beautify_plot(ax,x0min=False,y0min=False)
            
    ax = axes['A']
    obs = np.stack(logger.timestep_data['obs']).squeeze()
    preds = np.stack(logger.timestep_data['predictions']).squeeze()
    ax.plot(obs, 'o', label='obs', markersize=0.5, color=obs_color)
    if preds.shape[-1] == 2: # if there are two preds, plot both in different colors
        ax.plot(preds, 'o', label='preds', markersize=0.5)
    else:
        ax.plot(preds, 'o', label='preds', markersize=0.5, color=preds_color)

    switches_ts_padded = env.env_logger['switches_ts'] +[logger.timestep_data['timestep_i'][-1]]
    for i, switch in enumerate(switches_ts_padded[:-1]):
        if i%2 == 0:
            ax.axvspan(switches_ts_padded[i], switches_ts_padded[i+1], alpha=0.1, color='grey')
        # env_key = list(env.envs.keys())[i%2]
        env_key = logger.timestep_data['context_names'][switches_ts_padded[i]]
        mean = env.envs[env_key]['kwargs']['mean']
        # // draw a line at the mean of the current context between the context switches
        # ax.axhline(y=mean, xmin=switches_ts_padded[i]/(logger.timestep_data['timestep_i'][-1]), xmax=switches_ts_padded[i+1]/logger.timestep_data['timestep_i'][-1], color='black', linewidth=0.5, alpha=0.5)
        ax.plot([switches_ts_padded[i], switches_ts_padded[i+1]] , [mean,mean],
         color='black',linestyle='-', linewidth=1, alpha=0.4)

        # check if switch is less that the len of the array context_names
        # if switch < len(logger.timestep_data['context_names']):
        #     ax.text(switch, 0.95, logger.timestep_data['context_names'][switch], rotation=30, fontsize=5, alpha=0.5)
    ax.legend(loc='upper right', fontsize=6, ncol=2)
    ts_before, ts_after = 20, 50
    axes_labels(ax,'timestep','obs and preds')
    # ax.set_title(f'{"no oracle" if _use_oracle == 0 else "oracle"}')

    ax = axes['B']
    switches_ts = env.env_logger['switches_ts']
    switches_ts = np.array(switches_ts)
    for i, switch in enumerate(switches_ts[-6:-5]): # 
        if len (obs[switch-ts_before:switch+ts_after]) == ts_before+ts_after:
            ax.plot(range(-ts_before, ts_after), obs[switch-ts_before:switch+ts_after], '.', markersize=1)
            ax.plot(range(-ts_before, ts_after), preds[switch-ts_before:switch+ts_after], '.', markersize=1)
        else:
            ts_before_, ts_after_ = 2, 5
            print('not enough data')
            if len (obs[switch-ts_before_:switch+ts_after_]) == ts_before_+ts_after_:
                ax.plot(range(-ts_before_, ts_after_), obs[switch-ts_before_:switch+ts_after_], '.', markersize=1)
                ax.plot(range(-ts_before_, ts_after_), preds[switch-ts_before_:switch+ts_after_], '.', markersize=1)
    ax.set_xlabel('ts from ontext Switch')
    ax.axvline(x=0, color='grey', linestyle='--', linewidth=0.5, alpha=0.8)
    ax.set_ylabel('obs and preds')
        
    ax = axes['C']
    # calculate
    #  mean difference between obs and preds and plot averaged over ts_before timesteps before and ts_after timesteps after context switch context switches
    mean_diff = []
    for i, switch in enumerate(switches_ts[1:-1]): # ignore the first and last switch
        if len (obs[switch-ts_before:switch+ts_after]) == ts_before+ts_after:
            if config.capture_variance_experiment:
                mean_diff.append(np.abs((obs[switch-ts_before:switch+ts_after] - preds[switch-ts_before:switch+ts_after, 0])))
            else:
                mean_diff.append(np.abs((obs[switch-ts_before:switch+ts_after] - preds[switch-ts_before:switch+ts_after, ])))
    ax.plot(range(-ts_before, ts_after), np.stack(mean_diff).T, linewidth= 0.5, alpha=0.5)
    ax.plot(range(-ts_before, ts_after), np.mean(np.stack(mean_diff), axis=0), linewidth= 0.8, color='tab:red')
    ax.axvline(x=0, color='grey', linestyle='--', linewidth=0.5, alpha=0.8)
    ax.set_ylim([0,.7])
    ax.set_xlabel('ts from ontext Switch')
    ax.set_ylabel('Mean Absolute Difference')

    ax = axes['D']
    ax.plot(np.convolve(losses, np.ones(10)/10), linewidth=0.5)
    ax.set_xlabel('ts')
    ax.set_ylabel('loss')
    ax.set_ylim([0, 0.4])
    axes_labels(ax, 'ts', 'loss', )

    fig.tight_layout()
    # fig_clip_off(fig)
    # plt.savefig(logger.folder_name + f'{logger.experiment_name}.jpg', dpi=figure_dpi)

def plot_behavior_simple(logger, env, losses, config, _use_oracle = False, testing_memory_buffer=None, testing_env=None):
    fig, axes = plt.subplot_mosaic([['A','A'],['B', 'C']],
                                constrained_layout=False, figsize = [14/2.53, 6/2.53])
    import matplotlib.transforms as mtransforms
    for label, ax in axes.items():
        # label physical distance to the left and up: (left, up) raise up to move label up
        trans = mtransforms.ScaledTranslation(-23/72, 2/72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize='large', va='bottom', fontfamily='arial',weight='bold')

    # pltu.axes_labels(ax,'','Mean FR')
    # pltu.beautify_plot(ax,x0min=False,y0min=False)
            
    ax = axes['A']
    obs = np.stack(logger.timestep_data['obs']).squeeze()
    preds = np.stack(logger.timestep_data['predictions']).squeeze()
    if preds.shape[-1] == 2: # plot only mean preds
        preds = preds[..., :1]
    ax.plot(obs, 'o', label='obs', markersize=0.5, color=obs_color)
    ax.plot(preds, 'o', label='preds', markersize=0.5, color=preds_color)
    switches_ts_padded = env.env_logger['switches_ts'] +[logger.timestep_data['timestep_i'][-1]]
    for i, switch in enumerate(switches_ts_padded[:-1]):
        if i%2 == 0:
            ax.axvspan(switches_ts_padded[i], switches_ts_padded[i+1], alpha=0.1, color='grey')
        # env_key = list(env.envs.keys())[i%2]
        env_key = logger.timestep_data['context_names'][switches_ts_padded[i]]
        mean = env.envs[env_key]['kwargs']['mean']
        # // draw a line at the mean of the current context between the context switches
        # ax.axhline(y=mean, xmin=switches_ts_padded[i]/(logger.timestep_data['timestep_i'][-1]), xmax=switches_ts_padded[i+1]/logger.timestep_data['timestep_i'][-1], color='black', linewidth=0.5, alpha=0.5)
        ax.plot([switches_ts_padded[i], switches_ts_padded[i+1]] , [mean,mean],
         color='black',linestyle='-', linewidth=1, alpha=0.4)

        # check if switch is less that the len of the array context_names
        # if switch < len(logger.timestep_data['context_names']):
            # ax.text(switch, 0.95, logger.timestep_data['context_names'][switch], rotation=30, fontsize=8, alpha=0.5)
    ax.legend(loc='upper center', fontsize=6, ncol=2)
    ts_before, ts_after = 20, 50
    axes_labels(ax,'Training Time steps','obs and preds')
    # ax.set_title(f'{"no oracle" if _use_oracle == 0 else "oracle"}')
    ax.set_xlim([0, len(obs)])

    ax = axes['B']
    ax.plot(np.convolve(losses, np.ones(10)/10), linewidth=0.5, color='black')
    ax.set_xlabel('ts')
    ax.set_ylabel('loss')
    ax.set_ylim([0, 0.4])
    axes_labels(ax, 'Time steps', 'Training MSE loss', )
    ax.set_xlim([0, len(losses)])
    fig.tight_layout()
    fig_clip_off(fig)
    # plt.savefig(logger.folder_name + f'{logger.experiment_name}_simple.jpg', dpi=figure_dpi)

    # plot testing
    if testing_memory_buffer is not None:
        ax = axes['C']  
        logger = testing_memory_buffer
        env = testing_env
        obs = np.stack(logger.timestep_data['obs']).squeeze()
        preds = np.stack(logger.timestep_data['predictions']).squeeze()
        if preds.shape[-1] == 2: # plot only mean preds
            preds = preds[..., :1]

        ax.plot(obs, 'o', label='obs', markersize=0.5, color=obs_color)
        ax.plot(preds, 'o', label='preds', markersize=0.5, color=preds_color)
        switches_ts_padded = env.env_logger['switches_ts'] +[logger.timestep_data['timestep_i'][-1]]
        # ax.legend(loc='upper right', fontsize=6, ncol=2)
        ts_before, ts_after = 20, 50
        axes_labels(ax,'Testing time steps','')
        # ax.set_xticklabels([])

        switches_ts_padded = env.env_logger['switches_ts'] +[logger.timestep_data['timestep_i'][-1]]
        for i, switch in enumerate(switches_ts_padded[:-1]):
                if i%2 == 0:
                    ax.axvspan(switches_ts_padded[i], switches_ts_padded[i+1], alpha=0.1, color='grey')
        

def plot_bayesian_grads_comparison(plog, env, memory_buffer ):
    fig, axes = plt.subplot_mosaic([['A'],['B'],['C', ]],
                                constrained_layout=False, figsize = [12/2.53, 7/2.53])
    import matplotlib.transforms as mtransforms
    for label, ax in axes.items():
        # label physical distance to the left and up: (left, up) raise up to move label up
        trans = mtransforms.ScaledTranslation(-20/72, 5/72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize='large', va='bottom', fontfamily='arial',weight='bold')

    logger = memory_buffer
    ax = axes['A']
    obs = np.stack(logger.timestep_data['obs']).squeeze()
    preds = np.stack(logger.timestep_data['predictions']).squeeze()
    ax.plot(obs, 'o', label='obs', markersize=0.5)
    ax.plot(preds, 'o', label='preds', markersize=0.5)
    switches_ts_padded = env.env_logger['switches_ts'] +[logger.timestep_data['timestep_i'][-1]]
    ax.legend(loc='upper right', fontsize=6, ncol=2)
    ts_before, ts_after = 20, 50
    axes_labels(ax,'timestep','')
    ax.set_xticklabels([])

    ax = axes['B']
    if len(memory_buffer.timestep_data['thalamus_grad']) > 0:
        thalamus_grads = np.stack(memory_buffer.timestep_data['thalamus_grad'])
        ax.plot(thalamus_grads.squeeze(), linewidth=0.5, label='z gradients')
    else:
        print('no thalamus grads')

    ax.plot(np.stack(plog['x given u2']), linewidth=0.5, label='P(x|u2)')
    ax.plot(np.stack(plog['x given u1']), linewidth=0.5, label='P(x|u1)')
    ax.legend(loc='upper right', fontsize=6, ncol=2)

    axes_labels(ax, 'ts', 'Z grads', ypad=-2)
    ax.set_xticklabels([])
        
    ax = axes['C']
    thalamus = np.stack(memory_buffer.timestep_data['thalamus'])
    ax.plot(thalamus.squeeze(), linewidth=0.5, label='z')
    ax.plot(plog[0.2], linewidth=0.5, label='P(u1|x)')
    ax.plot(plog[0.8], linewidth=0.5, label='P(u2|x)')
    ax.legend(loc='upper right', fontsize=6, ncol=2)

    axes_labels(ax, 'Time step', 'Z values', ypad=-1)

    for ax in axes:
            for i, switch in enumerate(switches_ts_padded[:-1]):
                if i%2 == 0:
                    axes[ax].axvspan(switches_ts_padded[i], switches_ts_padded[i+1], alpha=0.1, color='grey')
    # fig.tight_layout()
    fig_clip_off(fig)
    plt.savefig(logger.folder_name + f'Bayesian_grads_{logger.experiment_name}.jpg', dpi=figure_dpi)

def plot_bayesian_grads_comparison_zoomed_in(plog, env, memory_buffer ):
    fig, axes = plt.subplot_mosaic( [['A', 'B',  ]],
                                constrained_layout=False, figsize = [10/2.53, 4/2.53])
    import matplotlib.transforms as mtransforms
    for label, ax in axes.items():
        # label physical distance to the left and up: (left, up) raise up to move label up
        trans = mtransforms.ScaledTranslation(-20/72, 5/72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize='large', va='bottom', fontfamily='arial',weight='bold')

    logger = memory_buffer


    likeihood_scaling_factor=1.
    # ax = axes['A']
    # x1, x2 = 0, 10
    # if len(memory_buffer.timestep_data['thalamus_grad']) > 0:
    #     thalamus_grads = np.stack(memory_buffer.timestep_data['thalamus_grad'])
    #     ax.plot(range(x1,x2),thalamus_grads.squeeze()[x1:x2], linewidth=0.5,)# label=['gradient 1', 'gradient 2'])
    # else:
    #     print('no thalamus grads')

    # ax.plot(range(x1,x2),np.stack(plog['x given u1'])[x1:x2]/likeihood_scaling_factor, linewidth=0.5, label='P(x|u1)')
    # ax.plot(range(x1,x2),np.stack(plog['x given u2'])[x1:x2]/likeihood_scaling_factor, linewidth=0.5, label='P(x|u2)')
    # # ax.legend(loc='upper right', fontsize=6, ncol=2)
    # ax.legend(fontsize=6)
    # axes_labels(ax, 'Time steps', 'Grads', ypad=-2)
    # # ax.set_xticklabels([])
    # ax.set_xlim([x1, x2])
        
    ax = axes['A']
    thalamus_grads = np.stack(memory_buffer.timestep_data['thalamus_grad'])
    def get_switch_triggered(values):
        mean_ = []
        ts_before = 5
        ts_after = 5
        switches_ts = np.stack(env.env_logger['switches_ts'])
    # for i, switch in enumerate((switches_ts[1:-1:2])): # ignore the first and last switch
        for i in range(1, len(switches_ts),2): # ignore the first and last switch
            switch = switches_ts[i]
        # print('i:, switch: ', i, switch)
            if len (values[switch-ts_before:switch+ts_after]) == ts_before+ts_after:
                val = values[switch-ts_before:switch+ts_after]
                mean_.append(val)
        return mean_,ts_before,ts_after

    mean_, ts_before, ts_after = get_switch_triggered(thalamus_grads.squeeze())
    mean_bayes1, ts_before, ts_after = get_switch_triggered(np.stack(plog['x given u1']))
    mean_bayes2, ts_before, ts_after = get_switch_triggered(np.stack(plog['x given u2']))
    likeihood_scaling_factor=1.
    # ax.plot(range(-ts_before, ts_after), np.stack(mean_).T, linewidth= 0.5, alpha=0.5)
    ax.plot(range(-ts_before, ts_after), np.mean(np.stack(mean_), axis=0), linewidth= 1, linestyle= ':',)# label=['grad1','grad2'])
    ax.plot(range(-ts_before, ts_after), np.mean(np.stack(mean_bayes1)/likeihood_scaling_factor, axis=0), linewidth= 0.5,  label='P(x|u1)')
    ax.plot(range(-ts_before, ts_after), np.mean(np.stack(mean_bayes2)/likeihood_scaling_factor, axis=0), linewidth= 0.5,  label='P(x|u2)')
    # ax.legend('lower left', fontsize=4, ncol=1)
    ax.legend(fontsize = 6)
    ax.axvline(x=0, color='grey', linestyle='--', linewidth=0.5, alpha=0.8)
    ax.set_xlabel('ts from ontext Switch')
    ax.set_ylabel('z grads', fontsize=6)

    ax = axes['B']
    thalamus = np.stack(memory_buffer.timestep_data['thalamus']).squeeze()
    # thalamus = np.exp(thalamus)/np.exp(thalamus).sum(axis=1, keepdims=True)

    # [env1, env2] = env.envs.keys()
    # u1 = env.envs[env1]['kwargs']['mean'] 
    # u2 = env.envs[env2]['kwargs']['mean']
    u1 = 0.2
    u2 = 0.8
    mean_, ts_before, ts_after = get_switch_triggered(thalamus.squeeze())
    mean_bayes1, ts_before, ts_after = get_switch_triggered(np.stack(plog[u1]))
    mean_bayes2, ts_before, ts_after = get_switch_triggered(np.stack(plog[u2]))

    likeihood_scaling_factor=1.
    # ax.plot(range(-ts_before, ts_after), np.stack(mean_).T, linewidth= 0.5, alpha=0.5)
    ax.plot(range(-ts_before, ts_after), np.mean(np.stack(mean_), axis=0), linewidth= 1, linestyle= ':',)# label=['grad1','grad2'])
    ax.plot(range(-ts_before, ts_after), np.mean(np.stack(mean_bayes1)/likeihood_scaling_factor, axis=0), linewidth= 0.5,  label='P(u1|x)')
    ax.plot(range(-ts_before, ts_after), np.mean(np.stack(mean_bayes2)/likeihood_scaling_factor, axis=0), linewidth= 0.5,  label='P(u2|x)')
    # ax.legend('lower left', fontsize=6)#4, ncol=1)
    ax.legend(fontsize = 6, )#loc=(.3,1.2))
    ax.axvline(x=0, color='grey', linestyle='--', linewidth=0.5, alpha=0.8)
    ax.set_xlabel('ts from ontext Switch')
    ax.set_ylabel('z1 z2', fontsize=6)

    fig.tight_layout()
    fig_clip_off(fig)
    plt.savefig(logger.folder_name + f'Bayesian_grads_zoomed_comp{logger.experiment_name}.jpg', dpi=figure_dpi)



def plot_grads(logger, env, x1=50, x2=np.inf):
    fig, axes = plt.subplot_mosaic([['A'],['B'],['C', ],], sharex=True,
                                constrained_layout=False, figsize = [21/2.53, 8/2.53]) #[12/2.53, 7/2.53])
    import matplotlib.transforms as mtransforms
    for label, ax in axes.items():
        # label physical distance to the left and up: (left, up) raise up to move label up
        trans = mtransforms.ScaledTranslation(-23/72, 2/72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize='large', va='bottom', fontfamily='arial',weight='bold')

    ax = axes['A']
    obs = np.stack(logger.timestep_data['obs']).squeeze()
    preds = np.stack(logger.timestep_data['predictions']).squeeze()
    ax.plot(obs, 'o', label='obs', markersize=0.5, color=obs_color)
    ax.plot(preds, 'o', label='preds', markersize=0.5, color=preds_color)
    switches_ts_padded = env.env_logger['switches_ts'] +[logger.timestep_data['timestep_i'][-1]]
    # ax.legend(loc='upper right', fontsize=6, ncol=2)
    ts_before, ts_after = 20, 50
    axes_labels(ax,'','')
    # ax.set_xticklabels([])
            # check if switch is less that the len of the array context_names
    # for switch in switches_ts_padded:
    #     if switch < len(logger.timestep_data['context_names']):
    #         ax.text(switch, 0.95, logger.timestep_data['context_names'][switch], rotation=30, fontsize=5, alpha=0.5)

    scale_grads = False
    ax = axes['B']
    if len(logger.timestep_data['thalamus_grad']) > 0:
        thalamus_grads = np.stack(logger.timestep_data['thalamus_grad'])
        if scale_grads:
            thalamus_grads_nan_filtered = np.nan_to_num(thalamus_grads.squeeze())
            # scaled_thalamus_grads = (thalamus_grads_nan_filtered-thalamus_grads_nan_filtered.min())/(thalamus_grads_nan_filtered.max()-thalamus_grads_nan_filtered.min())
            # ax.plot(scaled_thalamus_grads.squeeze(), linewidth=0.5)
            grads = -thalamus_grads_nan_filtered # minus, because gradients are the slope wiht respect to loss, not accuracy. So signs are flipped.
            grads =  grads + np.abs(grads.min())-0.2
            grads = np.clip(grads, 0, np.inf)
            grads = grads/grads.max()
            ax.plot(grads.squeeze(), linewidth=0.5)
        else:
            ax.plot(thalamus_grads.squeeze(), linewidth=0.5)
    else:
        print('no thalamus grads')
    axes_labels(ax, '', 'Z grads', ypad=-2)
    # ax.set_xticklabels([])
        
    ax = axes['C']
    thalamus = np.stack(logger.timestep_data['thalamus'])
    ax.plot(thalamus.squeeze(), linewidth=0.5, )#label=['Thalamus 1', 'Thalamus 2'])
    # ax.legend(loc='upper right', fontsize=6, ncol=1)
    axes_labels(ax, 'Time step', 'Z values', ypad=-1)

    # ax = axes ['D']
    # thalamus_grads_nan_filtered = np.nan_to_num(thalamus_grads.squeeze())
    # # scaled_thalamus_grads = (thalamus_grads_nan_filtered-thalamus_grads_nan_filtered.min())/(thalamus_grads_nan_filtered.max()-thalamus_grads_nan_filtered.min())
    # grads = -thalamus_grads_nan_filtered # minus, because gradients are the slope wiht respect to loss, not accuracy. So signs are flipped.
    # grads =  grads + np.abs(grads.min())-0.2
    # grads = np.clip(grads, 0, np.inf)
    # ax.plot((thalamus.squeeze()* grads).sum(axis=1), linewidth=0.5, color= 'grey', label='Evidence')
    # axes_labels(ax, 'Time step', 'Evidence', ypad=-1)
    # ax.legend(loc='upper right', fontsize=6, ncol=2)

    for ax in axes:
        if x2 !=np.inf: axes[ax].set_xlim([x1, x2])
        for i, switch in enumerate(switches_ts_padded[:-1]):
            if i%2 == 0 and switch < x2:
                axes[ax].axvspan(switches_ts_padded[i], switches_ts_padded[i+1], alpha=0.1, color='grey')
    # fig.tight_layout()

    # Entroyp attempt
    # ax = axes['E']    
    # # H.append(- plog[u1][i]*np.log2(plog[u1][i]) - plog[u2][i]*np.log2(plog[u2][i]))
    # entropy_of_grads = -np.sum(np.nan_to_num(grads)*np.log(1e-3 + grads), axis=1)
    # ax.plot(entropy_of_grads)
    # axes_labels(ax, 'Time step', 'Entropy (Nats)')



    # fig_clip_off(fig)
    # plt.savefig(logger.folder_name + f'grads_{logger.experiment_name}.jpg', dpi=figure_dpi)

def plot_bayesian_grads_comparison(plog, env, memory_buffer ):
    fig, axes = plt.subplot_mosaic([['A'],['B'],['C', ]],
                                constrained_layout=False, figsize = [12/2.53, 7/2.53])
    import matplotlib.transforms as mtransforms
    for label, ax in axes.items():
        # label physical distance to the left and up: (left, up) raise up to move label up
        trans = mtransforms.ScaledTranslation(-23/72, 2/72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize='large', va='bottom', fontfamily='arial',weight='bold')

    logger = memory_buffer
    ax = axes['A']
    obs = np.stack(logger.timestep_data['obs']).squeeze()
    preds = np.stack(logger.timestep_data['predictions']).squeeze()
    ax.plot(obs, 'o', label='obs', markersize=0.5)
    ax.plot(preds, 'o', label='preds', markersize=0.5)
    switches_ts_padded = env.env_logger['switches_ts'] +[logger.timestep_data['timestep_i'][-1]]
    ax.legend(loc='upper right', fontsize=6, ncol=2)
    ts_before, ts_after = 20, 50
    axes_labels(ax,'timestep','')
    ax.set_xticklabels([])

    ax = axes['B']
    if len(memory_buffer.timestep_data['thalamus_grad']) > 0:
        thalamus_grads = np.stack(memory_buffer.timestep_data['thalamus_grad'])
        ax.plot(thalamus_grads.squeeze(), linewidth=0.5,)# label='gradients')
    else:
        print('no thalamus grads')
    likelihood_scaling_factor= 1
    ax.plot(np.stack(plog['x given u1'])/likelihood_scaling_factor, linewidth=0.5, label='P(x|u1)')
    ax.plot(np.stack(plog['x given u2'])/likelihood_scaling_factor, linewidth=0.5, label='P(x|u2)')
    ax.legend(loc='upper right', fontsize=6, ncol=2)

    axes_labels(ax, '', 'Z grads', ypad=-2)
    ax.set_xticklabels([])
        
    ax = axes['C']
    thalamus = np.stack(memory_buffer.timestep_data['thalamus'])
    ax.plot(thalamus.squeeze(), linewidth=0.5,)# label='thalamus')
    ax.plot(plog[0.2], linewidth=0.5, label='P(u1|x)')
    ax.plot(plog[0.8], linewidth=0.5, label='P(u2|x)')
    ax.legend(loc='upper right', fontsize=6, ncol=2)

    axes_labels(ax, 'Time step', 'Z values', ypad=-1)

    for ax in axes:
            for i, switch in enumerate(switches_ts_padded[:-1]):
                if i%2 == 0:
                    axes[ax].axvspan(switches_ts_padded[i], switches_ts_padded[i+1], alpha=0.1, color='grey')
    # fig.tight_layout()
    fig_clip_off(fig)
    plt.savefig(logger.folder_name + f'Bayesian_grads_{logger.experiment_name}.jpg', dpi=figure_dpi)


def plot_bayesian(plog, env, logger, config=None):
    fig, axes = plt.subplot_mosaic([['A'],['B'],['C', ], ['D'], ['E']], sharex=False,
                                constrained_layout=False, figsize = [12/2.53, 10/2.53])
    import matplotlib.transforms as mtransforms
    for label, ax in axes.items():
        # label physical distance to the left and up: (left, up) raise up to move label up
        trans = mtransforms.ScaledTranslation(-24/72, 1/72, fig.dpi_scale_trans)
        ax.text(0.0, 1.0, label, transform=ax.transAxes + trans,
            fontsize='large', va='bottom', fontfamily='arial',weight='bold')
    if config is not None:
        u1 = config.env_kwargs[config.env_names[0]]['mean'] 
        u2 = config.env_kwargs[config.env_names[1]]['mean']
    else:
        u1 = 0.2
        u2 = 0.8

    ax = axes['A']
    obs = np.stack(logger.timestep_data['obs']).squeeze()
    preds = np.stack(logger.timestep_data['predictions']).squeeze()
    ax.plot(obs, 'o', label='obs', markersize=0.5)
    switches_ts_padded = env.env_logger['switches_ts'] +[logger.timestep_data['timestep_i'][-1]]
    ax.legend(loc='upper right', fontsize=6, ncol=2)
    axes_labels(ax,'','Observations')
    save_obs_only = False # just to create a zoomed in figure of the observations
    if save_obs_only:
        axes_labels(ax,'Time steps','Observations')
        for ax in axes:
            for i, switch in enumerate(switches_ts_padded[:-1]):
                if i%2 == 0:
                    axes[ax].axvspan(switches_ts_padded[i], switches_ts_padded[i+1], alpha=0.1, color='grey')
        fig.tight_layout()
        plt.savefig(logger.folder_name + f'Bayesian_obs_only_{logger.experiment_name}.jpg', dpi=figure_dpi)

    ax.set_xticklabels([])

    ax = axes['B']
    ax.plot(np.stack(plog['x given u1']), linewidth=0.5, label='P(x|C=0)', color='tab:green')
    ax.plot(np.stack(plog['x given u2']), linewidth=0.5, label='P(x|C=1)', color='tab:red')
    ax.legend(loc='upper right', fontsize=6, ncol=2)

    axes_labels(ax, '', 'Likelihoods', ypad=-2)
    ax.set_xticklabels([])
    
    ax = axes['C']
    ax.plot(plog[u1], linewidth=0.5, label='P(C=0|x)', color='tab:green')
    ax.plot(plog[u2], linewidth=0.5, label='P(C=1|x)', color='tab:red')
    ax.legend(loc='upper right', fontsize=6, ncol=2)
    axes_labels(ax, '', 'Posteriors', ypad=-1)
    ax.set_xticklabels([])

        
    ax = axes['D']
    ax.plot(np.stack(plog['normalizing constant']), linewidth=0.5, label='P(x)', color='grey')
    ax.legend(loc='upper right', fontsize=6, ncol=2)

    axes_labels(ax,'','Evidence')
    ax.set_xticklabels([])

    switches_ts_padded = env.env_logger['switches_ts'] +[logger.timestep_data['timestep_i'][-1]]
    
    ax = axes['E']
    H = []
    horizon = 1
    
    [H.append(np.nan) for i in range(horizon)]
    for i in range(horizon, len(obs)-2):
        H.append(- plog[u1][i]*np.log2(plog[u1][i]) - plog[u2][i]*np.log2(plog[u2][i]))
        # H.append(-0.5*np.log(2*np.pi*s1**2) - plog[u1][i]*np.log(plog[u1][i]) - 0.5*np.log(2*np.pi*s2**2) - plog[u2][i]*np.log(plog[u2][i]))
    ax.plot(H)
    # ax.set_title('Entropy of H(x,u) = -P(x,u)log(P(x,u))')
    axes_labels(ax, 'Time step', 'Entropy (Nats)')
    # ax.set_xticklabels(x_ticks)

    for ax in axes:
            for i, switch in enumerate(switches_ts_padded[:-1]):
                if i%2 == 0:
                    axes[ax].axvspan(switches_ts_padded[i], switches_ts_padded[i+1], alpha=0.1, color='grey')
    # fig.tight_layout()
    fig_clip_off(fig)
    plt.savefig(logger.folder_name + f'Bayesian_{logger.experiment_name}.jpg', dpi=figure_dpi)


