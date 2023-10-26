from logger import Logger
import torch
import numpy as np

from logger import Logger
import torch
import numpy as np


def adapt_model_v2(model, env,  _use_oracle, config, optimizer, horizon, criterion, ts_in_training, logger, _use_optimized_thalamus=False, bayesian_likelihoods = None, bayesian_posteriors = None, prespecified_thalamus = None, use_buffer_thalamus = False, input_distort = False):
    """
    Adapt the model to the current environment. Adaptation can be by changing the model parameters (traditional training) or by changing the thalamus (MD) inputs (gradient-based inference).
    This depends on the optimizer passed to this function and what parameters it is optimizing. 

    The buffer solves the following problem:
    I think of these experiments as models of online adaptation, so I would like the models to update their parameters with each timestep, to model animal updating their beliefs with each new observation.
    However, if I run the model for a single timestep and do backpropagation, the model will only update its parameters based on that single timestep. It will never discover any causal structure in the environment that is beyond a single timestep.
    To solve this, I run the model for a horizon of timesteps, and then backpropagate the loss from the last timestep to the first. This way, the model can discover causal structure that spans the horizon.
    
    Importantly, I keep the horizon shorter than the block length. Backpropagating through multiple different tasks is not realistic in any challenging set of tasks. THis makes this model a good simplified model of online adaptation, and continual learning in simple setting. 
    """
    logger.horizon = horizon
    accuracies = []
    _use_oracle = _use_oracle
    # start with unifrom obs, and thalamus
    obs, reward, done, info = np.ones((config.state_size[0], 1, config.state_size[1]))/config.state_size[1], 0, False, {}
    obs = torch.from_numpy(obs).float().to(model.device)
    thalamic_inputs = torch.ones(obs.shape[0], obs.shape[1], config.thalamus_size).float().to(model.device)/config.thalamus_size
    
    # init the buffer: 
    output, hidden = model(input=obs, reward = None, thalamic_inputs=thalamic_inputs)
    obs, reward, done, info = env.step(output.detach().cpu().numpy())
    obs = torch.tensor(obs).float().to(model.device)
    nan_md_grads = np.empty_like(model.thalamus.detach().cpu().numpy())
    nan_md_grads.fill(np.nan)
    info.update({'thalamus_grad': nan_md_grads})
    info.update({'thalamus': model.thalamus.detach().cpu().numpy()})
    info.update({'predictions': output.detach().cpu().numpy(), 
    'hidden': [h.detach().cpu().numpy() for h in hidden], 'loss':  criterion(output[-1:], obs).item(),            })
    logger.log_all(info)

    while not done:
        horizon_obs = logger.timestep_data['obs'][-logger.horizon:]
        horizon_obs = torch.from_numpy(np.stack(horizon_obs).squeeze(1)).float().to(model.device)

        if type(hidden) == tuple: # LSTM with tuple of hidden, cell states
            model.hidden= (hidden[0].detach(),hidden[1].detach())
        elif type(hidden) == torch.Tensor: # RNN
            model.hidden= hidden.detach()
            
        if input_distort:
            horizon_obs = horizon_obs + torch.randn_like(horizon_obs)*0.4
        output, hidden = model(input=horizon_obs, reward = None, thalamic_inputs=thalamic_inputs,)
        obs, reward, done, info = env.step(output[-1:].detach().cpu().numpy())
        obs = torch.tensor(obs).float().to(model.device)

        # prepare the thalamic inputs for the next timestep.
        # model expects a tensor of shape (horizon, batch_size, thalamus_size)
        # or (1, batch_size, thalamus_size) if a slow evolving thalamus (level II) and it will be repeated

        # Timing and order is very tricky here. 
        # So first obs and thalamus are uniform. 
        # Model makes a prediction about the next obs,
        # We step the env to get that next obs (referred to as obs).
        # Thalamus input should be for this next obs. The env provides it during learning the model.
        if _use_oracle: 
            thalamic_inputs_current = torch.from_numpy(info['context_oh']).float().to(model.device)
            # I had to add the +1 here because I'm including all the buffer AND the current value below, so asking one less from buffer.
            thalamic_inputs_buffer = torch.from_numpy(np.stack(logger.timestep_data['context_oh'][-logger.horizon+1:]).squeeze(1)).float().to(model.device)
            thalamic_inputs = torch.cat((thalamic_inputs_buffer, thalamic_inputs_current), axis=0)
        elif _use_optimized_thalamus:
            # NOTE Should I be passing these? it had accounted for previous obs, so it should use that knowledge
            thalamic_inputs_current = (model.thalamus.detach()[-1:]).float().to(model.device)
            thalamic_inputs_buffer = torch.from_numpy(np.stack(logger.timestep_data['thalamus'][-logger.horizon+1:]).squeeze(1)).float().to(model.device)
            # print(f'thalamic_inputs_buffer.shape: {thalamic_inputs_buffer.shape}')
            # print(f'thalamic_inputs_current.shape: {thalamic_inputs_current.shape}')
            thalamus_level_II = True
            if thalamus_level_II: # use the same value for all timesteps, higher level thalamus, slower timescale.
                thalamic_inputs = thalamic_inputs_current
            else: # use one value per timestep
                thalamic_inputs = torch.cat((thalamic_inputs_buffer, thalamic_inputs_current), axis=0)
        elif use_buffer_thalamus: # grab values from the buffer, though these are delayed by one timestep, so add one uniform value at the end
            thalamic_inputs_buffer = torch.from_numpy(np.stack(logger.timestep_data['thalamus'][-logger.horizon+1:]).squeeze(1)).float().to(model.device)
            thalamic_inputs = torch.cat((thalamic_inputs_buffer, torch.ones(1,1,config.thalamus_size).float().to(model.device)/config.thalamus_size), axis=0)
        elif prespecified_thalamus is not None: # take thalamic values inferred from previous round, but advance them by a step because they are helping predict the next value?
            # or current obs is to be predicted from the last thalamic value?
            current_ts = len(logger.timestep_data['obs']) +1 # this current_ts is the one already predicted. We need the next thalamus for the current obs that is to be predicted yet
            thalamic_inputs = torch.from_numpy(prespecified_thalamus[max(0,current_ts-horizon): current_ts].squeeze(1)).float().to(model.device)
        elif bayesian_posteriors is not None:
            current_ts = len(logger.timestep_data['obs'])
            thalamic_inputs = torch.from_numpy(bayesian_posteriors[current_ts:current_ts+1]).float().to(model.device)
        else: # if nothing else. use a uniform thalamus
            thalamic_inputs = torch.ones(obs.shape[0], obs.shape[1], config.thalamus_size).float().to(model.device)/config.thalamus_size
        
        optimizer.zero_grad()
        target = torch.cat((horizon_obs, obs), axis=0)[1:]
        if target.shape[0] != output.shape[0]:
            raise ValueError('target shape is shorter than output shape')
        if config.l2_loss:
            loss = criterion(output, target) + float(config.l2_loss) * torch.norm(model.thalamus)
        else:
            loss = criterion(output, target) 
        loss.backward()
        if config.gradient_clipping > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clipping)
        if bayesian_likelihoods is not None:
            thalamus_grads = torch.from_numpy(bayesian_likelihoods).float().to(model.device)
            model.thalamus.grad = thalamus_grads
        optimizer.step()

        # update info with the gradient of the model thalamus. Since model.thlalamus is len seq, take only last timestep
        if model.thalamus.grad is not None: info.update({'thalamus_grad': model.thalamus.grad[-1:].detach().clone().cpu().numpy()})
        info.update({'thalamus': model.thalamus[-1:].detach().cpu().numpy()})
        info.update({'predictions': output[-1:].detach().cpu().numpy(), 
        'hidden': [h.detach().cpu().numpy() for h in hidden], 'loss':  criterion(output[-1:], obs).item(),            })

        logger.log_all(info)

        accuracies.append(0)
    logger.log_tests({'accuracies': accuracies, 'timestep_i': ts_in_training})
    return (logger, np.stack(logger.timestep_data['loss']))


def adapt_model_v3(model, env,  _use_oracle, config, optimizer_to_use, horizon, criterion, ts_in_training, logger, _use_optimized_thalamus=False, bayesian_likelihoods = None, bayesian_posteriors = None, prespecified_thalamus = None, use_buffer_thalamus = False, input_distort = False):
    """
    Adapt the model to the current environment. Adaptation can be by changing the model parameters (traditional training) or by changing the thalamus (MD) inputs (gradient-based inference).
    This depends on the optimizer passed to this function and what parameters it is optimizing. 

    The buffer solves the following problem:
    I think of these experiments as models of online adaptation, so I would like the models to update their parameters with each timestep, to model animal updating their beliefs with each new observation.
    However, if I run the model for a single timestep and do backpropagation, the model will only update its parameters based on that single timestep. It will never discover any causal structure in the environment that is beyond a single timestep.
    To solve this, I run the model for a horizon of timesteps, and then backpropagate the loss from the last timestep to the first. This way, the model can discover causal structure that spans the horizon.
    
    Importantly, I keep the horizon shorter than the block length. Backpropagating through multiple different tasks is not realistic in any challenging set of tasks. THis makes this model a good simplified model of online adaptation, and continual learning in simple setting. 
    """
    logger.horizon = horizon
    accuracies = []
    _use_oracle = _use_oracle
    # start with unifrom obs, and thalamus
    obs, reward, done, info = np.ones((config.state_size[0], 1, config.state_size[1]))/config.state_size[1], 0, False, {}
    obs = torch.from_numpy(obs).float().to(model.device)
    thalamic_inputs = torch.ones(obs.shape[0], obs.shape[1], config.thalamus_size).float().to(model.device)/config.thalamus_size
    
    # init the buffer: 
    output, hidden = model(input=obs, reward = None, thalamic_inputs=thalamic_inputs)
    obs, reward, done, info = env.step(output.detach().cpu().numpy())
    obs = torch.tensor(obs).float().to(model.device)
    nan_md_grads = np.empty_like(model.thalamus.detach().cpu().numpy())
    nan_md_grads.fill(np.nan)
    info.update({'thalamus_grad': nan_md_grads})
    info.update({'thalamus': model.thalamus.detach().cpu().numpy()})
    info.update({'predictions': output.detach().cpu().numpy(), 
    'hidden': [h.detach().cpu().numpy() for h in hidden], 'loss':  criterion(output[-1:], obs).item(),            })
    logger.log_all(info)

    while not done:
        horizon_obs = logger.timestep_data['obs'][-logger.horizon:]
        horizon_obs = torch.from_numpy(np.stack(horizon_obs).squeeze(1)).float().to(model.device)

        if type(hidden) == tuple: # LSTM with tuple of hidden, cell states
            model.hidden= (hidden[0].detach(),hidden[1].detach())
        elif type(hidden) == torch.Tensor: # RNN
            model.hidden= hidden.detach()
        if input_distort:
            horizon_obs = horizon_obs + torch.randn_like(horizon_obs)*0.8
        output, hidden = model(input=horizon_obs, reward = None, thalamic_inputs=thalamic_inputs,)
        obs, reward, done, info = env.step(output[-1:].detach().cpu().numpy())
        obs = torch.tensor(obs).float().to(model.device)

        # prepare the thalamic inputs for the next timestep.
        # model expects a tensor of shape (horizon, batch_size, thalamus_size)
        # or (1, batch_size, thalamus_size) if a slow evolving thalamus (level II) and it will be repeated

        # Timing and order is very tricky here. 
        # So first obs and thalamus are uniform. 
        # Model makes a prediction about the next obs,
        # We step the env to get that next obs (referred to as obs).
        # Thalamus input should be for this next obs. The env provides it during learning the model.
        if _use_oracle: 
            thalamic_inputs_current = torch.from_numpy(info['context_oh']).float().to(model.device)
            # I had to add the +1 here because I'm including all the buffer AND the current value below, so asking one less from buffer.
            thalamic_inputs_buffer = torch.from_numpy(np.stack(logger.timestep_data['context_oh'][-logger.horizon+1:]).squeeze(1)).float().to(model.device)
            thalamic_inputs = torch.cat((thalamic_inputs_buffer, thalamic_inputs_current), axis=0)
        elif _use_optimized_thalamus:
            # NOTE Should I be passing these? it had accounted for previous obs, so it should use that knowledge
            thalamic_inputs_current = (model.thalamus.detach()[-1:]).float().to(model.device)
            thalamic_inputs_buffer = torch.from_numpy(np.stack(logger.timestep_data['thalamus'][-logger.horizon+1:]).squeeze(1)).float().to(model.device)
            # print(f'thalamic_inputs_buffer.shape: {thalamic_inputs_buffer.shape}')
            # print(f'thalamic_inputs_current.shape: {thalamic_inputs_current.shape}')
            thalamus_level_II = True
            if thalamus_level_II: # use the same value for all timesteps, higher level thalamus, slower timescale.
                thalamic_inputs = thalamic_inputs_current
            else: # use one value per timestep
                thalamic_inputs = torch.cat((thalamic_inputs_buffer, thalamic_inputs_current), axis=0)
        elif use_buffer_thalamus: # grab values from the buffer, though these are delayed by one timestep, so add one uniform value at the end
            thalamic_inputs_buffer = torch.from_numpy(np.stack(logger.timestep_data['thalamus'][-logger.horizon+1:]).squeeze(1)).float().to(model.device)
            thalamic_inputs = torch.cat((thalamic_inputs_buffer, torch.ones(1,1,config.thalamus_size).float().to(model.device)/config.thalamus_size), axis=0)
        elif prespecified_thalamus is not None: # take thalamic values inferred from previous round, but advance them by a step because they are helping predict the next value?
            # or current obs is to be predicted from the last thalamic value?
            current_ts = len(logger.timestep_data['obs']) +1 # this current_ts is the one already predicted. We need the next thalamus for the current obs that is to be predicted yet
            thalamic_inputs = torch.from_numpy(prespecified_thalamus[max(0,current_ts-horizon): current_ts].squeeze(1)).float().to(model.device)
        elif bayesian_posteriors is not None:
            current_ts = len(logger.timestep_data['obs'])
            thalamic_inputs = torch.from_numpy(bayesian_posteriors[current_ts:current_ts+1]).float().to(model.device)
        else: # if nothing else. use a uniform thalamus
            # thalamic_inputs = torch.ones(obs.shape[0], obs.shape[1], config.thalamus_size).float().to(model.device)/config.thalamus_size
            thalamus_timestep_no = min(horizon, horizon_obs.shape[0]+1) 
            # +1 because another obs will be added from the env to horizon next but # if horizon is full, then +1 becomes in appropriate, an obs will be added and another will be taken out
            thalamic_inputs = torch.ones(thalamus_timestep_no,horizon_obs.shape[1], config.thalamus_size).float().to(model.device)/config.thalamus_size 


        # optimize latent context (thalamus) or weights:
        optimizer = model.LU_optimizer if optimizer_to_use == 'LU' else model.WU_optimizer
        optimizer.zero_grad()
        target = torch.cat((horizon_obs, obs), axis=0)[1:]
        if target.shape[0] != output.shape[0]:
            raise ValueError('target shape is shorter than output shape')
        if config.l2_loss:
            loss = criterion(output, target) + float(config.l2_loss) * torch.norm(model.thalamus)
        else:
            loss = criterion(output, target) 
        if config.backprop_only_last_timestep:
            loss = criterion(output[-1:], target[-1:])            
        loss.backward()
        if config.gradient_clipping > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clipping)
        if bayesian_likelihoods is not None:
            thalamus_grads = torch.from_numpy(bayesian_likelihoods).float().to(model.device)
            model.thalamus.grad = thalamus_grads
        if config.accummulate_thalamus_temporally:
            if config.no_of_latents == 1:
                model.thalamus.grad[-1] += torch.sum(model.thalamus.grad, dim=0)
            elif config.no_of_latents > 1:
                latent_size = int(config.thalamus_size / config.no_of_latents)
                # reshape x to be [seq, batch, latent_size, no_of_latents]
                # then sum the gradients along the seq dimension according to the horizon len of each latent
                # then reshape back to [seq, batch, thalamus_size]
                _grads = model.thalamus.grad.reshape(model.thalamus.grad.shape[0], model.thalamus.grad.shape[1],  config.no_of_latents, latent_size,)
                for i in range(config.no_of_latents):
                    l_horizon = config.latent_accummulation_horizons[i]
                    _grads[-1:, :, i, :] += torch.sum(_grads[-l_horizon:, :, i, :], dim=0)
                model.thalamus.grad = _grads.reshape(_grads.shape[0], _grads.shape[1], config.thalamus_size)

        optimizer.step()
        if ((len(logger.timestep_data['obs'])+1) % horizon) ==0:
            # import matplotlib.pyplot as plt
            # obs and all of prespecified thalamus:
            # fig, axes = plt.subplots(1,1, figsize=[8,2]); axes.plot(horizon_obs.squeeze()); axes.plot(prespecified_thalamus.squeeze(), linewidth=0.5) ;plt.savefig('./output.jpg')    
            # thalamic input and obs:
            # fig, axes = plt.subplots(1,1, figsize=[8,2]); axes.plot(horizon_obs.squeeze()); axes.plot(thalamic_inputs.squeeze(), linewidth=0.5) ;plt.savefig('./output.jpg')
            uwhydoIhavetohavealine= False

        # update info with the gradient of the model thalamus. Since model.thlalamus is len seq, take only last timestep
        if model.thalamus.grad is not None: info.update({'thalamus_grad': model.thalamus.grad[-1:].detach().clone().cpu().numpy()})
        # if model.thalamus.grad is not None: info.update({'thalamus_grad': model.thalamus.grad.sum(0).detach().clone().cpu().numpy()})
        info.update({'thalamus': model.thalamus[-1:].detach().cpu().numpy()})
        info.update({'predictions': output[-1:].detach().cpu().numpy(), 
        'hidden': [h.detach().cpu().numpy() for h in hidden], 'loss':  criterion(output[-1:], obs).item(),            })

        logger.log_all(info)

        accuracies.append(0)
    logger.log_tests({'accuracies': accuracies, 'timestep_i': ts_in_training})
    return (logger, np.stack(logger.timestep_data['loss']))


