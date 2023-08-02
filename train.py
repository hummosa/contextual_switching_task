from logger import Logger
import torch
import numpy as np


def adapt_model_v2(model, env,  _use_oracle, config, optimizer, horizon, criterion, ts_in_training, logger, _use_optimized_thalamus=False, bayesian_likelihoods = None, bayesian_posteriors = None):
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

        model.hidden= (hidden[0].detach(),hidden[1].detach())

        output, hidden = model(input=horizon_obs, reward = None, thalamic_inputs=thalamic_inputs,)
        obs, reward, done, info = env.step(output[-1:].detach().cpu().numpy())
        obs = torch.tensor(obs).float().to(model.device)

        # prepare the thalamic inputs for the next timestep.
        # model expects a tensor of shape (horizon, batch_size, thalamus_size)
        # or (1, batch_size, thalamus_size) if a slow evolving thalamus (level II) and it will be repeated
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

        elif bayesian_posteriors is not None:
            current_ts = len(logger.timestep_data['obs'])
            thalamic_inputs = torch.from_numpy(bayesian_posteriors[current_ts:current_ts+1]).float().to(model.device)
        else:
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

# rewrite adapt_model_v2 but instead of predicting the next timesteps, merely train to just output the same input 
