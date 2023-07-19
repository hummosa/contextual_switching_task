from logger import Logger
import torch
import numpy as np

def test_adaptation(model, env, config, criterion, ts_in_training, logger):
    memory_buffer = Logger(config)
    accuracies = []
    _use_oracle = False #config.use_oracle
    hidden = model.hidden
    obs, reward, done, info = env.reset()
    obs = torch.from_numpy(obs).float().to(model.device)
    thalamic_inputs = None
    if _use_oracle: 
        thalamic_inputs = torch.from_numpy(info['context_oh']).float().to(model.device)
    else:
        thalamic_inputs = torch.ones(obs.shape[0], obs.shape[1], config.thalamus_size).float().to(model.device)/config.thalamus_size
    # interact with the env for testing_timesteps but no parameter updates
    # for t in range(config.testing_timesteps):
    while not done:
        model.hidden = (hidden[0].detach(),hidden[1].detach()  )
        output, hidden = model(input=obs, hidden= hidden, reward = None, thalamic_inputs=thalamic_inputs)
        obs, reward, done, info = env.step(output.detach().cpu().numpy())
        obs = torch.tensor(obs).float().to(model.device)

        if _use_oracle: 
            thalamic_inputs = torch.from_numpy(info['context_oh']).float().to(model.device)
        else:
            thalamic_inputs = torch.ones(obs.shape[0], obs.shape[1], config.thalamus_size).float().to(model.device)/config.thalamus_size
        config.use_LUs= True
        if config.use_LUs: 
            model.LU_optimizer.zero_grad()
            loss = criterion(output, obs)
            loss.backward()
            model.LU_optimizer.step()

            # model.hidden= (hidden[0].detach(),hidden[1].detach())
            # hidden= (hidden[0].detach(),hidden[1].detach())
            # model.thalamus = model.thalamus.detach()

            # update info with the gradient of the model thalamus
            info.update({'thalamus_grad': model.thalamus.grad.detach().cpu().numpy()})
            info.update({'thalamus': model.thalamus.detach().cpu().numpy()})

        # store obs, preds, reward and hidden state in a memory buffer
        info.update({'predictions': output.detach().cpu().numpy(), 
        'hidden': (hidden[0].detach(),hidden[1].detach()), 'loss':  criterion(output, obs).item(),            })
        memory_buffer.log_all(info)

        accuracies.append(np.mean(memory_buffer.timestep_data['accuracy']))
    logger.log_tests({'accuracies': accuracies, 'timestep_i': ts_in_training})
    return (accuracies, memory_buffer)

def adapt_model(model, env,  _use_oracle, config, optimizer, horizon, criterion, ts_in_training, logger, _use_optimized_thalamus=False):
    memory_buffer = Logger(config)
    memory_buffer.horizon = horizon
    accuracies = []
    _use_oracle = _use_oracle
    # _use_oracle = True
    obs, reward, done, info = env.reset()
    obs = torch.from_numpy(obs).float().to(model.device)
    thalamic_inputs = None
    if _use_oracle: 
        thalamic_inputs = torch.from_numpy(info['context_oh']).float().to(model.device)
    else:
        thalamic_inputs = torch.ones(obs.shape[0], obs.shape[1], config.thalamus_size).float().to(model.device)/config.thalamus_size
        # keep optimized values.
        
    # interact with the env for testing_timesteps but no parameter updates
    # for t in range(config.testing_timesteps):
    while not done:
        if len(memory_buffer.timestep_data['timestep_i']) < 1: #memory_buffer.horizon:        
            # interact with the env to fill the memory buffer
            output, hidden = model(input=obs, reward = None, thalamic_inputs=thalamic_inputs)
            obs, reward, done, info = env.step(output.detach().cpu().numpy())
            obs = torch.tensor(obs).float().to(model.device)
            nan_md_grads = np.empty_like(model.thalamus.detach().cpu().numpy())
            nan_md_grads.fill(np.nan)
            info.update({'thalamus_grad': nan_md_grads})
            info.update({'thalamus': model.thalamus.detach().cpu().numpy()})
            info.update({'predictions': output.detach().cpu().numpy(), 
            'hidden': [h.detach().cpu().numpy() for h in hidden], 'loss':  criterion(output[-1:], obs).item(),            })
            memory_buffer.log_all(info)
        else:
            # use the memory buffer to predict the next output
            horizon_obs = memory_buffer.timestep_data['obs'][-memory_buffer.horizon:]
            horizon_obs = torch.from_numpy(np.stack(horizon_obs).squeeze(1)).float().to(model.device)

            model.hidden= (hidden[0].detach(),hidden[1].detach())

            output, hidden = model(input=horizon_obs, reward = None, thalamic_inputs=thalamic_inputs,)
            obs, reward, done, info = env.step(output[-1:].detach().cpu().numpy())
            obs = torch.tensor(obs).float().to(model.device)
            if _use_oracle: 
                thalamic_inputs_current = torch.from_numpy(info['context_oh']).float().to(model.device).unsqueeze(0)
                # gather the last horizon context and use it as thalamic input
                thalamic_inputs_buffer = torch.from_numpy(np.stack(memory_buffer.timestep_data['context_oh'][-memory_buffer.horizon:])).float().to(model.device)
                # print shapes
                thalamic_inputs = torch.cat((thalamic_inputs_buffer[1:], thalamic_inputs_current), axis=0)
            else:
                if _use_optimized_thalamus:
                    # NOTE Should I be passing these? it had accounted for previous obs, so it should use that knowledge
                    thalamic_inputs_current = (model.thalamus.detach()[-1:]).float().to(model.device)
                    thalamic_inputs_buffer = torch.from_numpy(np.stack(memory_buffer.timestep_data['thalamus'][-memory_buffer.horizon:]).squeeze(1)).float().to(model.device)
                    print(f'thalamic_inputs_buffer.shape: {thalamic_inputs_buffer.shape}')
                    print(f'thalamic_inputs_current.shape: {thalamic_inputs_current.shape}')

                    thalamic_inputs = torch.cat((thalamic_inputs_buffer[1:], thalamic_inputs_current), axis=0)
                else:
                    thalamic_inputs = torch.ones(obs.shape[0], obs.shape[1], config.thalamus_size).float().to(model.device)/config.thalamus_size
            config.use_LUs= True
            if config.use_LUs: 
                optimizer.zero_grad()
                target = torch.cat((horizon_obs[1:], obs), axis=0)
                if target.shape[0] != output.shape[0]:
                    raise ValueError('target shape is shorter than output shape')
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            # update info with the gradient of the model thalamus
            if model.thalamus.grad is not None: info.update({'thalamus_grad': model.thalamus.grad[-1:].detach().cpu().numpy()})
            info.update({'thalamus': model.thalamus[-1:].detach().cpu().numpy()})
            info.update({'predictions': output[-1:].detach().cpu().numpy(), 
            'hidden': [h.detach().cpu().numpy() for h in hidden], 'loss':  criterion(output[-1:], obs).item(),            })

            memory_buffer.log_all(info)

        accuracies.append(0)
        # accuracies.append(np.mean(memory_buffer.timestep_data['accuracy']))
    logger.log_tests({'accuracies': accuracies, 'timestep_i': ts_in_training})
    return (memory_buffer, np.stack(memory_buffer.timestep_data['loss']))

def run_experiments_from_dict(Experiments_to_run):
    import torch.nn as nn
    import torch.optim as optim
    from utils.logger import Logger
    from utils.plot_functions import plot_behavior, plot_grads
    from bayesian_task.configs import Config
    from models.model_lstm import LSTM_model
    import tasks.generative_1d
    from tasks.generative_1d import Generative_playground

    for model_name, experiments in Experiments_to_run.items():
        for experiment in experiments:
            print('running model: ', model_name)
            print('running experiment: ', experiment)
            _use_oracle = experiment['_use_oracle']
            horizon = experiment['horizon']
            use_optimized_thalamus = experiment['use_optimized_thalamus']
            adapt_env = experiment['env']
            if adapt_env == 0: # training. Only reset the model in training
                config = Config()
                config.use_oracle = _use_oracle
                config.use_optimized_thalamus = use_optimized_thalamus
                scale = 10
                training_phase_1_config = {'context_transition_function':'geometric',
                    'max_trials_per_block':int(scale*5), 'min_trials_per_block':int(scale*2),
                    'context_switch_rate': 1/(scale*(5-3)), }
                config.training_phases= [{'start_ts': 1, 'config': training_phase_1_config}]
                model = LSTM_model(config,hidden_size=100)
            
            logger = Logger(config)
            env = Generative_playground(config, adapt_env=adapt_env) # 0 for training 1 for testing 2 for novel
            logger.experiment_name = f'generative_1d_{model_name}_or{float(config.use_oracle)}_th{float(config.use_optimized_thalamus)}_hor{horizon}_env{adapt_env}'
            logger.folder_name = f'./exports/automated/'
            WU_optimizer = optim.Adam(model.parameters(), lr=0.001)
            ts_in_training= 0
            if adapt_env == 0:
                optimizer = WU_optimizer
                logger, losses = adapt_model_v2(model, env, _use_oracle, config, optimizer, horizon, nn.MSELoss(), ts_in_training, logger, _use_optimized_thalamus=use_optimized_thalamus)
            else:
                optimizer = torch.optim.SGD([model.get_parameter('thalamus')], lr=.50, momentum=0.5)
                criterion = nn.MSELoss(reduction='sum')
                logger, losses = adapt_model_v2(model, env, _use_oracle, config, optimizer, horizon, criterion, ts_in_training, logger, _use_optimized_thalamus=use_optimized_thalamus)
            plot_behavior(logger, env, losses, config, _use_oracle)
            if model_name not in ['RNN'] and adapt_env != 0:
                plot_grads(logger, env )
    return(logger, env, model, config)

def adapt_model_v2(model, env,  _use_oracle, config, optimizer, horizon, criterion, ts_in_training, logger, _use_optimized_thalamus=False, bayesian_likelihoods = None, bayesian_posteriors = None):
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

def test_inference(model, env, config, criterion, ts_in_training, logger, horizon):
    memory_buffer = Logger(config)
    memory_buffer.horizon = horizon
    accuracies = []
    _use_oracle = False #config.use_oracle
    # _use_oracle = True
    obs, reward, done, info = env.reset()
    obs = torch.from_numpy(obs).float().to(model.device)
    thalamic_inputs = None
    if _use_oracle: 
        thalamic_inputs = torch.from_numpy(info['context_oh']).float().to(model.device)
    else:
        thalamic_inputs = torch.ones(obs.shape[0], obs.shape[1], config.thalamus_size).float().to(model.device)/config.thalamus_size
    # for t in range(config.testing_timesteps):
    while not done:
        if len(memory_buffer.timestep_data['timestep_i']) < memory_buffer.horizon:        
            # interact with the env to fill the memory buffer
            output, hidden = model(input=obs, reward = None, thalamic_inputs=thalamic_inputs)
            obs, reward, done, info = env.step(output.detach().cpu().numpy())
            obs = torch.tensor(obs).float().to(model.device)
            if model.thalamus.grad is not None: info.update({'thalamus_grad': model.thalamus.grad.detach().cpu().numpy()})
            info.update({'thalamus': model.thalamus.detach().cpu().numpy()})
            info.update({'predictions': output.detach().cpu().numpy(), 
            'hidden': [h.detach().cpu().numpy() for h in hidden], 'loss':  criterion(output[-1:], obs).item(),            })
            memory_buffer.log_all(info)
        else:
            # use the memory buffer to predict the next output
            horizon_obs = memory_buffer.timestep_data['obs'][-memory_buffer.horizon:]
            horizon_obs = torch.from_numpy(np.stack(horizon_obs).squeeze(1)).float().to(model.device)

            model.hidden= (hidden[0].detach(),hidden[1].detach())

            output, hidden = model(input=horizon_obs, reward = None, thalamic_inputs=thalamic_inputs,)
            obs, reward, done, info = env.step(output[-1:].detach().cpu().numpy())
            obs = torch.tensor(obs).float().to(model.device)
            if _use_oracle: 
                thalamic_inputs = torch.from_numpy(info['context_oh']).float().to(model.device)
            else:
                thalamic_inputs = torch.ones((horizon_obs.shape[0],horizon_obs.shape[1], config.thalamus_size)).float().to(model.device)/config.thalamus_size
            config.use_LUs= True
            if config.use_LUs: 
                model.LU_optimizer.zero_grad()
                target = torch.cat((horizon_obs[1:], obs), axis=0)
                if target.shape[0] != output.shape[0]:
                    raise ValueError('target shape is shorter than output shape')
                loss = criterion(output, target)
                loss.backward()
                model.LU_optimizer.step()

            # update info with the gradient of the model thalamus
            if model.thalamus.grad is not None: info.update({'thalamus_grad': model.thalamus.grad.detach().cpu().numpy()})
            info.update({'thalamus': model.thalamus[-1:].detach().cpu().numpy()})

            # store obs, preds, reward and hidden state in a memory buffer
            # info.update({'predictions': output[-1:].detach().cpu().numpy(), 
            info.update({'predictions': output[-1:].detach().cpu().numpy(), 
            'hidden': [h.detach().cpu().numpy() for h in hidden], 'loss':  criterion(output[-1:], obs).item(),            })
            memory_buffer.log_all(info)

        accuracies.append(0)
        # accuracies.append(np.mean(memory_buffer.timestep_data['accuracy']))
    logger.log_tests({'accuracies': accuracies, 'timestep_i': ts_in_training})
    return (accuracies, memory_buffer)


    
# train the model on the generative playground task
def train_model(model, config, env, optimizer, criterion, logger, epochs=5000):
    model.train()
    thalamic_inputs = None
    losses = []
    obs, reward, done, info = env.reset() # just to get obs initialized 
    if config.use_oracle: thalamic_inputs = torch.from_numpy(info['context_oh']).float().to(model.device)
    obs = torch.from_numpy(obs).float().to(model.device)
    for epoch in range(epochs):
        optimizer.zero_grad()
        for t in range(config.no_of_timesteps_per_trial):
            output, hidden = model(input=obs, reward = None, thalamic_inputs=thalamic_inputs)
            obs, reward, done, info = env.step(output.detach().cpu().numpy())
            if config.use_oracle: thalamic_inputs = torch.from_numpy(info['context_oh']).float().to(model.device)
            info.update({'predictions': output.detach().cpu().numpy()})
            # logger.log_timestep(info)
            logger.log_all(info)
            obs = torch.tensor(obs).float().to(model.device)
            loss = criterion(output, obs)
            loss.backward()
            optimizer.step()
            model.reset()
            losses.append(loss.item())
        if epoch % 1000 == 0:
            print('Epoch: {}, Loss: {}'.format(epoch, loss.item()))

    return np.stack(losses)