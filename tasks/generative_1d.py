import gym
import numpy as np
from collections import defaultdict
import torch

class Generative_1d_gauss(gym.Env):
    """ 
    Defines a simple 1 d env. Agent gets obs 1 d float, and responds with action 1 d float.
    Reward is by default distance between action and next obs
    For SL, next obs is returnd and can be used for training. 
    Expects config with input_size, output_size or action_size defined. 
    Also expects kwargs with mean and std of the gaussian. 
    """
    def __init__(self, config, mean, std, **kwargs):
        super().__init__()
        # self.mean = kwargs['mean']
        # self.std = kwargs['std']
        self.mean = mean
        self.std = std
        self.input_size = config.input_size
        self.state_size = [1,1] # defined separate from input_size to allow additional context or reward input
        self.action_size, self.output_size = config.action_size, config.action_size # alias
        self.observation_space =  gym.spaces.Box(low=-np.inf, 
                                                high=np.inf,
                                                shape=(self.input_size,),
                                                dtype=np.float32)
        self.action_space =       gym.spaces.Discrete(self.action_size)
        self.action_space =       gym.spaces.Box(low=-np.inf, 
                                                high=np.inf,
                                                shape=(self.output_size,),
                                                dtype=np.float32)
        self.config = config
        # self.rng = np.random.RandomState(config.seed + int(std*1000))
        self.rng = np.random.RandomState(config.seed)
        self.current_obs = 0        

    def sample_obs(self):
        return self.rng.normal(size=(self.config.batch_size, *self.state_size), loc=self.mean, scale=self.std).astype(float)

    def step(self, action):
        self.next_obs = self.sample_obs()
        reward = self.reward_fn(self.next_obs, action)
        done = False
        info = {'correct_action': self.next_obs}
        return np.array(self.next_obs), reward, done, info

    def reward_fn (self, obs, action):
        r= np.power(obs-action, 2).astype(float)
        r = np.expand_dims(r, axis=0)
        return (r)
    def reset(self):
        self.current_obs = self.sample_obs()
        return self.current_obs.astype(float)

class Generative_1d_uniform(gym.Env):
    """ 
    Defines a simple 1 d env. Agent gets obs 1 d float, and responds with action 1 d float.
    Reward is by default distance between action and next obs
    For SL, next obs is returnd and can be used for training. 
    Expects config with input_size, output_size or action_size defined. 
    Also expects kwargs with mean and std of the gaussian, but these used as min max. Kept just for compatiability with other envs, hack hack.
    """
    def __init__(self, config, mean, std, **kwargs):
        super().__init__()
        # self.mean = kwargs['mean']
        # self.std = kwargs['std']
        self.mean = mean
        self.std = std
        self.input_size = config.input_size
        self.state_size = [1,1] # defined separate from input_size to allow additional context or reward input
        self.action_size, self.output_size = config.action_size, config.action_size # alias
        self.observation_space =  gym.spaces.Box(low=-np.inf, 
                                                high=np.inf,
                                                shape=(self.input_size,),
                                                dtype=np.float32)
        self.action_space =       gym.spaces.Discrete(self.action_size)
        self.action_space =       gym.spaces.Box(low=-np.inf, 
                                                high=np.inf,
                                                shape=(self.output_size,),
                                                dtype=np.float32)
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        self.current_obs = 0        

    def sample_obs(self):
        return self.rng.uniform(size=(self.config.batch_size, *self.state_size), low=self.mean, high= self.std ).astype(float)
        

    def step(self, action):
        self.next_obs = self.sample_obs()
        reward = self.reward_fn(self.next_obs, action)
        done = False
        info = {'correct_action': self.next_obs}
        return np.array(self.next_obs), reward, done, info

    def reward_fn (self, obs, action):
        r= np.power(obs-action, 2).astype(float)
        r = np.expand_dims(r, axis=0)
        return (r)
    def reset(self):
        self.current_obs = self.sample_obs()
        return self.current_obs.astype(float)
        
class Generative_environment(gym.Env):
    """
    Environment is meant to hold a number of tasks, and switch between them.
    It serves trials to the agent from the appropriate task.
    It takes a Config file which has listed the names of the tasks, and their constructors and kwargs.
     This part is a bit rudementary really because I have only two tasks Gaussian and Uniform, and I just hard coded them in.
    """
    def __init__(self, config, experiment=0, novel_mean=0, novel_std=0.1, **kwargs):
        super().__init__()
        self.experiment = experiment
        self.input_size = config.input_size
        self.action_size, self.output_size = config.action_size, config.action_size # alias
        self.observation_space =  gym.spaces.Box(low=-np.inf, 
                                                high=np.inf,
                                                shape=(self.input_size,),
                                                dtype=np.float32)
        self.action_space =       gym.spaces.Discrete(self.action_size)
        self.action_space =       gym.spaces.Box(low=-np.inf, 
                                                high=np.inf,
                                                shape=(self.output_size,),
                                                dtype=np.float32)
        self.config = config
        self.rng = np.random.RandomState(config.seed)
        ## Two absractions available, block is a no_of_trials_per_block trials, and trials are no_of_timesteps_per_trial timesteps
        self.no_of_timesteps_per_trial = 1 # 1 results in ignoring concept of trial

        self.no_of_tasks = len(config.env_names)

        
        # Define the tasks, and their ids, constructors and kwargs
        task_dicts = dict()
        for task_name in config.env_names:
            env_id = np.argwhere(np.array(config.env_names)==task_name)[0][0]
            if env_id >= self.config.thalamus_size:
               env_id = min(env_id, self.config.thalamus_size-1)
               print('env_id is larger than thalamus size, setting it to thalamus size-1')
            task_dicts[task_name] = {'env_id': env_id ,# config.env_names.index(task_name), 
                                    'env_constructor': Generative_1d_gauss, 
                                    'kwargs': config.env_kwargs[task_name]}
            
            
        # define the tasks to be used in the experiment
        self.experiment_tasks = config.env_names # currently using all. But it could be a subset to train on and a subset to test on.

        # instantiate the tasks
        self.envs = dict()
        for task_name in self.experiment_tasks:
            task = task_dicts[task_name]
            kwargs = task['kwargs']
            task.update({'env_instant': task['env_constructor'](self.config, **kwargs)})
            self.envs[task_name] = task

        # construct each experiment based on the experiment integer passed. Expect messiness ahead. 
        if experiment==1: 
            # self.envs= self.experiments
            pass
        elif experiment == 2: # add novel env 
            novel_env_kwargs= {'gauss4':  {'env_id': 3, 'env_constructor': Generative_1d_gauss, 'kwargs': {'mean': novel_mean, 'std': novel_std},  }}
            self.novel_envs = dict()
            for name, task in novel_env_kwargs.items():
                kwargs = task['kwargs']
                task.update({'env_instant': task['env_constructor'](self.config, **kwargs)})
                self.novel_envs[name] = task
            self.envs.update(self.novel_envs)

        elif experiment == 3: # Testing generalization systematically betwee -0.2 to 1.2
            means = np.array(list(range(-2, 13)))/10
            novel_env_kwargs= {f'gauss{i}': {'mean': means[i], 'std': novel_std} for i in range(len(means))}
            self.novel_envs = defaultdict(dict)
            for i in range(len(means)):
                self.novel_envs[f'gauss{i}'] = {'env_id':0, 'env_constructor': Generative_1d_gauss, 'kwargs': novel_env_kwargs[f'gauss{i}']}
            for env_name, task_d in self.novel_envs.items():
                kwargs = task_d['kwargs'] 
                task_d.update({'env_instant': task_d['env_constructor'](self.config, **kwargs) })
            self.envs= self.novel_envs
        elif experiment == 6: # Testing VARIANCE generalization systematically betwee 0.1 to 0.5
            # base block:
            self.novel_envs = defaultdict(dict)
            base_env_kwargs= {f'base_gauss': {'mean': self.config.default_mean1, 'std': self.config.default_std}}
            vars = np.array(list(range(0, 6)))/10
            self.novel_envs[f'base_gauss'] = {'env_id':0, 'env_constructor': Generative_1d_gauss, 'kwargs': base_env_kwargs[f'base_gauss']}
            novel_env_kwargs= {f'gauss{i}': {'mean': novel_mean, 'std': vars[i]} for i in range(len(vars))}
            for i in range(len(vars)):
                self.novel_envs[f'gauss{i}'] = {'env_id':0, 'env_constructor': Generative_1d_gauss, 'kwargs': novel_env_kwargs[f'gauss{i}']}
            for env_name, task_d in self.novel_envs.items():
                kwargs = task_d['kwargs'] 
                task_d.update({'env_instant': task_d['env_constructor'](self.config, **kwargs) })
            self.envs= self.novel_envs
        elif experiment == 4: # Testing single run generalization
            self.interesting_envs = defaultdict(dict)
            self.interesting_envs['uniform2'] = {'env_id':0, 'env_constructor': Generative_1d_uniform, 'kwargs': {'mean': 0.0, 'std': 1.0}} # Really min and max
            self.interesting_envs['gauss2'] = {'env_id':1, 'env_constructor': Generative_1d_gauss, 'kwargs': {'mean': 0.0, 'std': 0.3}} #
            self.interesting_envs['gauss1'] = {'env_id':2, 'env_constructor': Generative_1d_gauss, 'kwargs': {'mean': 0.8, 'std': 0.3}}
            # self.interesting_envs['gauss3'] = {'env_id':3, 'env_constructor': Generative_1d_gauss, 'kwargs': {'mean': 0.8, 'std': 0.6}} #
            for env_name, task_d in self.interesting_envs.items():
                kwargs = task_d['kwargs'] 
                task_d.update({'env_instant': task_d['env_constructor'](self.config, **kwargs) })
            self.envs= self.interesting_envs

        # self.novel_envs = dict({'env_name': 'env_class_constroctor'}) # samples from novel tasks not see in training
        self.no_of_tasks = len(self.envs) #  TODO need to add test and novel envs later
        ## randomly pick current context and env
        self.current_context = self.rng.choice(list(self.envs.keys()), replace=True)
        self.current_env = self.envs[self.current_context]['env_instant']
        
        ## logs:
        self.env_logger = defaultdict(list)
        
        self.trial_i = 0
        self.block_i = 0
        self.ts_i = 0
        self.trial_in_block = 0
        self.ts_in_trial = 0
        self.trials_remaining_in_block = 20 # TODO define better if this option gets used

    def context_transition_update(self):
        '''
        updates the current context, and logs the ts of the transition

        '''
        for training_phase in self.config.training_phases:
            if self.ts_i == training_phase['start_ts']:
                self.config.update(training_phase['config']) # load the relevant config for this training phase
                # self.config=training_phase['config']

        if self.config.context_transition_function == 'fixed_alternating':
            self.trial_in_block+=1
            if self.trial_in_block >= self.config.max_trials_per_block:
                context_options = list(self.envs.keys())
                # but exclude current context
                context_options.remove(self.current_context)
                self.current_context = self.rng.choice(context_options, replace=True)
                self.trial_in_block = 0
                self.block_i += 1
                self.env_logger['switches_ts'].append(self.ts_i)
        elif self.config.context_transition_function == 'geometric':
            self.trials_remaining_in_block -=1
            if self.trials_remaining_in_block==0:
                context_options = list(self.envs.keys())
                # but exclude current context
                context_options.remove(self.current_context)
                self.current_context = self.rng.choice(context_options, replace=True)
                self.trial_in_block = 0
                self.trials_remaining_in_block = np.clip(self.rng.geometric(p=self.config.context_switch_rate, ),
                    self.config.min_trials_per_block, self.config.max_trials_per_block)

                self.block_i += 1
                self.env_logger['switches_ts'].append(self.ts_i)
        elif self.config.context_transition_function == 'base_block_alternating': # alternates between some base block, and all others.
            self.trials_remaining_in_block -=1
            if self.trials_remaining_in_block==0:
                context_options = list(self.envs.keys())
                # print('self.block_i: ', self.block_i)
                if self.block_i % 2 == 0: # if even block, then use base block
                    self.current_context = 'base_gauss'
                    # print('self.current_context: ', self.current_context)
                else:
                    con_id = (self.block_i//2)+1
                    con_id = con_id % len(context_options) # loop through all contexts at the end
                    self.current_context = context_options[con_id] # use the next context in the list
                    # print('self.current_context: ', self.current_context)
                    # print('con_id: ', con_id)
                    # print('context_options: ', context_options)

                self.trial_in_block = 0
                # uses self.trials_remaining_in_block =20 above as fixed block width. 
                # self.trials_remaining_in_block = np.clip(self.rng.geometric(p=self.config.context_switch_rate, ),
                #     self.config.min_trials_per_block, self.config.max_trials_per_block)
                self.trials_remaining_in_block = self.config.min_trials_per_block
                self.block_i += 1
                self.env_logger['switches_ts'].append(self.ts_i)
        # add another context transition function that simply loops through all contexts in order, and then repeats.
        elif self.config.context_transition_function == 'sequential': # alternates between some base block, and all overs.
            self.trials_remaining_in_block -=1
            if self.trials_remaining_in_block==0:
                context_options = list(self.envs.keys())
                con_id = self.block_i % len(context_options)
                self.current_context = context_options[con_id]
                self.trial_in_block = 0
                self.trials_remaining_in_block = self.config.max_trials_per_block
                self.block_i += 1
                self.env_logger['switches_ts'].append(self.ts_i)
    
    def new_trial(self):
        self.trial_i +=1
        self.context_transition_update() ## updates current context
        self.current_env = self.envs[self.current_context]['env_instant']
        # new_trial, info = self.current_env.new_trial()

    def step(self, action):
        '''
        step function
        obs output is in torch, and has dims batch_size x input_dim1 x input_dim2   # though I never ended up using input_dim2. 
        info gathers all the relevant info for logging and debugging, but also I use it to pass the context one-hot encoded vector to the agent.
        '''
        if self.config.capture_variance_experiment or self.config.shrew_modifications:
            action = action[:,:, :1] # only use first action

        obs, reward, done, info = self.current_env.step(action)
        info.update({'obs': obs,})

        if self.ts_in_trial >= self.no_of_timesteps_per_trial: # transition context AFTER the obs, oracle context signal should be informative to predict the next obs, not this one. 
            self.new_trial()
            self.ts_in_trial = 0
        # obs = self.augment_obs(obs, self.current_context, reward)    

        # add one-hot context signal to obs
        if self.config.context_signal_type == 'one_hot':
            current_cxt_id = self.envs[self.current_context]['env_id'] # integer id for env
            oh_current_cxt = np.zeros((1, self.config.batch_size,self.config.thalamus_size))
            oh_current_cxt[0, :, current_cxt_id] = 1.0
        elif self.config.context_signal_type == 'compositional':
            # This is for the additional experiment where the models are trained on two different means, but also no two different stds. Compsitional means splitting up the context signal into two parts, one for mean and one for std. As opposed to one-hot where each combination of mean and std gets one unique value.
            # the first two dims are 1, 0 for the first two tasks, and 0, 1 for the last two tasks. the last 2 dims are 1, 0 for tasks 1 and 3, and 0, 1 for tasks 2 and 4.
            current_cxt_id = self.envs[self.current_context]['env_id'] # integer id for env
            oh_current_cxt = np.ones((1, self.config.batch_size,self.config.thalamus_size))*0.3 # avoid zeros
            current_cxt_mean = self.envs[self.current_context]['kwargs']['mean']
            current_cxt_std = self.envs[self.current_context]['kwargs']['std']
            if not self.config.ablate_context_signal:
                if current_cxt_mean < 0.5 :
                    oh_current_cxt[0, :, 0] = 1.0
                elif current_cxt_mean >= 0.5:
                    oh_current_cxt[0, :, 1] = 1.0
                else:
                    # raise ValueError('mean not found')
                    print('mean not found')
            if current_cxt_std < 0.2:
                oh_current_cxt[0, :, 2] = 1.0
            elif current_cxt_std >= 0.2:
                oh_current_cxt[0, :, 3] = 1.0
            
            # print('oh_current_cxt: ', oh_current_cxt)
            # print('current_cxt_id: ', current_cxt_id)
            # print('self.current_context: ', self.current_context)

        info.update({'context_oh': oh_current_cxt})

        info.update({'reward': reward, 'action': action, 'done': done, 'obs_augmented': obs,
        'context_names': self.current_context, 'context_id': self.envs[self.current_context]['env_id'],
        'trial_i': self.trial_i, 'timestep_trial': self.ts_in_trial, 'timestep_i': self.ts_i,})
        self.ts_i +=1
        self.ts_in_trial +=1
        total_blocks = self.config.no_of_blocks
        # if self.experiment ==3: total_blocks = 8
        done = True if self.block_i == total_blocks else False
        return obs, float(reward), done, info
    
    def reset(self):
        self.ts_in_trial = 0
        self.ts_i = 0
        self.trial_i = 0
        obs = self.current_env.reset()

        info = {}
        if self.config.use_oracle:
            current_cxt = self.envs[self.current_context]['env_id'] # integer id for env
            oh_current_cxt = np.zeros(self.no_of_tasks)
            oh_current_cxt[current_cxt] = 1.0
            info.update({'context_oh': oh_current_cxt})

        return obs, 0, False, info