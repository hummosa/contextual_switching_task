import gym
import numpy as np

class hierarchical_inference_task(gym.Env):
  """
    Takes discrete actions of size 2.
    Returns an observation space of size 1
    IBL task
  """
  def __init__(self, config):
      super().__init__()
      self.input_size = 1
      self.action_size = 2
      self.observation_space = gym.spaces.Box(low=-np.inf, 
                                              high=np.inf,
                                              shape=(self.input_size,),
                                              dtype=np.float32)
      self.action_space = gym.spaces.Discrete(self.action_size)
      self.config = config
      self.rng = np.random.RandomState(config.seed)
      self.current_context = 0
      self.current_context_i = 0
      self.current_trial_i = 0
      self.current_timestep_i = 0
      self.time_step_within_trial_i = 0
      self.trials_remaining_in_block = config.min_trials_per_block \
         + self.rng.poisson(config.context_switch_rate)
      self.correct_action = None

      # trial history loggers
      self.difficulties = []
      self.context_ids = [] # gathers the context id for each trial
      self.switches = []   # gathers switch point indexes
      self.new_trial()
      

  def new_trial(self):
    self.time_step_within_trial = 0
    self.current_trial_i +=1 
    self.trials_remaining_in_block +=-1
    if self.trials_remaining_in_block == 0: # switch context and resent trials remaining
        self.current_context = 1- self.current_context # flip context
        self.trials_remaining_in_block = self.rng.randint(self.config.min_trials_per_block, self.config.max_trials_per_block)
        self.switches.append(self.current_trial_i)

    # sample current difficulty u
    u = self.rng.choice(self.config.stimulus_strengths)# p=[0.5, 0.5])
    self.current_difficulty = u

    # sample side
    concordant_prob = self.config.concordant_trial_prob
    self.concordant_trial = self.rng.choice([0, 1], p=[1-concordant_prob, concordant_prob])
    stimulus_side = self.current_context if self.concordant_trial == 1 else 1-self.current_context

    # define correct action: actions 0, left, action 1, right,
    correct_action = stimulus_side

    # sample observations
    means = [u, 0] if stimulus_side==0 else [0,u]
    observations = self.rng.normal(loc=means, scale=1., size=(self.config.max_timesteps_per_trial,2)).astype(np.float32)
    # print('observations: ', observations)
    self.trial_observations = observations # set trial observations to consume during step calls.
    self.trial_correct_action = correct_action
    return (observations, correct_action)
  
  def step(self, action=0): # action 0 is left, action 1 is right, action 2 is no action
      self.time_step_within_trial += 1
      self.current_timestep_i +=1
      reward = 0
      info = {'trial': self.current_trial_i, 'timestep_trial': self.time_step_within_trial,
       'timestep_session': self.current_timestep_i, 'context': self.current_context, 'difficulty': self.current_difficulty, 'concordant': self.concordant_trial}
      # get current observation
      ob = self.trial_observations[self.time_step_within_trial]
      correct_action = self.trial_correct_action
      self.current_ob = ob
      
      # calculate reward
      if action == 2:  # no action
          reward = self.config.time_delay_penalty
      elif action == correct_action: # correct action
          reward = 1.
      else:
          reward = 1.
      # check if environment over
      if self.time_step_within_trial >= self.config.max_timesteps_per_trial-1:
        done = True
        self.new_trial() # generate observations in observations_stream for new trial
      else:
        done = False
      info.update({'reward': reward, 'action': action, 'correct_action': correct_action, 'done': done, 'ob': ob, })
      return ob, float(reward), done, info

  def reset(self):
      self.new_trial()
      return self.trial_observations[0]