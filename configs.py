import numpy as np
import torch

class Training_phases():
    def create_list():
        training_phase_1_config = Config()
        training_phase_2_config = Config()

class Config():
    def __init__(self, env_kwargs= None, context_units = 2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_oracle = True
        # self.use_oracle = False
        self.output_oracle = False
        self.use_reward_feedback = False
        self.context_signal_type = 'one_hot' # 'one_hot' or 'compositional'
        
        self.capture_variance_experiment = False
        self.shrew_modifications = False
        self.additional_inputs = False
        self.ablate_context_signal = False
        self.l2_loss = False
        self.gradient_clipping = 0
        self.backprop_only_last_timestep = False
        self.default_mean1 = 0.2
        self.default_mean2 = 0.8
        self.default_std = 0.2
        if env_kwargs is None:
            env_kwargs = { 'gauss1': {'mean': self.default_mean1, 'std': self.default_std},
                            'gauss2': {'mean': self.default_mean2, 'std': self.default_std}}

        # Environments
        self.env_names = [k for k in env_kwargs.keys()]
        self.env_kwargs = env_kwargs
        self.env_libarrary = {} # TODO create a list of env names classes, ids, and args.
        self.no_of_contexts = context_units #len(self.env_names)
        self.thalamus_size = self.no_of_contexts
        self.use_thalamus = False # to use thalamus withouth the oracle
        self.reward_fn = lambda y,x: x-y
        self.thalamus_activation_function = 'softmax' # 'softmax' or 'none'
        self.accummulate_thalamus_temporally = False
        
        self.batch_size = 1 # to simulate human cognition experience, but can later increase for efficiency
        self.seq_size = 1 #
        self.state_size = [1, 1] # defines the state size of the environment
        self.input_size = self.state_size[1] # Used to define the input size of the model
        if self.use_oracle:
            self.input_size += self.no_of_contexts # defined below, after context size is defined
        elif self.use_reward_feedback:
            self.input_size +=1  # 2 stimulus, 1 reward
        self.output_size = 1
        if self.output_oracle:
            self.output_size += self.no_of_contexts # defined below, after context size is defined
        self.action_size = self.output_size # alias for outpun_size 
        self.training_blocks = 20

        # Model parameters
        # self.lr  = 0.0005
        self.LU_lr = 0.3
        self.WU_lr = 0.001
        self.momentum = 0.5
        self.rnn_type = 'LSTM' # 'LSTM' or 'RNN'
        # self.lr_pi           = 0.0005
        # self.lr_pi           = 0.001
        # self.lr_q            = 0.001
        # self.latent_lr = 1e-1
        self.latent_decay = 0.9
        self.hidden_size = 64
        self.no_of_hypothesis = 5
        self.latent_shape = 10
        self.activation_fxn = 'relu'
        self.seed = 2

        #Hyperparameters
        self.gamma         = 0.98
        self.lmbda         = 0.95
        self.eps_clip      = 0.1
        self.K_epoch       = 1
        self.T_horizon     = 20
        self.entropy_coef  = 0.01
        self.value_loss_coef = 0.5
        self.data_buffer_size = 100 

        # context change dynamics:
        # self.context_transition_function = 'fixed_alternating'
        self.context_transition_function = 'geometric'
        self.alternating_contexts  = True  # Create the alternating contexts scenario in hierarchical contextual task in Halassa lab
        self.max_trials_per_block= 100
        self.min_trials_per_block= 50
        self.context_switch_rate = 50/100 #  used for geometric context transition function
        self.no_of_timesteps_per_trial= 1
        self.testing_timesteps = 100
        self.no_of_blocks = 40
        # self.rnn_steps_before_obs= 2
        # self.time_delay_penalty= -0.05
        training_phase_1_config = {'context_transition_function':'geometric',
        'max_trials_per_block':50, 'min_trials_per_block':10,
        'context_switch_rate': 0.015, 'no_of_blocks': 20}
        training_phase_2_config = {'context_transition_function':'fixed_alternating',
            'max_trials_per_block':self.max_trials_per_block, 'no_of_blocks': 4}
        self.training_phases= [{'start_ts': 1, 'config': training_phase_1_config}, {'start_ts': 2000, 'config': training_phase_2_config }]

        # adaptive testing parameters
        self.use_LUs = True

    def update(self, config_dict):
        for key, value in config_dict.items():
            setattr(self, key, value)
    def __str__(self):
        return str(self.__dict__)       

class generalization_mean_and_var_Config(Config):
    def __init__(self, env_kwargs= { 'gauss1': {'mean': 0.3, 'std': 0.3},
                                     'gauss2': {'mean': 0.7, 'std': 0.1},
                                     'gauss3': {'mean': 0.3, 'std': 0.1},
                                     'gauss4': {'mean': 0.7, 'std': 0.3}},
                                     context_units =4 ):
        super().__init__( env_kwargs= env_kwargs, context_units =context_units)
        
        # using only the envs passed in env_kwardgs. Note that the kwargs for each are passed to the environmnet class
        self.env_names = [k for k in self.env_kwargs.keys()]
        self.no_of_contexts = len(self.env_names)
        self.thalamus_size = context_units # Use provided, because I need 4 units, despite only 3 training tasks.  self.no_of_contexts

