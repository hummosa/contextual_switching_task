#tasks coded in neurogym
#%%
import numpy as np
rng = np.random.default_rng()
import matplotlib.pyplot as plt
import neurogym as ngym
import gym
from neurogym import TrialEnv
import sys; sys.path.insert(0, '.')

def get_difficulty(total_cues):
    difficulties = list(range((total_cues//2)+1-1, total_cues+1)) # 5 to 10
    difficulty_probs = [0.1]*2 + [0.2]*4
    sampled_no_of_coherent_cues = rng.choice(difficulties, p=difficulty_probs)
    return(sampled_no_of_coherent_cues)

class NoiseyMean(TrialEnv):
    def __init__(self, mean_noise= 0.1, mean_drift = 0, odd_balls_prob = 0.0, change_point_prob = 0.0, trials_remaining = 5, config = None):
        super().__init__(dt=1)

        self.observation_space = ngym.spaces.Box(low=0.0, high=1.0, shape=(1,), name = {'outcomes':[0]})
        self.latent_space = ngym.spaces.Box(low=0.0, high=1.0, shape=(1,))#, names = ['mean'])
        self.action_space = ngym.spaces.Box(low=0.0, high=1.0, shape=(1,))
        
        #get initial random values
        self.outcome_now = self.observation_space.sample()
        self.mean_now = self.latent_space.sample()

        # self.trial_length = sum(self.timing.values())

        if config == None:
            self.trial_len = 100
            self.far_definition = 0.2 *self.observation_space.__getattribute__('high')
            self.odd_balls_prob =odd_balls_prob 
            self.change_point_prob = change_point_prob
            self.mean_drift = mean_drift 
            self.mean_noise= mean_noise
        else:
            self.trial_len = 100
            self.far_definition = config.far_definition *self.observation_space.__getattribute__('high')
            self.odd_balls_prob =odd_balls_prob 
            self.change_point_prob = change_point_prob
            self.mean_drift = mean_drift 
            self.mean_noise= mean_noise

        self.timing = {'outcomes': self.trial_len-1}
        self.hazzard_rate = max(odd_balls_prob, change_point_prob)
        self.safe_trials = self.trial_len // (self.trial_len * self.hazzard_rate * 2) # get safety trials that is half of the extected inter-hazzard trials len
       
    def _new_trial(self):
        # Setting time periods for this trial
        periods = ['outcomes']
        # Will add stimulus and decision periods sequentially using self.timing info
        self.add_period(periods)
        if True: #alternate way to sample events with a safety trial built in. Measuring comp overhead.
            events_idx = rng.integers(self.safe_trials, int(self.safe_trials *3), size = int(self.trial_len))
            events_idx = np.clip(np.cumsum(events_idx), 0, self.trial_len-1)
            events = np.zeros(self.trial_len)
            events[events_idx] = 1
        else:
            events = rng.binomial(1, self.hazzard_rate, self.trial_len)
        magnitude_pos = rng.uniform(self.far_definition,0.6, self.trial_len)
        # Sample events, or jumps, negative or positive, and
        magnitude_neg = rng.uniform(-0.6,-self.far_definition, self.trial_len)
        # Randomly chose between them, negative or postiive
        magnitude = np.choose(rng.binomial(1, 0.5, self.trial_len), [magnitude_neg, magnitude_pos])
        # Sample noise around the ongoing mean
        gaussian_samples = rng.normal( 0, self.mean_noise, size=(self.trial_len))
        # if self.change_point_prob > 0.: # WHy am i doing this?? It seems like I'm cumsuming the magnitudes below
            # add to the drifts around the mean the jumpy magnitudes according to the sampled events
            # gaussian_samples[events.astype(bool)] += magnitude[events.astype(bool)]
        if self.mean_drift > 0.:
            # gaussian_samples[0] = rng.uniform(0., 1.0)
            gaussian_samples = np.cumsum(gaussian_samples)
        else:
            if self.change_point_prob > 0.: 
                zero_samples = np.zeros(self.trial_len)
                zero_samples[events.astype(bool)] += magnitude[events.astype(bool)]
                gaussian_samples += np.cumsum(zero_samples) # adding change points to noisy mean
            # gaussian_samples+= rng.uniform(0., 1.0) # Shift all the samples to avoid always starting at 0
        if self.odd_balls_prob > 0.: # this relies on having some mean drift to add 
            #off set from 0 and also to cumsum the gaussian samples.
            gaussian_samples[events.astype(bool)] += magnitude[events.astype(bool)] # adding magnitudes of the oddballs

        coerce_to_0_to_1 = False
        if coerce_to_0_to_1:
            # reflect outcomes that go above one a couple of times to encourage most trials to remain 0-1
            _outcomes = abs(gaussian_samples)
            gaussian_samples[gaussian_samples>1] = 2- gaussian_samples[gaussian_samples>1]     
            _outcomes = abs(gaussian_samples)
            gaussian_samples[gaussian_samples>1] = 2- gaussian_samples[gaussian_samples>1]     
            _outcomes = abs(gaussian_samples)
            gaussian_samples[gaussian_samples>1] = 2- gaussian_samples[gaussian_samples>1]     
            _outcomes = abs(gaussian_samples)
        else:
            _outcomes = gaussian_samples

        _outcomes = _outcomes.reshape([-1, 1])
        trial = dict()
        # Ground-truth is 1 if ob > 0, else 0
        trial['outcomes'] = _outcomes[:-1, :]
        # trial['means'] = np.stack(means)
        trial['oddballs'] = events
        trial['changepoints'] = events
        trial['ground_truth'] = _outcomes[1:, :]


        self.add_ob(trial['outcomes'], period='outcomes', where='outcomes')
        self.set_groundtruth(trial['ground_truth'])
        return trial
    
    def _step(self, action):
        ob = self.next_ob
        # If action equals to ground_truth, reward=1, otherwise 0
        reward = (action == self.trial['ground_truth']) * 1.0
        done = False
        info = {'new_trial': True}
        return ob, reward, done, info


class Shrew_task(TrialEnv):
    def __init__(self, dt=10, attend_to = 'either', context= 1, no_of_coherent_cues = None, timing=None):
        super().__init__(dt=dt)  # dt is passed to base task
        
        # Setting default task timing
        self.timing = {'cues': 200, 'delay': 50, 'stimulus': 30, 'decision': 20}
        self.trial_length = sum(self.timing.values())
        self.context = context
        self.no_of_coherent_cues = no_of_coherent_cues
        self.total_cues = 10
        self.probabilistic = False
        self.input_uncertainty = False
        self.use_oddballs = False
        # Update timing if provided externally
        if timing:
            self.timing.update(timing)
        
        self.attend_to = attend_to # restrict trials to only one modality or both  {audition, vision, or either}.
        # Here we use ngym.spaces, which allows setting name of each dimension
        self.observation_space = ngym.spaces.Box(
            low=0., high=1., shape=(6,), name={'stimulus': [0,1,2,3], 'cues': [4,5]})
        # self.cues_space = ngym.spaces.Box(
            # low=-1., high=1., shape=(2,), name={'cues': [0,1]})
        name = { 'choice': [0,1,2,3]}
        # name = { 'choice': [[0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1],]} #Turining this off because passing gt as index is the norm for neurogym.
        self.action_space = ngym.spaces.Discrete(1, name=name) # still self.action_space.shape returns shape () and that does not allow adding gt in R3
        self.high_pass_cue_mean = 0.7
        self.cue_variance = 0.2

    def _new_trial(self):
        # Setting time periods for this trial
        periods = ['cues', 'delay', 'stimulus', 'decision']
        # Will add stimulus and decision periods sequentially using self.timing info
        self.add_period(periods)
        
        if self.attend_to == 'either':
            modality = rng.choice(['audition', 'vision'])
        else:
            modality = self.attend_to
        # Sample observation for the next trial
        cues = np.zeros(shape=(self.total_cues) ) 
        if self.no_of_coherent_cues is None:
            if self.probabilistic:
                sampled_no_of_coherent_cues= get_difficulty(self.total_cues)
                # sampled_no_of_coherent_cues = rng.integers((self.total_cues//2)+1-2, self.total_cues, endpoint=True )
            else:
                sampled_no_of_coherent_cues = rng.integers((self.total_cues//2)+1, self.total_cues, endpoint=True )
        else:
            sampled_no_of_coherent_cues = self.no_of_coherent_cues
        # print(sampled_no_of_coherent_cues)
        if self.input_uncertainty: # sample from two gaussians with some overlap. 
            cues=self.cues_sampled_from_parent_hierarchical_class
        else:
            cues[:sampled_no_of_coherent_cues] = np.ones(sampled_no_of_coherent_cues)
        if self.use_oddballs:
            flip_probabilities = {5:0.5, 6:0.30, 7:0.22, 8:0.18, 9:0.12, 10:0.08}
            # flip_probabilities = {5:0.2, 6:0.1, 7:0.05, 8:0.01, 9:0.005, 10:0.0}
            flip_context_for_this_trial = rng.uniform() < (flip_probabilities[sampled_no_of_coherent_cues] * self.oddball_prob_scale)
            if flip_context_for_this_trial: cues = 1-cues # flip the cues to create an oddball with the opposite response. 
        #shuffle cues
        idx = rng.permutation(range(len(cues)))
        cues = (cues[idx])
        if modality == 'audition':
            cues = np.vstack([cues, 1-cues, ]) # assign dominant cues to audition
        else:
            cues = np.vstack([1-cues, cues, ]) # assign dominant cues to vision

        cues_ones = np.ones(shape=(2, int(2*cues.shape[1]))) 
        cues_ones[:,::2] = cues # interperse cues with wideband noise.
        # cues = cues_ones.reshape((2,20,1)).repeat(10, axis = 2).reshape(2, -1) # some kungfoo to make each cue last for 10ms. Not relaly sure this is necessary
        cues = cues_ones.T
        # plt.plot(cues, 'o')
        # plt.savefig('./cues.jpg')
        stimulus = np.zeros(4)
        bin_a, bin_v = np.random.binomial(size=2, n=1, p= 0.5)
        stimulus[:2] = np.array([bin_a, 1-bin_a]) 
        stimulus[2:] = np.array([bin_v, 1-bin_v]) 
        
        # Add value 1 to stimulus period at fixation location
        self.add_ob(stimulus, period='stimulus', where='stimulus')
        # Add cues to cues period at cues location
        self.add_ob(cues, period='cues', where='cues')
        
        # Set ground_truth
        audition = 1.* (modality == 'audition') 
        if (self.context == 2):
            audition = 1.- audition  # If context 2 flip the ground truth to the other context. 

        # groundtruth = np.concatenate([np.array(audition).reshape([1]), stimulus[:2].T if audition else stimulus[2:].T, ]) # choice is to report type of trial (audition (1) or not (0), and then the right choice of stimulus side  )
        groundtruth = np.zeros(4)
        if audition:
            groundtruth[:2] = stimulus[:2].T 
        else:
            groundtruth[2:] = stimulus[2:].T# choice is to report type of trial (audition (1) or not (0), and then the right choice of stimulus side  )
        # groundtruth = np.concatenate([np.array(audition).reshape([1]), stimulus[:2].T if audition else stimulus[2:].T, ]) # choice is to report type of trial (audition (1) or not (0), and then the right choice of stimulus side  )
        # set up ground truth vector as [audition (1), left, right]
        # gt_index = [ np.all(row == groundtruth) for row in [[0., 1., 0.], [0., 0., 1.], [1., 1., 0.], [1., 0., 1.]] ] #Turining this off because passing gt as index is the norm for neurogym.
        # gt_index =   np.argmax(groundtruth) #Turining this off because passing gt as index is the norm for neurogym.
        # self.set_groundtruth(np.argwhere(gt_index).squeeze(), period='decision', where='choice')  
        self.set_groundtruth( np.argmax(groundtruth), period='decision', where='choice')  
        # see note above. Self.action_space.shape still returns () which does not allow adding 3 dim for ground truth.

        trial = dict()
        trial['stimulus'] = stimulus
        trial['cues'] = cues
        trial['ground_truth'] = groundtruth
        trial['difficulty'] = sampled_no_of_coherent_cues/self.total_cues

        return trial
    
    def _step(self, action):
        # self.ob_now and self.gt_now correspond to
        # current step observation and groundtruth

        # If action equals to ground_truth, reward=1, otherwise 0
        reward = (action == self.gt_now) * 1.0
        
        done = False
        # By default, the trial is not ended
        info = {'new_trial': False}
        return self.ob_now, reward, done, info

class Shrew_task_hierarchical(TrialEnv):
    def __init__(self, dt=10, ):
        super().__init__(dt=dt)  # dt is passed to base task
        self.timing = {'cues': 200, 'delay': 50, 'stimulus': 30, 'decision': 20}
        self.total_cues = 10
        
        self.sampled_no_of_coherent_cues = rng.integers((self.total_cues//2)+1, self.total_cues )
        self.high_pass_cue_mean = 0.7
        self.cue_variance = 0.2

        self.input_uncertainty = False
        self.env_context1 = Shrew_task(dt =10, attend_to='either', context=1, )
        self.env_context2 = Shrew_task(dt =10, attend_to='either', context=2, )
        self.cues = []

        ### preparing these on the first run.
        cues = np.zeros(shape=(self.total_cues) ) 
        cues[:self.sampled_no_of_coherent_cues] = rng.normal(loc=self.high_pass_cue_mean, scale=self.cue_variance, size=self.sampled_no_of_coherent_cues) 
        cues[self.sampled_no_of_coherent_cues:] = rng.normal(loc=1-self.high_pass_cue_mean, scale=self.cue_variance, size=self.total_cues-self.sampled_no_of_coherent_cues) 
        cues=np.clip(cues, a_min=0., a_max=1.)
        self.env_context1.cues_sampled_from_parent_hierarchical_class = cues
        self.env_context2.cues_sampled_from_parent_hierarchical_class = cues
        self.cues.append(cues)
        ####
        
        self.history_of_contexts = []
        self.block_duration_low, self.block_duraion_high = 15, 25
        # self.switches = rng.integers(block_duration_low,block_duraion_high, 10000) # sample 100 block durations uniformly between low and high values.
        self.trials_remaining = 20 # initial value
        self.current_context = 0
        self.current_trial_context = self.current_context
        self.current_idx = 0
        self.switches = []
        self.context_ids = []
        self.difficulties = []
        self.use_oddballs = False
        self.oddballs = []
        self.probabilistic = False
        self.oddball_prob_scale = 0.8

        self.observation_space = ngym.spaces.Box(
            low=0., high=1., shape=(6,), name={'stimulus': [0,1,2,3], 'cues': [4,5]})
        name = { 'choice': [0,1,2,3]}
        self.action_space = ngym.spaces.Discrete(1, name=name) # still self.action_space.shape returns shape () and that does not allow adding gt in R3

    def update_context(self):
        # decrement trials remaining in block and switch to the next block. 
        self.context_ids.append(self.current_context)
        self.current_idx +=1 
        self.sampled_no_of_coherent_cues= get_difficulty(self.total_cues)
        # for each difficulty level it is associated with a probablity of being the other context
        # linear ramp? p(wrong) = 0.2, 0.1, 0.05, 0.01, 0.0
        if self.use_oddballs:
            flip_probabilities = {5:0.5, 6:0.30, 7:0.22, 8:0.18, 9:0.12, 10:0.08}
            # flip_probabilities = {5:0.2, 6:0.1, 7:0.05, 8:0.01, 9:0.005, 10:0.0}
            flip_context_for_this_trial = rng.uniform() < (flip_probabilities[self.sampled_no_of_coherent_cues] * self.oddball_prob_scale)
            self.current_trial_context = 1-self.current_context if flip_context_for_this_trial else self.current_context
            if flip_context_for_this_trial: self.oddballs.append(self.current_idx)
            # self.sampled_no_of_coherent_cues = rng.integers((self.total_cues//2)+1-2, self.total_cues, endpoint=True ) # endpoint makes the high limit inclusive
        else:
            self.sampled_no_of_coherent_cues = rng.integers((self.total_cues//2)+1, self.total_cues, endpoint=True )
            self.current_trial_context = self.current_context
        if self.input_uncertainty: # samples difficulty for all trials in the batch and gives them to children cxt classes. THat way all the batch is the same difficulty but permuted cues. 
            cues = np.zeros(shape=(self.total_cues) ) 
            cues[:self.sampled_no_of_coherent_cues] = rng.normal(loc=self.high_pass_cue_mean, scale=self.cue_variance, size=self.sampled_no_of_coherent_cues) 
            cues[self.sampled_no_of_coherent_cues:] = rng.normal(loc=1-self.high_pass_cue_mean, scale=self.cue_variance, size=self.total_cues-self.sampled_no_of_coherent_cues) 
            cues=np.clip(cues, a_min=0., a_max=1.)
            self.env_context1.cues_sampled_from_parent_hierarchical_class = cues
            self.env_context2.cues_sampled_from_parent_hierarchical_class = cues
            self.cues.append(cues)

        self.difficulties.append(self.sampled_no_of_coherent_cues)
        self.trials_remaining +=-1
        if self.trials_remaining == 0:
            self.trials_remaining = rng.integers(self.block_duration_low,self.block_duraion_high)
            self.current_context = 1-self.current_context
            self.switches.append(self.current_idx)

    def _new_trial(self):
        # Setting time periods for this trial
        periods = ['cues', 'delay', 'stimulus', 'decision']
        self.add_period(periods)

        if self.current_trial_context ==0:
            self.env_context1.no_of_coherent_cues = self.sampled_no_of_coherent_cues # control the level of difficulty so the whole batch has the same difficulty.
            trial = self.env_context1.new_trial()
        else:
            self.env_context2.no_of_coherent_cues = self.sampled_no_of_coherent_cues
            trial = self.env_context2.new_trial()
        stimulus = trial['stimulus']
        cues = trial['cues'] 
        groundtruth = trial['ground_truth']
            
        self.add_ob(stimulus, period='stimulus', where='stimulus')
        self.add_ob(cues, period='cues', where='cues')
        self.set_groundtruth( np.argmax(groundtruth), period='decision', where='choice')  

        trial['context'] = self.current_context
        
        return trial
    

# test = True
test = False
# test_changepoint = True
test_changepoint = False
if test:

    env = Shrew_task(attend_to='either', context=1,  no_of_coherent_cues=9)
    t = env.new_trial()
    ob_size = env.observation_space.shape[0]
    act_size = env.action_space.n
    print('ob_size :', ob_size, '   act_size: ', act_size)
    print(env.ob)
    print(env.gt)
    for i in range(20):
        t = env.new_trial()
        print(env.gt)
if test_changepoint:
    params = {
        'noisy_mean':       [0.05, 0, 0 , 0], 
        'drifting_mean':    [0.05, 0.05, 0 , 0],
        'oddball':          [0.05,  0.05, 0.1, 0],
        'changepoint':      [0.05, 0.0, 0.0 , 0.1]
    }
    param= params['oddball']
    env =  NoiseyMean(mean_noise= param[0], mean_drift = param[1], odd_balls_prob = param[2], change_point_prob = param[3], trials_remaining = 5)

    t = env.new_trial()
    ob_size = env.observation_space.shape[0]
    act_size = env.action_space.shape[0]
    print('ob_size :', ob_size, '   act_size: ', act_size)
    print(env.ob)
    print(env.gt)

    from time import perf_counter


    t1_start = perf_counter()
    for i in range(100):
        t = env.new_trial()
        # print(env.gt.shape)
    t2_start = perf_counter()
    t3_start = perf_counter()
    t4_start = perf_counter()
    print(f't1: {t2_start-t1_start} t2: {t3_start-t2_start} t3: {t4_start-t3_start}')
        #The main loop below takes 0.017 which for a batch of 100 is 1.7 seconds! Not sure how to shorten. Other than pre-generate all trials and save to desk.
