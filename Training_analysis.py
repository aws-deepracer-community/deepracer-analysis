# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Training analysis for DeepRacer
#
# This notebook has been built based on the `DeepRacer Log Analysis.ipynb` provided by the AWS DeepRacer Team. It has been reorganised and expanded to provide new views on the training data without the helper code which was moved into utility `.py` files.
#
# ## Usage
#
# I have expanded this notebook from to present how I'm using this information. It contains descriptions that you may find not that needed after initial reading. Since this file can change in the future, I recommend that you make its copy and reorganize it to your liking. This way you will not lose your changes and you'll be able to add things as you please.
#
# **This notebook isn't complete.** What I find interesting in the logs may not be what you will find interesting and useful. I recommend you get familiar with the tools and try hacking around to get the insights that suit your needs.
#
# ## Contributions
#
# As usual, your ideas are very welcome and encouraged so if you have any suggestions either bring them to [the AWS DeepRacer Community](http://join.deepracing.io) or share as code contributions.
#
# ## Training environments
#
# Depending on whether you're running your training through the console or using the local setup, and on which setup for local training you're using, your experience will vary. As much as I would like everything to be taylored to your configuration, there may be some problems that you may face. If so, please get in touch through [the AWS DeepRacer Community](http://join.deepracing.io).
#
# ## Requirements
#
# Before you start using the notebook, you will need to install some dependencies. If you haven't yet done so, have a look at [The README.md file](/edit/README.md#running-the-notebooks) to find what you need to install.
#
# Apart from the install, you also have to configure your programmatic access to AWS. Have a look at the guides below, AWS resources will lead you by the hand:
#
# AWS CLI: https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html
#
# Boto Configuration: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html
#
# ## Credits
#
# I would like to thank [the AWS DeepRacer Community](http://join.deepracing.io) for all the feedback about the notebooks. If you'd like, follow [my blog](https://codelikeamother.uk) where I tend to write about my experiences with AWS DeepRacer.
#
# # Log Analysis
#
# Let's get to it.
#
# ## Imports
#
# Run the imports block below:

# +
import pandas as pd
import matplotlib.pyplot as plt

from deepracer.tracks import TrackIO, Track
from deepracer.tracks.track_utils import track_breakdown
from deepracer.logs import CloudWatchLogs as cw, \
    SimulationLogsIO as slio, \
    NewRewardUtils as nr, \
    AnalysisUtils as au, \
    PlottingUtils as pu, \
    ActionBreakdownUtils as abu

import extensions.sage as ex

# Ignore deprecation warnings we have no power over
import warnings
warnings.filterwarnings('ignore')
# -


# ## Load waypoints for the track you want to run analysis on
#
# The track waypoint files usually show up as new races start. Be sure to check for them in repository updates. You only need to load them in the block below.
#
# These files represent the coordinates of characteristic points of the track - the center line, inside border and outside border. Their main purpose is to visualise the track in images below. One thing that you may want to remember is that at the moment not all functions below work with all values of the coordinates. Especially some look awkward with bigger tracks or with negative coordinates. Usually there is an explanation on what to do to fix the view.
#
# The naming of the tracks is not super consistent. I'm also not sure all of them are available in the console or locally. You may want to know that:
# * London_Loop and Virtual_May19_Train_track - are the AWS DeepRacer Virtual League London Loop tracks
# * Tokyo - is the AWS DeepRacer Virtual League Kumo Torakku track
# * New_York - are the AWS DeepRacer Virtual League Empire City training and evaluation tracks
# * China - are the AWS Deepracer Virtual League Shanghai Sudu training and evaluation tracks
# * reinvent_base - is the re:Invent 2019 racing track
#
# There are also other tracks that you may want to explore. Each of them has its own properties that you might find useful for your model.
#
# Remeber that evaluation npy files are a community effort to visualise the tracks in the trainings, they aren't 100% accurate.
#
# Tracks Available:

# +
# !ls tracks/

tu = TrackIO()
# -

# Take the name from results above and paste below to load the key elements of the track and view the outline of it.

# +
track: Track = tu.load_track("reinvent_base")

track.road_poly
# -

# ## Get the logs
#
# Depending on which way you are training your model, you will need a different way to load the data.
#
# **AWS DeepRacer Console**
# The logs are being stored in CloudWatch, in group `/aws/robomaker/SimulationJobs`. You will be using boto3 to download them based on the training ID (stream name prefix). If you wish to bulk export the logs from Amazon Cloudwatch to Amazon S3 :: https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/S3ExportTasks.html
#
# **DeepRacer for Dummies/ARCC local training**
# Those two setups come with a container that runs Jupyter Notebook (as you noticed if you're using one of them and reading this text). Logs are stored in `/logs/` and you just need to point at the latest file to see the current training. The logs are split for long running training if they exceed 500 MB. The log loading method has been extended to support that.
#
# **Chris Rhodes' repo**
# Chris repo doesn't come with logs storage out of the box. I would normally run `docker logs dr > /path/to/logfile` and then load the file.
#
# Below I have prepared a section for each case. In each case you can analyse the logs as the training is being run, just in case of the Console you may need to force downloading of the logs as the `cw.download_log` method has a protection against needless downloads.
#
# Select your preferred way to get the logs below and you can get rid of the rest.

# +
# AWS DeepRacer Console
stream_name = 'sim-sample' ## CHANGE This to your simulation application ID
fname = 'logs/deepracer-%s.log' %stream_name  # The log will be downloaded into the specified path
cw.download_log(fname, stream_prefix=stream_name)  # add force=True if you downloaded the file before but want to repeat
fnameRobo = fname
fname = !ls logs/sagemaker*.log
fnameSage = fname[0]


# DeepRacer for Dummies / ARCC repository - comment the above and uncomment
# the lines below. They rely on a magic command to list log files
# ordered by time and pick up the most recent one (index zero).
# If you want an earlier file, change 0 to larger value.
# # !ls -t /workspace/venv/logs/*.log
# fname = !ls -t /workspace/venv/logs/*.log
# fname = fname[0]
# fnameRobo = fname
# fnameSage = ''


# Chris Rhodes' repository
# Use a preferred way of saving the logs to a file , then set an fname value to load it
# fname = /path/to/your/log/file
# fnameRobo = fname
# fnameSage = ''


# Mattc' repo
# fname1 = !ls -t logs/rl_coach*.log*
# fnameRobo = fname1[0]
# fname2 = !ls -t logs/sagemaker*.log*
# fnameSage = fname2[0]


print('robomaker log: {}'.format(fnameRobo))
print('sagemaker log: {}'.format(fnameSage))
# -

#
# ## Training Worker Stats
#
# For PPO we have:
#
# 1. Surrogate Loss
# 2. KL Divergence
# 3. Entropy
#
# These should all fluctuate but Surrogate Loss should tend toward zero.  Entropy will also tend towards zero. 
#

# +
df = ex.extract_training_epochs(fnameSage)
final_epochs = None if df is None else df.tail(1000)
print(0 if df is None else df.shape)

ex.plot_worker_stats(final_epochs)
# -

# ## Load the trace training log
#
# Now that the data is downloaded, we need to load it into memory. We will first read it from file and then convert to data frames in Pandas. [Pandas](https://pandas.pydata.org/) is a Python library for handling and analysing large amounts of data series. Remember this name, you may want to learn more about how to use it to get more information that you would like to get from the logs. Examples below are hardly scratching the surface.
#
# One important information to enter is the setting of your Episodes per iteration hyperparameter. This is used to group the episodes into iterations. This information is valuable when later looking at graphs showing how the training progresses per iteration. You can use it to detect which iteration gave you better outcomes and, if in local training, you could move to that iteration's outcome for submissions in the AWS DeepRacer League or  for continuing the training.
#
# The log files you have just gathered above have lines like this one:
# ```
# SIM_TRACE_LOG:799,111,1.7594,4.4353,3.0875,-0.26,2.50,2,1.0000,False,True,71.5802,49,17.67,1555554451.1110387
# ```
# This is all that matters for us. The first two are some tests I believe and when loading they get skipped, then each next line has the following fields:
# * episode number
# * step number
# * x coordinate
# * y coordinate
# * yaw of the car (where the car is heading)
# * decision about turning (turn value from your action space)
# * decision about throttle (speed value from your action space)
# * decision index (value from your action space)
# * reward value
# * is the lap complete
# * are all wheels on track?
# * progress in the lap
# * closest waypoint
# * track length
# * timestamp
#
# `la.load_data` and then `la.convert_to_pandas` read it and prepare for your usage. Sorting the values may not be needed, but I have experienced under some circumstances that the log lines were not ordered properly.

# +
EPISODES_PER_ITERATION = 20 #  Set to value of your hyperparameter in training

data = slio.load_data(fnameRobo)
df = slio.convert_to_pandas(data, episodes_per_iteration=EPISODES_PER_ITERATION)

df = df.sort_values(['episode', 'steps'])
# personally I think normalizing can mask too high rewards so I am commenting it out,
# but you might want it.
# slio.normalize_rewards(df)

#Uncomment the line of code below to evaluate a different reward function
#nr.new_reward(df, track.center_line, 'reward.reward_sample') #, verbose=True)
# -

# ## New reward
#
# Note the last line above: it takes a reward class from log-analysis/rewards, imports it, instantiates and recalculates reward values based on the data from the log. This lets you do some testing before you start training and rule out some obvious things.
#
# *If you find this confusing, don't worry, because it is confusing. You can safely ignore it for now and come back to it later.*
#
# This operation is possible because the logs contain all information needed to recreate the params for a given step. That said some could be implemented better and some were ignored for now and should be implemented.
#
# The sample reward mentioned in that line is located in `log-analysis/rewards/reward_sample.py` and looks like this:
#
# ```
# from time import time
#
#
# class Reward:
#     def __init__(self, verbose=False):
#         self.previous_steps = None
#         self.initial_time = None
#         self.verbose = verbose
#
#     @staticmethod
#     def get_time(params):
#         # remember: this will not return time before
#         # the first step has completed so the total
#         # time for lap will be lower by about 0.2s
#         return params.get('timestamp', None) or time()
#
#     def reward_function(self, params):
#         if self.previous_steps is None \
#                 or self.previous_steps > params['steps']:
#             # new lap!
#             self.initial_time = self.get_time(params)
#         else:
#             # we're continuing a lap
#             pass
#
#         steering_factor = 1.0
#
#         if abs(params['steering_angle']) > 14:
#             steering_factor = 0.7
#
#         reward = float(steering_factor)
#
#         self.previous_steps = params['steps']
#
#         if self.verbose:
#             print(params)
#
#         return reward
#
#
# reward_object = Reward()
#
#
# def reward_function(params):
#     return reward_object.reward_function(params)
#
# ```
#
# After some imports a class is declared, it's called `Reward`, then the class is instantiated and a function `reward_function` is declared. This somewhat bloated structure has a couple benefits:
# * It works in console/local training for actual training
# * It lets you reload the definition for class Reward and retry the reward function multiple times after changes without much effort
# * If you want to rely on state carried over between the steps, it's all contained in a reward object 
#
# The reward class hides two or three tricks for you:
# * `get_time` lets you abstract from machine time in log analysis - the supporting code adds one extra param, `timestamp`. That lets you get the right time value in new_reward function
# * the first condition allows detecting the beginning of an episode or even start of training you can use it for some extra operations between the episodes
# * `verbose` can be used to provide some noisier prints in the reward function - you can switch them on when loading the reward function above.
#
# Just remember: not all params are provided, you are free to implement them and raise a Pull Request for log_analysis.df_to_params method.
#
# If you just wrap your reward function like in the above example, you can use it in both log analysis notebook and the training.
#
# Final warning: there is a loss of precision in the logs (rounded numbers) and also potentially potential bugs. If you find any, please fix, please report.
#
# ## Graphs
#
# The original notebook has provided some great ideas on what could be visualised in the graphs. Below examples are a slightly extended version. Let's have a look at what they are presenting and what this may mean to your training.
#
# ### Training progress
#
# As you have possibly noticed by now, training episodes are grouped into iterations and this notebook also reflects it. What also marks it are checkpoints in the training. After each iteration a set of ckpt files is generated - they contain outcomes of the training, then a model.pb file is built based on that and the car begins a new iteration. Looking at the data grouped by iterations may lead you to a conclusion, that some earlier checkpoint would be a better start for a new training. While this is limited in the AWS DeepRacer Console, with enough disk space you can keep all the checkpoints along the way and use one of them as a start for new training (or even as a submission to a race).
#
# While the episodes in a given iteration are a mixture of decision process and random guesses, mean results per iteration may show a specific trend. Mean values are accompanied by standard deviation to show the concentration of values around the mean.
#
# #### Rewards per Iteration
#
# You can see these values as lines or dots per episode in the AWS DeepRacer console. When the reward goes up, this suggests that a car is learning and improving with regards to a given reward function. **This does not have to be a good thing.** If your reward function rewards something that harms performance, your car will learn to drive in a way that will make results worse.
#
# At first the rewards just grow if the progress achieved grows. Interesting things may happen slightly later in the training:
#
# * The reward may go flat at some level - it might mean that the car can't get any better. If you think you could still squeeze something better out of it, review the car's progress and consider updating the reward function, the action space, maybe hyperparameters, or perhaps starting over (either from scratch or from some previous checkpoint)
# * The reward may become wobbly - here you will see it as a mesh of dots zig-zagging. It can be a gradually growing zig-zag or a roughly stagnated one. This usually means the learning rate hyperparameter is too high and the car started doing actions that oscilate around some local extreme. You can lower the learning rate and hope to step closer to the extreme. Or run away from it if you don't like it
# * The reward plunges to near zero and stays roughly flat - I only had that when I messed up the hyperparameters or the reward function. Review recent changes and start training over or consider starting from scratch
#
# The Standard deviation says how close from each other the reward values per episode in a given iteration are. If your model becomes reasonably stable and worst performances become better, at some point the standard deviation may flat out or even decrease. That said, higher speeds usually mean there will be areas on track with higher risk of failure. This may bring the value of standard deviation to a higher value and regardless of whether you like it or not, you need to accept it as a part of fighting for significantly better times.
#
# #### Time per iteration
#
# I'm not sure how useful this graph is. I would worry if it looked very similar to the reward graph - this could suggest that slower laps will be getting higher rewards. But there is a better graph for spotting that below.
#
# #### Progress per Iteration
#
# This graph usually starts low and grows and at some point it will get flatter. The maximum value for progress is 100% so it cannot grow without limits. It usually shows similar initial behaviours to reward and time graphs. I usually look at it when I alter an action in training. In such cases this graph usually dips a bit and then returns or goes higher.
#
# #### Total reward per episode
#
# This graph has been taken from the orignal notebook and can show progress on certain groups of behaviours. It usually forms something like a triangle, sometimes you can see a clear line of progress that shows some new way has been first taught and then perfected.
#
# #### Mean completed lap times per iteration
#
# Once we have a model that completes laps reasonably often, we might want to know how fast the car gets around the track. This graph will show you that. I use it quite often when looking for a model to shave a couple more miliseconds. That said it has to go in pair with the last one:
#
# #### Completion rate per iteration
#
# It represents how big part of all episodes in an iteration is full laps. The value is from range [0, 1] and is a result of deviding amount of full laps in iteration by amount of all episodes in iteration. I say it has to go in pair with the previous one because you not only need a fast lapper, you also want a race completer.
#
# The higher the value, the more stable the model is on a given track.

# +
simulation_agg = au.simulation_agg(df)

au.analyze_training_progress(simulation_agg, title='Training progress')
# -
# ### Stats for all laps
#
# Previous graphs were mainly focused on the state of training with regards to training progress. This however will not give you a lot of information about how well your reward function is doing overall.
#
# In such case `scatter_aggregates` may come handy. It comes with three types of graphs:
# * progress/steps/reward depending on the time of an episode - of this I find reward/time and new_reward/time especially useful to see that I am rewarding good behaviours - I expect the reward to time scatter to look roughly triangular
# * histograms of time and progress - for all episodes the progress one is usually quite handy to get an idea of model's stability
# * progress/time_if_complete/reward to closest waypoint at start - these are really useful during training as they show potentially problematic spots on track. It can turn out that a car gets best reward (and performance) starting at a point that just cannot be reached if the car starts elsewhere, or that there is a section of a track that the car struggles to get past and perhaps it's caused by an aggressive action space or undesirable behaviour prior to that place
#
# Side note: `time_if_complete` is not very accurate and will almost always look better for episodes closer to 100% progress than in case of those 50% and below.


au.scatter_aggregates(simulation_agg, 'Stats for all laps')

# ### Stats for complete laps
# The graphs here are same as above, but now I am interested in other type of information:
# * does the reward scatter show higher rewards for lower completion times? If I give higher reward for a slower lap it might suggest that I am training the car to go slow
# * what does the time histogram look like? With enough samples available the histogram takes a normal distribution graph shape. The lower the mean value, the better the chance to complete a fast lap consistently. The longer the tails, the greater the chance of getting lucky in submissions
# * is the car completing laps around the place where the race lap starts? Or does it only succeed if it starts in a place different to the racing one?

# +
complete_ones = simulation_agg[simulation_agg['progress']==100]

if complete_ones.shape[0] > 0:
    au.scatter_aggregates(complete_ones, 'Stats for complete laps')
else:
    print('No complete laps yet.')
# -

# ### Categories analysis
# We're going back to comparing training results based on the training time, but in a different way. Instead of just scattering things in relation to iteration or episode number, this time we're grouping episodes based on a certaing information. For this we use function:
# ```
# analyze_categories(panda, category='quintile', groupcount=5, title=None)
# ```
# The idea is pretty simple - determine a way to cluster the data and provide that as the `category` parameter (alongside the count of groups available). In the default case we take advantage of the aggregated information to which quintile an episode belongs and thus build buckets each containing 20% of episodes which happened around the same time during the training. If your training lasted for five hours, this would show results grouped per each hour.
#
# A side note: if you run the function with `category='start_at'` and `groupcount=20` you will get results based on the waypoint closest to the starting point of an episode. If you need to, you can introduce other types of categories and reuse the function.
#
# The graphs are similar to what we've seen above. I especially like the progress one which shows where the model tends to struggle and whether it's successful laps rate is improving or beginning to decrease. Interestingly, I also had cases where I saw the completion drop on the progress rate only to improve in a later quintile, but with a better time graph.
#
# A second side note: if you run this function for `complete_ones` instead of `simulation_agg`, suddenly the time histogram becomes more interesting as you can see whether completion times improve.

au.scatter_by_groups(simulation_agg, title='Quintiles')

# ## Plot the action space coverage

ex.plot_action_space_coverage(df)

# ## Plot rewards per Iteration
#
# This graph is useful to understand the mean reward and standard deviation within each episode 

# +
REWARD_THRESHOLD = 80

#TIMESTAMP_COLUMN = 'timestamp' # For cloudwatch
#TIMESTAMP_COLUMN = 'tstamp' # For csv logs
#TIMESTAMP_COLUMN = 'simtime' # For local training either this or steps
TIMESTAMP_COLUMN = 'steps'
#df['simtime_from_steps'] = df['steps'] * 1/15
#TIMESTAMP_COLUMN = 'simtime_from_steps'

PACE_STANDARD_DEVIATIONS=4

ex.plot_rewards_per_iteration(df, REWARD_THRESHOLD, TIMESTAMP_COLUMN, EPISODES_PER_ITERATION, 
                              PACE_STANDARD_DEVIATIONS)
# -

# ## Data in tables
#
# While a lot can be seen in graphs that cannot be seen in the raw numbers, the numbers let us get into more detail. Below you will find a couple examples. If your model is behaving the way you would like it to, below tables may provide little added value, but if you struggle to improve your car's performance, they may come handy. In such cases I look for examples where high reward is giving to below-expected episode and when good episodes are given low reward.
#
# You can then take the episode number and scatter it below, and also look at reward given per step - this can in turn draw your attention to some rewarding anomalies and help you detect some unexpected outcomes in your reward function.
#
# There is a number of ways to select the data for display:
# * `nlargest`/`nsmallest` lets you display information based on a specific value being highest or lowest
# * filtering based on a field value, for instance `df[df['episode']==10]` will display only those steps in `df` which belong to episode 10
# * `head()` lets you peek into a dataframe
#
# There isn't a right set of tables to display here and the ones below may not suit your needs. Get to know Pandas more and have fun with them. It's almost as addictive as DeepRacer itself.
#
# The examples have a short comment next to them explaining what they are showing.

# View ten best rewarded episodes in the training
simulation_agg.nlargest(10, 'new_reward')

# View five fastest complete laps
complete_ones.nsmallest(5, 'time')

# View five best rewarded completed laps
complete_ones.nlargest(5, 'reward')

# View five best rewarded in completed laps (according to new_reward if you are using it)
complete_ones.nlargest(5, 'new_reward')

# View five most progressed episodes
simulation_agg.nlargest(5, 'progress')

# View information for a couple first episodes
simulation_agg.head()

# +
# Set maximum quantity of rows to view for a dataframe display - without that
# the view below will just hide some of the steps
pd.set_option('display.max_rows', 500)

# View all steps data for episode 10
df[df['episode']==10]
# -

# ## Analyze the reward distribution for your reward function

# This shows a histogram of actions per closest waypoint for episode 771.
# Will let you spot potentially problematic places in reward granting.
# In this example reward function is clearly `return 1`. It may be worrying
# if your reward function has some logic in it.
# If you have a final step reward that makes the rest of this histogram
# unreadable, you can filter the last step out by using
# `episode[:-1].plot.bar` instead of `episode.plot.bar`
episode = df[df['episode']==771]
episode.plot.bar(x='closest_waypoint', y='reward')

# ### Path taken for top reward iterations
#
# NOTE: at some point in the past in a single episode the car could go around multiple laps, the episode was terminated when car completed 1000 steps. Currently one episode has at most one lap. This explains why you can see multiple laps in an episode plotted below.
#
# Being able to plot the car's route in an episode can help you detect certain patterns in its behaviours and either promote them more or train away from them. While being able to watch the car go in the training gives some information, being able to reproduce it after the training is much more practical.
#
# Graphs below give you a chance to look deeper into your car's behaviour on track.
#
# We start with plot_selected_laps. The general idea of this block is as follows:
# * Select laps(episodes) that have the properties that you care about, for instance, fastest, most progressed, failing in a certain section of the track or not failing in there,
# * Provide the list of them in a dataframe into the plot_selected_laps, together with the whole training dataframe and the track info,
# * You've got the laps to analyse.

# +
# Some examples:
# highest reward for complete laps:
episodes_to_plot = complete_ones.nlargest(3,'reward')

# highest progress from all episodes:
# episodes_to_plot = simulation_agg.nlargest(3,'progress')

pu.plot_selected_laps(episodes_to_plot, df, track)
# -
# ### Plot a heatmap of rewards for current training. 
# The brighter the colour, the higher the reward granted in given coordinates.
# If instead of a similar view as in the example below you get a dark image with hardly any 
# dots, it might be that your rewards are highly disproportionate and possibly sparse.
#
# Disproportion means you may have one reward of 10.000 and the rest in range 0.01-1.
# In such cases the vast majority of dots will simply be very dark and the only bright dot
# might be in a place difficult to spot. I recommend you go back to the tables and show highest
# and average rewards per step to confirm if this is the case. Such disproportions may
# not affect your traning very negatively, but they will make the data less readable in this notebook.
#
# Sparse data means that the car gets a high reward for the best behaviour and very low reward
# for anything else, and worse even, reward is pretty much discrete (return 10 for narrow perfect,
# else return 0.1). The car relies on reward varying between behaviours to find gradients that can
# lead to improvement. If that is missing, the model will struggle to improve.


# +
#If you'd like some other colour criterion, you can add
#a value_field parameter and specify a different column

pu.plot_track(df, track)
# -

# ### Plot a particular iteration
# This is same as the heatmap above, but just for a single iteration.

# +
#If you'd like some other colour criterion, you can add
#a value_field parameter and specify a different column
iteration_id = 3

pu.plot_track(df[df['iteration'] == iteration_id], track)
# -

# ### Path taken in a particular episode

# +
episode_id = 122

pu.plot_selected_laps(simulation_agg[simulation_agg['episode'] == episode_id], df, track)
# -

# ### Path taken in a particular iteration

# +
iteration_id = 10

pu.plot_selected_laps([iteration_id], df, track, section_to_plot = 'iteration')
# -

# # Bulk training load
#
# This is some slow and heavy stuff. You can download all logs from CloudWatch (or part of them if you play with `not_older_than` and `older_than` parameters that take a string representation of a date in ISO format, for instance `DD-MM-YYYY` works).
#
# Since it can be a lot of downloading, it is commented out in here to avoid accidental runs.
#
# Files downloaded once will not be downloaded again unless you add `force=True`.

# +
#logs = cw.download_all_logs('logs/deepracer-', '/aws/robomaker/SimulationJobs')
# -

# Load every log from a folder. Every single one. This is a lot of data. If you want to save yourself some time later, below you have code to save and load all that with use of pickle.
#
# Alternatively, `logs` returned from `download_all_logs` is a list of tuples in which first element is a path to a downloaded log file (even if it already exists, but would've been donwloaded if `force=True`), so you can use that to load logs in bulk.

# +
import os

base_folder = 'logs'
df_list = list()
big_training_panda = None
for stream in os.listdir(base_folder):
    data = slio.load_data('%s/%s' % (base_folder, stream))
    df = slio.convert_to_pandas(data)
    df['stream'] = stream[10:]
    if big_training_panda is not None:
        big_training_panda = big_training_panda.append(df)
    else:
        big_training_panda = df
# -

# Have I mentioned a lot of data? This stores the data preprocessed for time savings
big_training_panda.to_pickle('bulk_training_set.pickle')

# +
from pandas import read_pickle

big_training_panda = read_pickle('bulk_training_set.pickle')

# +
# as usual, handle with care. Towards the end of the May race I needed 30-45 minutes to recalculate the reward.
#nr.new_reward(big_training_panda, track.center_line, 'reward.reward_sample') #, verbose=True)

# +
# Below code is using stream name as part of grouping since otherwise there would be episode number collisions
big_simulation_agg = au.simulation_agg(big_training_panda, 'stream')

big_complete_ones = big_simulation_agg[big_simulation_agg['progress']==100]

# +
grouped = big_simulation_agg.groupby(['stream'])

for name, group in grouped:
    au.scatter_aggregates(group, title=name)
# -

# By the end of London Loop I had so much noise and random tries that wanted to find the most promising version of my model to submit. I used the below piece of code to iterate through all the stream values to detect the one with most promising times histogram. I should've added progress as well since the fastest ones hardly ever completed a lap. I will leave adding that as an exercise for the reader.

values = []
show = []
show_above = -1
i = 0
for value in big_complete_ones.stream.values:
    if value in values:
        continue
    values.append(value)
    if i in show or i > show_above:
        print(value)
        big_complete_ones[big_complete_ones['stream']==value].hist(column=['time'], bins=20)
    i += 1

# display loads of everything
big_simulation_agg

# # Action breakdown per iteration and historgram for action distribution for each of the turns - reinvent track
#
# This plot is useful to understand the actions that the model takes for any given iteration. Unfortunately at this time it is not fit for purpose as it assumes six actions in the action space and has other issues. It will require some work to get it to done but the information it returns will be very valuable.
#
# This is a bit of an attempt to abstract away from the brilliant function in the original notebook towards a more general graph that we could use. It should be treated as a work in progress. The track_breakdown could be used as a starting point for a general track information object to handle all the customisations needed in methods of this notebook.
#
# A breakdown track data needs to be available for it. If you cannot find it for the desired track, MAKEIT.
#
# Currently supported tracks:

track_breakdown.keys()

# The second parameter is either a single index or a list of indices for df iterations that you would like to view. You can for instance use `sorted_idx` list which is a sorted list of iterations from the highest to lowest reward.
#
# Bear in mind that you will have to provide a proper action naming in parameter `action_names`, this function assumes only six actions by default. I think they need to match numbering of actions in your model's metadata json file.

abu.action_breakdown(df, 20, track, track_breakdown['reinvent2018'])
