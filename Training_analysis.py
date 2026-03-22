# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: .venv (3.12.10)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Training analysis for DeepRacer
#
# This notebook has been built based on the `DeepRacer Log Analysis.ipynb` provided by the AWS DeepRacer Team. It has been reorganised and expanded to provide new views on the training data without the helper code which was moved into the [`deepracer-utils` library](https://github.com/aws-deepracer-community/deepracer-utils).

# %% [markdown]
# ## Introduction
#

# %% [markdown]
# ### Training environments
#
# Depending on whether you're running your training through the console or using the local setup, and on which setup for local training you're using, your experience will vary. While every effort has been made to support various configurations, there may still be some problems you face. If so, please get in touch through [the AWS DeepRacer Community](http://join.deepracing.io).
#
# ### Requirements
#
# Before you start using the notebook, you will need to install some dependencies. If you haven't yet done so, have a look at [The README.md file](/edit/README.md#running-the-notebooks) to find what you need to install.
#
# Apart from the install, you also have to configure your programmatic access to AWS. Have a look at the guides below, AWS resources will lead you by the hand:
# * AWS CLI: https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html
# * Boto Configuration: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html
#
# ### Credits
#
# * AWS DeepRacer Team for initial workbooks created for DeepRacer Workshops at Summits and re:Invent.
# * [CodeLikeAMother](https://codelikeamother.uk) for initial rework of the notebook.
# * [The AWS DeepRacer Community](http://join.deepracing.io) for feedback and incremental improvements.

# %% [markdown]
# ## Prepare

# %% [markdown]
# ### Prerequisites
#
# If you are using an AWS SageMaker Notebook or SageMaker Studio Lab to run the log analysis, you will need to ensure you install required dependencies. To do that uncomment and run the following:

# %%
# import sys
# # !{sys.executable} -m pip install --upgrade -r requirements.txt

# %% [markdown]
# ### Imports
#
# Run the imports block below:

# %%
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from pprint import pprint
import os
from enum import Enum

from deepracer.tracks import TrackIO, Track
from deepracer.tracks.track_utils import track_breakdown, track_meta
from deepracer.logs import \
    SimulationLogsIO as slio, \
    NewRewardUtils as nr, \
    AnalysisUtils as au, \
    PlottingUtils as pu, \
    ActionBreakdownUtils as abu, \
    DeepRacerLog, \
    TarFileHandler, S3FileHandler, FSFileHandler, \
    SimtraceStabilityAnalyzer

# Ignore deprecation warnings we have no power over
import warnings
warnings.filterwarnings('ignore')

class MODE(Enum):
    FS = "FS"
    S3 = "S3"
    TAR = "TAR"


# %% [markdown]
# ### Login
#
# Login to AWS. There are several ways to log in:
# 1. On EC2 instance or Sagemaker Notebook with correct IAM execution role assigned.
# 2. AWS credentials available in `.aws/` through using the `aws configure` command. (DeepRacer-for-Cloud's `dr-start-loganalysis` supports this)
# 3. Setting the relevant environment variables by uncommenting the below section.

# %%
# os.environ["AWS_DEFAULT_REGION"] = "" #<-Add your region
# os.environ["AWS_ACCESS_KEY_ID"] = "" #<-Add your access key
# os.environ["AWS_SECRET_ACCESS_KEY"] = "" #<-Add you secret access key
# os.environ["AWS_SESSION_TOKEN"] = "" #<-Add your session key if you have one

# %% [markdown]
# ## Get the logs
#
# Depending on which way you are training your model, you will need a slightly different way to load the data. The simplest way to read in training data is using the sim-trace files, but the current workbook supports reading in a set of different formats, including the `tar.gz` files from DeepRacer on AWS. For other ways to read in data look at the [configuration examples](https://github.com/aws-deepracer-community/deepracer-utils/blob/master/docs/examples.md).

# %% [markdown]
# ### Load the files
#
# Populate the `my_mode` variable to select what kind of log file you want to process.

# %% tags=["parameters"]
my_mode = MODE.FS # Change this to switch between FS, S3 and TAR modes

# %% [markdown]
# #### Folder on disk
#
# The `FSFileHandler` will load in a model folder extracted onto your local disk.

# %%
if my_mode == MODE.FS:
    fh = FSFileHandler(model_folder=os.path.join('logs', 'sample-console-logs'))
    log = DeepRacerLog(filehandler=fh, verbose=True)
    log.load_training_trace()

# %% [markdown]
# #### S3 Bucket with Prefix
#
# The `S3FileHandler` will load in a model folder that is remote in an S3 Bucket.

# %% tags=["parameters"]
PREFIX='model-name'   # Name of the model, without trailing '/'
BUCKET='bucket'       # Bucket name is default 'bucket' when training locally
PROFILE=None          # The credentials profile in .aws - 'minio' for local training
S3_ENDPOINT_URL=None  # Endpoint URL: None for AWS S3, 'http://minio:9000' for local training

# %%
if my_mode == MODE.S3:
    fh = S3FileHandler(bucket=BUCKET, prefix=PREFIX, profile=PROFILE, s3_endpoint_url=S3_ENDPOINT_URL)
    log = DeepRacerLog(filehandler=fh, verbose=True)
    log.load_training_trace()

# %% [markdown]
# #### DeepRacer on AWS zipped archive
#
# The `TarFileHandler` will unzip and process the files inside a log file provided by DeepRacer on AWS.

# %% tags=["parameters"]
ARCHIVE_PATH='logs\deepracerindy-training-kJuf1ySmGZlhSMI-logs.tar.gz'

# %%
if my_mode == MODE.TAR:
    fh = TarFileHandler(archive_path=ARCHIVE_PATH)
    log = DeepRacerLog(filehandler=fh, verbose=True)
    log.load_training_trace()

# %% [markdown]
# ### Validate files
#
# The following cell checks whether the files loaded correctly. For file-system or S3 files, metadata will also be loaded, and you will see details printed below: the agent and network configuration, the hyperparameters, and the action space. Files loaded from DeepRacer on AWS will not provide this.

# %%
try:
    pprint(log.agent_and_network())
    print("-------------")
    pprint(log.hyperparameters())
    print("-------------")
    pprint(log.action_space())
except Exception:
    print("Metadata not available")

# %% [markdown]
# Now let's see what got loaded into the dataframe - the data structure holding your simulation information. The `head()` method prints out the first few lines of the data:

# %%
df = log.dataframe()
df.head()

# %% [markdown]
# ### Stability
# When loading in the traces we can also analyze the stability of the simulator during the training. The DeepRacer simulator should run at 15 fps; meaning that each step should be on average 66.6ms apart. If the average is >70ms, and/or the 95th percentile is >90ms, this means that there were performance problems with the simulator, which again could mean that the training was not as good as it could have been.

# %%
df_stats = log.stability.print_summary()

# %% [markdown]
# ## Load track waypoints
#
# The track waypoint files represent the coordinates of characteristic points of the track - the center line, inside border and outside border. Their main purpose is to visualise the track in images below.
#
# The naming of the tracks is not super consistent. The ones that we already know have been mapped to their official names in the track_meta dictionary.
#
# Tracks Available:

# %%
tu = TrackIO()
tracks_df = pd.DataFrame(
    [{"filename": t, "name": track_meta.get(t[:-4], "Unknown")}
     for t in tu.get_tracks()],
    columns=["filename", "name"]
)

@widgets.interact(filter=widgets.Text(placeholder="Filter by filename or name..."))
def show_tracks(filter=""):
    mask = (tracks_df["filename"].str.contains(filter, case=False) |
            tracks_df["name"].str.contains(filter, case=False))
    result = tracks_df[mask].reset_index(drop=True)
    display(result.head(10).style.hide())



# %% [markdown]
# Now let's load the track waypoints and visualize them.

# %%
try:
    track_name = log.agent_and_network()["world"]
except Exception as e:
    track_name = "reinvent_base"


track: Track = tu.load_track(track_name)
pu.plot_trackpoints(track)

# %% [markdown]
# ## Analyze the Training Data
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
# * The reward may become wobbly - here you will see it as a mesh of dots zig-zagging. It can be a gradually growing zig-zag or a roughly stagnated one. This usually means the learning rate hyperparameter is too high and the car started doing actions that oscillate around some local extreme. You can lower the learning rate and hope to step closer to the extreme, or abandon it in favour of a different starting point
# * The reward plunges to near zero and stays roughly flat - this typically occurs when the hyperparameters or reward function contain an error. Review recent changes and start training over or consider starting from scratch
#
# The Standard deviation says how close from each other the reward values per episode in a given iteration are. If your model becomes reasonably stable and worst performances become better, at some point the standard deviation may flat out or even decrease. That said, higher speeds usually mean there will be areas on track with higher risk of failure. This may bring the value of standard deviation to a higher value and regardless of whether you like it or not, you need to accept it as a part of fighting for significantly better times.
#
# #### Time per iteration
#
# The usefulness of this graph is limited. It is worth watching if it looks very similar to the reward graph - this could suggest that slower laps are getting higher rewards. There is a better graph for spotting that below.
#
# #### Progress per Iteration
#
# This graph usually starts low and grows and at some point it will get flatter. The maximum value for progress is 100% so it cannot grow without limits. It usually shows similar initial behaviours to reward and time graphs. It is worth checking this graph when altering an action during training. In such cases this graph usually dips a bit and then returns or goes higher.
#
# #### Total reward per episode
#
# This graph has been taken from the original notebook and can show progress on certain groups of behaviours. It usually forms something like a triangle, sometimes you can see a clear line of progress that shows some new way has been first taught and then perfected.
#
# #### Mean completed lap times per iteration
#
# Once we have a model that completes laps reasonably often, we might want to know how fast the car gets around the track. This graph will show you that. It is especially useful when looking for a model to shave a couple more milliseconds. That said it has to go in pair with the last one:
#
# #### Completion rate per iteration
#
# It represents what fraction of all episodes in an iteration are complete laps. The value is in the range [0, 1], calculated by dividing the number of complete laps in an iteration by the total number of episodes in that iteration. It should be read alongside the previous graph, because a fast lap time is only meaningful if the car also completes laps reliably.
#
# The higher the value, the more stable the model is on a given track.

# %%
simulation_agg = au.simulation_agg(df)
try: 
    if df.nunique(axis=0)['worker'] > 1:
        print("Multiple workers have been detected, reloading data with grouping by unique_episode")
        simulation_agg = au.simulation_agg(df, secondgroup="unique_episode")
except:
    print("Multiple workers not detected, assuming 1 worker")

au.analyze_training_progress(simulation_agg, title='Training progress')
# %% [markdown]
# ### Stats for all laps
#
# Previous graphs were mainly focused on the state of training with regards to training progress. This however will not give you a lot of information about how well your reward function is doing overall.
#
# In such cases `scatter_aggregates` may come in handy. It provides three types of graphs:
# * progress/steps/reward depending on the time of an episode - reward/time and new_reward/time are especially useful to confirm that good behaviours are being rewarded - the reward to time scatter should look roughly triangular
# * histograms of time and progress - for all episodes the progress one is usually quite handy to get an idea of model's stability
# * progress/time_if_complete/reward to closest waypoint at start - these are really useful during training as they show potentially problematic spots on track. It can turn out that a car gets best reward (and performance) starting at a point that just cannot be reached if the car starts elsewhere, or that there is a section of a track that the car struggles to get past and perhaps it's caused by an aggressive action space or undesirable behaviour prior to that place
#
# Side note: `time_if_complete` is not very accurate and will almost always look better for episodes closer to 100% progress than in case of those 50% and below.


# %%
au.scatter_aggregates(simulation_agg, 'Stats for all laps')

# %% [markdown]
# ### Stats for complete laps
# The graphs here are the same as above, but the focus is on a different type of information:
# * does the reward scatter show higher rewards for lower completion times? Granting a higher reward for a slower lap suggests the model is being trained to go slow
# * what does the time histogram look like? With enough samples available the histogram takes a normal distribution graph shape. The lower the mean value, the better the chance to complete a fast lap consistently. The longer the tails, the greater the chance of getting lucky in submissions
# * is the car completing laps around the place where the race lap starts? Or does it only succeed if it starts in a place different to the racing one?

# %%
complete_ones = simulation_agg[simulation_agg['progress']==100]

if complete_ones.shape[0] > 0:
    au.scatter_aggregates(complete_ones, 'Stats for complete laps')
else:
    print('No complete laps yet.')

# %% [markdown]
# ### Categories analysis
# We're going back to comparing training results based on training time, but in a different way. Instead of scattering values against iteration or episode number, this time episodes are grouped by position within the training. For this we use the function:
# ```
# scatter_by_groups(panda, groupcount=5, title=None)
# ```
# The idea is straightforward - episodes are divided into equally sized buckets (quintiles by default), each containing 20% of all episodes ordered by time. If your training lasted five hours, this would show results grouped roughly per hour.
#
# A side note: if you run the function with `category='start_at'` and `groupcount=20` you will get results based on the waypoint closest to the starting point of an episode. If you need to, you can introduce other types of categories and reuse the function.
#
# The graphs are similar to what we've seen above. The progress graph is particularly revealing - it shows where the model tends to struggle and whether its lap completion rate is improving or beginning to decrease. Interestingly, there are cases where the completion rate drops on the progress graph only to improve in a later quintile, accompanied by a better time graph.
#
# A second side note: if you run this function for `complete_ones` instead of `simulation_agg`, suddenly the time histogram becomes more interesting as you can see whether completion times improve.

# %%
au.scatter_by_groups(simulation_agg, title='Quintiles')

# %% [markdown]
# ## Data in tables
#
# While a lot can be seen in graphs that cannot be seen in the raw numbers, the numbers let us get into more detail. Below you will find a couple examples. If your model is behaving the way you would like it to, below tables may provide little added value, but if you struggle to improve your car's performance, they may come in handy. Look for examples where high reward is given to a below-expected episode and when good episodes are given low reward.
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

# %%
# View ten best rewarded episodes in the training
simulation_agg.nlargest(10, 'new_reward')

# %%
# View five fastest complete laps
complete_ones.nsmallest(5, 'time')

# %%
# View five best rewarded completed laps
complete_ones.nlargest(5, 'reward')

# %%
# View five best rewarded in completed laps (according to new_reward if you are using it)
complete_ones.nlargest(5, 'new_reward')

# %%
# View five most progressed episodes
simulation_agg.nlargest(5, 'progress')

# %%
# View information for a couple first episodes
simulation_agg.head()

# %%
# Set maximum quantity of rows to view for a dataframe display - without that
# the view below will just hide some of the steps
pd.set_option('display.max_rows', 500)

# View steps data for episode 10. Remove .head() for all steps.
df[df['episode']==10].head()

# %% [markdown]
# ## Analyze the reward distribution

# %% [markdown]
# This shows a bar chart of reward per closest waypoint for the selected episode (episode 9 by default).
# It will let you spot potentially problematic places in reward granting.
# In this example, the reward function is clearly `return 1`. It may be worrying
# if your reward function has some logic in it.
# If you have a final step reward that makes the rest of this histogram
# unreadable, you can filter the last step out by using
# `episode[:-1].plot.bar` instead of `episode.plot.bar`
#

# %%
episode = df[df['episode']==9]

if episode.empty:
    print("You probably don't have episode with this number, try a lower one.")
else:
    episode.plot.bar(x='closest_waypoint', y='reward')

# %% [markdown]
# ## Path Analysis
#
# NOTE: in earlier versions of the simulator, a single episode could span multiple laps, terminating only after 1000 steps. Currently, each episode covers at most one lap. If you are analysing logs from the older simulator, plots of individual episodes may therefore show more than one lap.
#
# Being able to plot the car's route in an episode can help you detect certain patterns in its behaviours and either promote them more or train away from them. While being able to watch the car go in the training gives some information, being able to reproduce it after the training is much more practical.
#
# Graphs below give you a chance to look deeper into your car's behaviour on track.
#
# We start with plot_selected_laps. The general idea of this block is as follows:
# * Select laps(episodes) that have the properties that you care about, for instance, fastest, most progressed, failing in a certain section of the track or not failing in there,
# * Provide the list of them in a dataframe into the plot_selected_laps, together with the whole training dataframe and the track info,
# * You've got the laps to analyse.

# %%
# Some examples:
# highest reward for complete laps:
# episodes_to_plot = complete_ones.nlargest(3,'reward')

# highest progress from all episodes:
episodes_to_plot = simulation_agg.nlargest(3,'progress')

try:
    if df.nunique(axis=0)['worker'] > 1:
        pu.plot_selected_laps(episodes_to_plot, df, track, section_to_plot="unique_episode")
    else:
        pu.plot_selected_laps(episodes_to_plot, df, track)
except:
    print("Multiple workers not detected, assuming 1 worker")
    pu.plot_selected_laps(episodes_to_plot, df, track, single_plot=True)
# %% [markdown]
# ### Plot a heatmap of rewards for current training. 
# The brighter the colour, the higher the reward granted in given coordinates.
# If instead of a similar view as in the example below you get a dark image with hardly any 
# dots, it might be that your rewards are highly disproportionate and possibly sparse.
#
# Disproportion means you may have one reward of 10.000 and the rest in range 0.01-1.
# In such cases the vast majority of dots will simply be very dark and the only bright dot
# might be in a place difficult to spot. It is worth going back to the tables to show the highest
# and average rewards per step to confirm if this is the case. Such disproportions may
# not affect your training very negatively, but they will make the data less readable in this notebook.
#
# Sparse data means that the car gets a high reward for the best behaviour and very low reward
# for anything else, and worse even, reward is pretty much discrete (return 10 for narrow perfect,
# else return 0.1). The car relies on reward varying between behaviours to find gradients that can
# lead to improvement. If that is missing, the model will struggle to improve.


# %%
#If you'd like some other colour criterion, you can add
#a value_field parameter and specify a different column

pu.plot_track(df, track)

# %% [markdown]
# ### Plot a particular iteration
# This is the same as the heatmap above, but just for a single iteration.

# %%
#If you'd like some other colour criterion, you can add
#a value_field parameter and specify a different column
iteration_id = 3

pu.plot_track(df[df['iteration'] == iteration_id], track)

# %% [markdown]
# ### Path taken in a particular episode

# %%
episode_id = 12

# %%
try:
    if df.nunique(axis=0)['worker'] > 1:
        pu.plot_selected_laps([episode_id], df, track, section_to_plot="unique_episode")
    else: 
        pu.plot_selected_laps([episode_id], df, track)
except:
    print("Multiple workers not detected, assuming 1 worker")
    pu.plot_selected_laps([episode_id], df, track)

# %% [markdown]
# ### Path taken in a particular iteration

# %%
iteration_id = 10

pu.plot_selected_laps([iteration_id], df, track, section_to_plot = 'iteration')

# %% [markdown]
# ## Action breakdown per turn - reinvent track
#
# This plot is useful to understand the actions that the model takes for any given iteration. Unfortunately at this time it is not fit for purpose as it assumes six actions in the action space and has other issues. It will require some work to get it done but the information it returns will be very valuable.
#
# This is an attempt to generalise the function from the original notebook into a reusable graph that works for any action space. It should be treated as a work in progress. The track_breakdown could be used as a starting point for a general track information object to handle all the customisations needed in methods of this notebook.
#
# Track breakdown data needs to be available for it. If you cannot find it for the desired track, create it.
#
# Currently supported tracks:

# %%
track_breakdown.keys()

# %% [markdown]
# You can replace episode_ids with iteration_ids and make a breakdown for a whole iteration.
#
# **Note: does not work for continuous action space (yet).** 

# %%
abu.action_breakdown(df, track, track_breakdown=track_breakdown.get('reinvent2018'), episode_ids=[12])
