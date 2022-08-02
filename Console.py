# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3.6.9 64-bit
#     language: python
#     name: python3
# ---

# # Console Analysis
#
# ## Introduction
#
# This notebook has been created to enable easy monitoring of the training progress in the console. It integrates directly with the console to retrieve information and metrics, 
# so it is easy to reload to get updated charts, and more details than the current console UI provides.
#
# ### Usage
#
# Out of the box the file will only need the name of your model to load in the data.
#
# ### Contributions
#
# As usual, your ideas are very welcome and encouraged so if you have any suggestions either bring them to [the AWS DeepRacer Community](http://join.deepracing.io) or share as code contributions.
#
# ### Requirements
#
# Before you start using the notebook, you will need to install some dependencies. If you haven't yet done so, have a look at [The README.md file](/edit/README.md#running-the-notebooks) to find what you need to install.
# This workbook will require `deepracer-utils>=0.23`.
#
# Apart from the install, you also have to configure your programmatic access to AWS. Have a look at the guides below, AWS resources will lead you by the hand:
#
# AWS CLI: https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html
#
# Boto Configuration: https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html

# ## Core configuration

# Run the imports

# +
from deepracer.console import ConsoleHelper
from deepracer.logs import AnalysisUtils as au
from deepracer.logs.metrics import TrainingMetrics
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings

from IPython.display import display

from boto3.exceptions import PythonDeprecationWarning
warnings.filterwarnings("ignore", category=PythonDeprecationWarning)

# +
MODEL='Analysis-Demo'
ch = ConsoleHelper()
model_arn = ch.find_model(MODEL)
training_job = ch.get_training_job(model_arn)
training_job_status = training_job['ActivityJob']['Status']['JobStatus']
metrics_url = training_job['ActivityJob']['MetricsPreSignedUrl']

if model_arn is not None:
    print("Found model {} as {}".format(MODEL, model_arn))

if training_job_status is not None:
    print("Training status is {}".format(training_job_status))
# -

# ## Metrics Analysis

#
# ### Loading Metrics Data
#
# The basic setup covers loading in data from one single model. It obtains the metrics.json URL via API call and directly loads it into a TrainingMetrics object.

tm = TrainingMetrics(None, url=metrics_url)
rounds = np.array([[1,1]])
NUM_ROUNDS = len(rounds)

# ### Analysis

# The first analysis we will do is to display the basic statistics of the last 5 iterations.

summary=tm.getSummary(method='mean', summary_index=['r-i','master_iteration'])
display(summary[-5:])

# +
train=tm.getTraining()
ev=tm.getEvaluation()

print("Latest iteration: %s / master %i" % (max(train['r-i']),max(train['master_iteration'])))
print("Episodes: %i" % len(train))
# -

# ### Plotting progress
#
# The next command will display the desired progress chart. It shows the data per iteration (dots), and a rolling average to allow the user to easily spot a trend. 
#
# One can control the number of charts to show, based on which metric one wants to use `min`, `max`, `median` and `mean` are some of the available options.
#
# By altering the `rounds` parameter one can choose to not display all training rounds.

_ = tm.plotProgress(method=['median','mean','max'], rolling_average=5, figsize=(20,5), rounds=rounds[:,0])

# ### Best laps
#
# The following rounds will show the fastest 5 training and evaluation laps.

train_complete_lr = train[(train['round']>(NUM_ROUNDS-1)) & (train['complete']==1)]
display(train_complete_lr.nsmallest(5,['time']))

eval_complete_lr = ev[(ev['round']>(NUM_ROUNDS-1)) & (ev['complete']==1)]
display(eval_complete_lr.nsmallest(5,['time']))

# ### Best lap progression
#
# The below plot will show how the best laps for training and evaluation changes over time. This is useful to see if your model gets faster over time.

plt.figure(figsize=(15,5))
plt.title('Best lap progression')
plt.scatter(train_complete_lr['master_iteration'],train_complete_lr['time'],alpha=0.4)
plt.scatter(eval_complete_lr['master_iteration'],eval_complete_lr['time'],alpha=0.4)
plt.show()

# ### Lap progress
#
# The below shows the completion for each training and evaluation episode.

plt.figure(figsize=(15,5))
plt.title('Progress per episode')
train_r = train[train['round']==NUM_ROUNDS]
eval_r = ev[ev['round']==NUM_ROUNDS]
plt.scatter(train_r['episode'],train_r['completion'],alpha=0.5)
plt.scatter(eval_r['episode'],eval_r['completion'],c='orange',alpha=0.5)
plt.show()

# ## Robomaker Log Analysis
# If the training status is COMPLETED it is also possible to directly load the Robomaker logfile.
# Here we only provide a brief summary of the analysis. See `Training_analysis.ipynb` for more details.
#
# ### Load log file

# +
if training_job_status == "COMPLETED":
    df = ch.get_training_log_robomaker(model_arn)
else:
    df = pd.DataFrame()

simulation_agg = au.simulation_agg(df)
complete_ones = simulation_agg[simulation_agg['progress'] == 100]
# -

# ### Analysis

au.analyze_training_progress(simulation_agg, title='Training progress')

au.scatter_aggregates(simulation_agg, 'Stats for all laps')

# View five fastest complete laps
complete_ones.nsmallest(5, 'time')
