# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Metrics Analysis
#
# This notebook has been created to enable easy monitoring of the training progress by providing graphical charts similar to the chart shown in the DeepRacer Console. It integrates directly with S3, so it is easy to reload to get updated charts.
#
# ## Usage
#
# Out of the box the file will only need the bucket and prefix of your model to load in the data. If you run S3 locally (using minio) some additional parameters are required. Examples are listed below.
#
# Since this file can change in the future, I recommend that you make its copy and reorganize it to your liking. This way you will not lose your changes and you'll be able to add things as you please.
#
# ## Contributions
#
# As usual, your ideas are very welcome and encouraged so if you have any suggestions either bring them to [the AWS DeepRacer Community](http://join.deepracing.io) or share as code contributions.
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
#
# ## Installs and setups
#
# If you are using an AWS SageMaker Notebook or Sagemaker Studio Lab to run the log analysis, you will need to ensure you install required dependencies. To do that uncomment and run the following:

# +
# Make sure you have the required pre-reqs

# import sys

# # !{sys.executable} -m pip install --upgrade -r requirements.txt
# -

# ## Imports & Functions
#
# Run the imports and function blocks below:

# +
from deepracer.logs import metrics
import matplotlib.pyplot as plt
import numpy as np
import os
import boto3

NUM_ROUNDS=1 # default to define variable
# -
# ## Login
#
# Login to AWS. There are several ways to log in:
# 1. On EC2 instance or Sagemaker Notebook with correct IAM execution role assigned.
# 2. AWS credentials available in `.aws/` through using the `aws configure` command. (DeepRacer-for-Cloud's `dr-start-loganalysis` supports this)
# 3. Setting the relevant environment variables by uncommenting the below section.

# +
# os.environ["AWS_DEFAULT_REGION"] = "" #<-Add your region
# os.environ["AWS_ACCESS_KEY_ID"] = "" #<-Add your access key
# os.environ["AWS_SECRET_ACCESS_KEY"] = "" #<-Add you secret access key
# os.environ["AWS_SESSION_TOKEN"] = "" #<-Add your session key if you have one
# -

# ## Core configuration

# + tags=["parameters"]
# For basic setup set prefix to exact S3 location.  For advanced setup set prefix without the hypen and number at the end e.g. for test-1, test-2 set prefix as test 
PREFIX='model-name'   # Name of the model, without trailing '/'
BUCKET='bucket'       # Bucket name is default 'bucket' when training locally
PROFILE=None          # The credentials profile in .aws - 'minio' for local training
S3_ENDPOINT_URL=None  # Endpoint URL: None for AWS S3, 'http://minio:9000' for local training
# -

# ## Loading data
#
# ### Basic setup
#
# The basic setup covers loading in data from one single prefix, and one single worker. 

tm = metrics.TrainingMetrics(BUCKET, model_name=PREFIX, profile=PROFILE, s3_endpoint_url=S3_ENDPOINT_URL)

# +
# # Alternative from file
# tm = metrics.TrainingMetrics(None, fname='logs/sample-console-logs/metrics/training/training-20220611205309-EHNgTNY2T9-77qXhqjBi6A.json')
# -

# ### Advanced setup
#
# In the advanced setup we can load in multiple (cloned) training sessions in sequence to show progress end-to-end even if the training was stopped and restarted/cloned. Furthermore it is possible to load in training sessions with more than one worker (i.e. more than one Robomaker). 
#
# The recommended way to do this is to have a naming convention for the training sessions (e.g. MyModel-1, MyModel-2). The below loading lines require this.
#

# +
# rounds=np.array([[1,2],[2,2]])
# NUM_ROUNDS=rounds.shape[0]
# -

# Load in the models. You will be given a brief statistic of what has been loaded. To save on bandwidth, if already loaded once, only reload last round.

# +
# if 'tm' not in globals():
#     tm = metrics.TrainingMetrics(BUCKET, profile=PROFILE, s3_endpoint_url=S3_ENDPOINT_URL)
#
#    for r in rounds:
#        tm.addRound('{}-{}'.format(PREFIX, r[0]), training_round=r[0], workers=r[1])
# else:
#    tm.reloadRound('{}-{}'.format(PREFIX, rounds[-1][0]), training_round=rounds[-1][0], workers=rounds[-1][1])    
# -

# ## Analysis

# The first analysis we will do is to display the basic statistics of the last 5 iterations.

summary=tm.getSummary(method='mean', summary_index=['r-i','master_iteration'])
summary[-5:]

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

_ = tm.plotProgress(method=['median','mean','max'], rolling_average=5, figsize=(20,5))

_ = tm.plotProgress(method=['mean'],rolling_average=20,figsize=(20,5), 
                series=[('eval_completed','Evaluation','orange'),('train_completed','Training','blue')], 
                title="Percent completed laps ({}).", 
                ylabel="Percent", completedLapsOnly=False, grid=True)

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

_ = tm.plotProgress(method=['min','mean'],rolling_average=15, figsize=(20,5), 
                series=[('eval_time','Evaluation','orange'),('train_time','Training','blue')], 
                title="Laptime for completed laps ({}).", 
                ylabel="Laptime", completedLapsOnly=True, grid=True)

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


