"""
Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

from datetime import datetime
from decimal import *

import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from shapely.geometry.polygon import LineString
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns


# aph: ported from Chris Thompson repo    
# Get the data for policy training based on experiences received from rollout workers
def extract_training_epochs(fname):
  if not fname:
    return None

  df_list = list()
  tm = datetime.timestamp(datetime.now())
  with open(fname, 'r') as f:
    for line in f.readlines():
      if 'Policy training' in line:
        # Policy training> Surrogate loss=-0.08960115909576416, KL divergence=0.031318552792072296, Entropy=3.070063352584839, training epoch=3, learning_rate=0.0003\r"
        parts = line.split("Policy training> ")[1].strip().split(',')
        surrogate_loss = float(parts[0].split('Surrogate loss=')[1])
        kl_divergence = float(parts[1].split('KL divergence=')[1])
        entropy = float(parts[2].split('Entropy=')[1])
        epoch = float(parts[3].split('training epoch=')[1])
        learning_rate = float(parts[4].split('learning_rate=')[1])

        # "log_time":"2019-09-19T20:14:31+00:00"
        df_list.append((tm, surrogate_loss, kl_divergence, entropy, epoch, learning_rate))
        tm += 1

  columns = ['timestamp', 'surrogate_loss', 'kl_divergence', 'entropy', 'epoch', 'learning_rate']
  return pd.DataFrame(df_list, columns=columns).sort_values('timestamp',axis='index').reset_index(drop=True)

# Get the experience iteration summary upon receipt from rollout workers
def extract_training_iterations(fname):
  df_list = list()
  tm = datetime.timestamp(datetime.now())
  with open(fname, 'r') as f:
    for line in f.readlines():
      if 'Training' in line:
        # Training> Name=main_level/agent, Worker=0, Episode=781, Total reward=67.4, Steps=20564, Training iteration=39\r","container_id":"8963076f3fa49d5c0aac3129ba277675445e45819daf31505b41647ca2c2ec84","container_name":"/dr-training","log_time":"2019-09-20T06:02:54+00:00"}
        parts = line.split("Training> ")[1].strip().split(',')
        name = parts[0].split('Name=')[1]
        worker = int(parts[1].split('Worker=')[1])
        episode = int(parts[2].split('Episode=')[1])
        reward = float(parts[3].split('Total reward=')[1])
        steps = int(parts[4].split('Steps=')[1])
        iteration = int(parts[5].split('Training iteration=')[1])

        # "log_time":"2019-09-19T20:14:31+00:00"
        df_list.append((tm, name, worker, episode, reward, steps, iteration))
        tm += 1

  columns = ['timestamp', 'name', 'worker', 'episode', 'reward', 'steps','iteration']
  return pd.DataFrame(df_list, columns=columns).sort_values('timestamp',axis='index').reset_index(drop=True)

def plot_worker_stats(fe):
  if fe is None:
    return

  fig = plt.figure(figsize=(16, 18))
  ax = fig.add_subplot(311)
  ax.plot(fe['timestamp'],fe['surrogate_loss'])
  ax.set_title('Surrogate Loss')
  ax = fig.add_subplot(312)
  ax.plot(fe['timestamp'],fe['kl_divergence'])
  ax.set_title('KL Divergence')
  ax = fig.add_subplot(313)
  ax.plot(fe['timestamp'],fe['entropy'])
  ax.set_title('Entropy')  

def plot_action_space_coverage(df):
  # Reject initial actions where there is no action index
  #df_not_empty = df[df['throttle'] != 0.0]
  df['degrees'] = df['steer'].transform(lambda x: math.degrees(x))
  unique_actions = df.groupby(['throttle','degrees']).size().reset_index(name='count')
  #print(df.groupby(['throttle','degrees'])

  # Plot how often model used the action choices
  fig = plt.figure(figsize=(10,12))
  ax = fig.add_subplot(211)
  sns.scatterplot(x='degrees', 
                y='throttle', 
                size='count', 
                sizes=(2,1000),
                hue='count', 
                data=unique_actions, 
                ax=ax)
  ax.set_title('Action Counts')
  ax.set_xlim(df['degrees'].max()+5,df['degrees'].min()-5);
  #pd.DataFrame(unique_actions)

  # Plot how much reward was given for action choices
  action_rewards = df.groupby(['throttle','degrees']).reward.describe().reset_index().rename(columns={'mean': 'mean reward', 'max': 'max reward'})

  ax = fig.add_subplot(212)
  sns.scatterplot(x='degrees', 
                y='throttle', 
                size='mean reward',
                sizes=(10,2000), 
                size_norm=(0,action_rewards['max reward'].max()),
                hue='mean reward',
                hue_norm=(0,action_rewards['max reward'].max()),
                legend='brief',
                data=action_rewards, 
                ax=ax)
  sns.scatterplot(x='degrees', 
                y='throttle', 
                size='max reward',
                sizes=(10,2000),
                hue='max reward',
                alpha=0.2,
                legend='brief',
                data=action_rewards, 
                ax=ax)
  ax.set_title('Mean / Max Reward Values Per Action')
  ax.set_xlim(df['degrees'].max()+5,df['degrees'].min()-5);
  return pd.DataFrame(action_rewards)

def plot_rewards_per_iteration(df, rt, tc, ei, psd):
  # Normalize the rewards to a 0-1 scale

  from sklearn.preprocessing import  MinMaxScaler
  from sklearn.preprocessing import FunctionTransformer
  min_max_scaler = MinMaxScaler()
  # Use this for normal scaled factors 0..1
  scaled_vals = min_max_scaler.fit_transform(df['reward'].values.reshape(df['reward'].values.shape[0], 1))
  # Use this for log-scale rewards
  #log_transformer = FunctionTransformer(np.log1p, validate=True)
  #scaled_vals = min_max_scaler.fit_transform(log_transformer.transform(df['reward'].values.reshape(df['reward'].values.shape[0], 1)))

  df['scaled_reward'] = pd.DataFrame(scaled_vals.squeeze())

  # reward graph per episode
  min_episode = np.min(df['episode'])
  max_episode = np.max(df['episode'])
  print('Number of episodes = ', max_episode - min_episode)

  # Gather per-episode metrics
  total_reward_per_episode = list()
  total_progress_per_episode = list()
  pace_per_episode = list()
  # dive into per-step rewards
  mean_step_reward_per_episode = list()
  min_step_reward_per_episode = list()
  max_step_reward_per_episode = list()

  for epi in range(min_episode, max_episode+1):
    df_slice = df[df['episode'] == epi]
    #    total_reward_per_episode.append(np.sum(df_slice['scaled_reward']))
    total_reward_per_episode.append(np.sum(df_slice['reward']))
    total_progress_per_episode.append(np.max(df_slice['progress']))
    elapsed_time = float(np.max(df_slice[tc])) - float(np.min(df_slice[tc]))
    pace_per_episode.append(elapsed_time * (100 / np.max(df_slice['progress'])))
    mean_step_reward_per_episode.append(np.mean(df_slice['reward']))
    min_step_reward_per_episode.append(np.min(df_slice['reward']))
    max_step_reward_per_episode.append(np.max(df_slice['reward']))

  # Generate per-iteration averages
  average_reward_per_iteration = list()
  deviation_reward_per_iteration = list()
  buffer_rew = list()
  for val in total_reward_per_episode:
    buffer_rew.append(val)
    if len(buffer_rew) == ei:
        average_reward_per_iteration.append(np.mean(buffer_rew))
        deviation_reward_per_iteration.append(np.std(buffer_rew))
        buffer_rew = list()

  average_mean_step_reward_per_iteration = list()
  deviation_mean_step_reward_per_iteration = list()
  buffer_rew = list()
  for val in mean_step_reward_per_episode:
    buffer_rew.append(val)
    if len(buffer_rew) == ei:
        average_mean_step_reward_per_iteration.append(np.mean(buffer_rew))
        deviation_mean_step_reward_per_iteration.append(np.std(buffer_rew))
        buffer_rew = list()

  average_pace_per_iteration = list()
  buffer_rew = list()
  for val in pace_per_episode:
    buffer_rew.append(val)
    if len(buffer_rew) == ei:
        average_pace_per_iteration.append(np.mean(buffer_rew))
        buffer_rew = list()

  average_progress_per_iteration = list()
  lap_completion_per_iteration = list()
  buffer_rew = list()
  for val in total_progress_per_episode:
    buffer_rew.append(val)
    if len(buffer_rew) == ei:
        average_progress_per_iteration.append(np.mean(buffer_rew))
        lap_completions = buffer_rew.count(100.0)
        lap_completion_per_iteration.append(lap_completions / ei)
        buffer_rew = list()
        
  # Plot the data
  fig = plt.figure(figsize=(16, 20))

  ax = fig.add_subplot(511)

  for rr in range(len(average_reward_per_iteration)):
    if average_reward_per_iteration[rr] >= rt :
        ax.plot(rr, average_reward_per_iteration[rr], 'r.')

  line1, = ax.plot(np.arange(len(deviation_reward_per_iteration)), deviation_reward_per_iteration)
  ax.set_ylabel('dev episode reward')
  ax.set_xlabel('Iteration')
  ax = ax.twinx()
  line2, = ax.plot(np.arange(len(deviation_mean_step_reward_per_iteration)), deviation_mean_step_reward_per_iteration, color='g')
  ax.set_ylabel('dev step reward')
  plt.legend((line1, line2), ('dev episode reward', 'dev step reward'))
  plt.grid(True)

  for rr in range(len(average_reward_per_iteration)):
    if average_reward_per_iteration[rr] >= rt:
        ax.plot(rr, deviation_reward_per_iteration[rr], 'r.')


  ax = fig.add_subplot(512)
  ax.plot(np.arange(len(total_reward_per_episode)), total_reward_per_episode, '.')
  ax.plot(np.arange(0, len(average_reward_per_iteration)*ei, ei), average_reward_per_iteration)
  ax.set_ylabel('Total Episode Rewards')
  ax.set_xlabel('Episode')
  plt.grid(True)

  ax = fig.add_subplot(513)
  ax.plot(np.arange(len(mean_step_reward_per_episode)), mean_step_reward_per_episode, '.')
  ax.plot(np.arange(0, len(average_mean_step_reward_per_iteration)*ei, ei), average_mean_step_reward_per_iteration)
  #min_reward_per_episode = list()
  #max_reward_per_episode = list()
  ax.set_ylabel('Mean Step Rewards')
  ax.set_xlabel('Episode')
  plt.grid(True)

  ax = fig.add_subplot(514)
  ax.plot(np.arange(len(total_progress_per_episode)), total_progress_per_episode, '.')
  line1, = ax.plot(np.arange(0, len(average_progress_per_iteration)*ei, ei), average_progress_per_iteration)
  ax.set_ylabel('Progress')
  ax.set_xlabel('Episode')
  ax.set_ylim(0,100)
  ax = ax.twinx()
  ax.set_ylabel('Completion Ratio')
  ax.set_ylim(0,1.0)
  line2, = ax.plot(np.arange(0, len(lap_completion_per_iteration)*ei, ei), lap_completion_per_iteration)
  ax.axhline(0.2, linestyle='--', color='r') # Target minimum 20% completion ratio
  ax.axhline(0.4, linestyle='--', color='g') # Completion of 40% should accomodate higher speeds
  plt.legend((line1, line2), ('mean progress', 'completion ratio'), loc='upper left')
  plt.grid(True)

  
  ax = fig.add_subplot(515)
  ax.plot(np.arange(len(pace_per_episode)), pace_per_episode, '.')
  ax.plot(np.arange(0, len(average_pace_per_iteration)*ei, ei), average_pace_per_iteration)
  ax.set_ylim(np.mean(pace_per_episode) - 2 * np.std(pace_per_episode), np.mean(pace_per_episode) + 2 * np.std(pace_per_episode))
  ax.set_ylabel('Pace')
  ax.set_xlabel('Episode')
  plt.grid(True)

  plt.show()

  # See if rewards correlate with pace
  print("Lower pace should equate to higher rewards")

  df_mean_step_reward_per_episode = pd.DataFrame(list(zip(mean_step_reward_per_episode, 
                                                        pace_per_episode,
                                                        total_progress_per_episode)), 
                                               columns=['mean_step_reward','pace', 'progress'])
  cropped = df_mean_step_reward_per_episode[abs(df_mean_step_reward_per_episode - np.mean(df_mean_step_reward_per_episode)) < psd * np.std(df_mean_step_reward_per_episode)]
  ax = cropped.plot.scatter('mean_step_reward',
                          'pace', 
                          c='progress', 
                          colormap='viridis', 
                          figsize=(12,8))
  ax.set_xlabel('Mean step reward per episode')
  ax.set_ylabel('Mean pace per episode')
