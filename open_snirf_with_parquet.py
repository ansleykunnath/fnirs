import numpy as np
import matplotlib.pyplot as plt
from mne.io.snirf import read_raw_snirf

path_to_snirf = "Subject_002.snirf"     # change to file name
raw = read_raw_snirf(path_to_snirf)
data = raw.get_data()
channel_names = [x['ch_name'] for x in raw.info['chs']]
is_hbo = np.array([1 if channel_name.endswith('HbO') else 0 for channel_name in channel_names], dtype=bool)
oxyHb = np.transpose(data[is_hbo,:])
deoxyHb = np.transpose(data[~is_hbo,:])

from pandas import read_parquet
import calendar
df = read_parquet('Subject_002.parquet')     # change to file name
start_rec_unix = calendar.timegm(raw.annotations.orig_time.timetuple())
start_exp_unix = df.timestamp.iat[0]
audio_t = df[(df['event'] == 'event_audio')]
audio_t = audio_t['timestamp'] - (start_rec_unix)
audio_t = audio_t.to_numpy()
visual_t = df[(df['event'] == 'event_visual')]
visual_t = visual_t['timestamp'] - (start_rec_unix)
visual_t = visual_t.to_numpy()
audiovisual_t = df[(df['event'] == 'event_audiovisual')]
audiovisual_t = audiovisual_t['timestamp'] - (start_rec_unix)
audiovisual_t = audiovisual_t.to_numpy()
written_t = df[(df['event'] == 'event_written')]
written_t = written_t['timestamp'] - (start_rec_unix)
written_t = written_t.to_numpy()
rest_t = df[(df['event'] == 'event_rest')]
rest_t = rest_t['timestamp'] - (start_rec_unix)
rest_t = rest_t.to_numpy()

def plot_all_channels():
   _, ax = plt.subplots(ncols=2, figsize=(10, 5))

   ax[0].pcolor(raw.times, np.arange(np.sum(is_hbo)),
               data[is_hbo, :], shading='nearest')
   ax[0].set_title('HbO')
   ax[0].set_xlabel('Time [s]')
   ax[0].set_ylabel('Channel number')

   ax[1].pcolor(raw.times, np.arange(np.sum(~is_hbo)),
               data[~is_hbo, :], shading='nearest')
   ax[1].set_title('HbR')
   ax[1].set_xlabel('Time [s]')

def plot_all_events():
   plt.plot(raw.times,oxyHb, color='red')
   plt.plot(raw.times,deoxyHb, color='blue')
   [plt.axvline(x, linewidth=1, color='#4477AA', label="Audio") for x in audio_t]
   [plt.axvline(x, linewidth=1, color='#EE7733', label="Visual") for x in visual_t]
   [plt.axvline(x, linewidth=1, color='#CCBB44', label="Audiovisual") for x in audiovisual_t]
   [plt.axvline(x, linewidth=1, color='#AA3377', label="Written") for x in written_t]
   [plt.axvline(x, linewidth=1, color='black', label="Rest") for x in rest_t]

c = 190    # set channel here

def plot_channel():
   plt.plot(raw.times,oxyHb[:,c], color='red')
   plt.plot(raw.times,deoxyHb[:,c], color='blue')

def plot_channel_all_events():
   plt.plot(raw.times,oxyHb[:,c], color='red')
   plt.plot(raw.times,deoxyHb[:,c], color='blue')
   [plt.axvline(x, linewidth=1, color='#4477AA', label="Audio") for x in audio_t]
   [plt.axvline(x, linewidth=1, color='#EE7733', label="Visual") for x in visual_t]
   [plt.axvline(x, linewidth=1, color='#CCBB44', label="Audiovisual") for x in audiovisual_t]
   [plt.axvline(x, linewidth=1, color='black', label="Rest") for x in rest_t]

def plot_channel_average_audio():
   stack_o = np.empty(shape=(0,14),dtype=object)
   stack_d = np.empty(shape=(0,14),dtype=object)
   for i in range(0,len(audio_t)):
      start_t = audio_t[i]
      end_t = start_t+2
      block_n = np.array(np.where((raw.times>=start_t) & (raw.times<=end_t)))
      start_n = block_n[0,0]
      end_n = start_n+14
      y1 = oxyHb[start_n:end_n,c]
      stack_o = np.vstack([stack_o,y1])
      y2 = deoxyHb[start_n:end_n,c]
      stack_d = np.vstack([stack_d,y2])
   stack_o_mean = np.mean(stack_o, axis=0)
   stack_d_mean = np.mean(stack_d, axis=0)
   block_t = raw.times[(raw.times >= start_t) & (raw.times <= end_t)] - start_t
   plt.plot(block_t, stack_o_mean, color='red')
   plt.plot(block_t, stack_d_mean, color='blue')

def plot_channel_average_visual():
   stack_o = np.empty(shape=(0,15),dtype=object)
   stack_d = np.empty(shape=(0,15),dtype=object)
   for i in range(0,len(visual_t)):
      start_t = visual_t[i]
      end_t = start_t+2
      block_n = np.array(np.where((raw.times>=start_t) & (raw.times<=end_t)))
      start_n = block_n[0,0]
      end_n = start_n + 15
      y1 = oxyHb[start_n:end_n,c]
      stack_o = np.vstack([stack_o,y1])
      y2 = deoxyHb[start_n:end_n,c]
      stack_d = np.vstack([stack_d,y2])
   stack_o_mean = np.mean(stack_o, axis=0)
   stack_d_mean = np.mean(stack_d, axis=0)
   block_t = raw.times[(raw.times >= start_t) & (raw.times <= end_t)] - start_t
   plt.plot(block_t, stack_o_mean, color='red')
   plt.plot(block_t, stack_d_mean, color='blue')

def plot_channel_average_audiovisual():
   stack_o = np.empty(shape=(0,14),dtype=object)
   stack_d = np.empty(shape=(0,14),dtype=object)
   for i in range(0,len(audiovisual_t)):
      start_t = audiovisual_t[i]
      end_t = start_t+2
      block_n = np.array(np.where((raw.times>=start_t) & (raw.times<=end_t)))
      start_n = block_n[0,0]
      end_n = start_n+14
      y1 = oxyHb[start_n:end_n,c]
      stack_o = np.vstack([stack_o,y1])
      y2 = deoxyHb[start_n:end_n,c]
      stack_d = np.vstack([stack_d,y2])
   stack_o_mean = np.mean(stack_o, axis=0)
   stack_d_mean = np.mean(stack_d, axis=0)
   block_t = raw.times[(raw.times >= start_t) & (raw.times <= end_t)] - start_t
   plt.plot(block_t, stack_o_mean, color='red')
   plt.plot(block_t, stack_d_mean, color='blue')

def plot_channel_average_rest():
   stack_o = np.empty(shape=(0,107),dtype=object)
   stack_d = np.empty(shape=(0,107),dtype=object)
   for i in range(0,len(rest_t)):
      start_t = rest_t[i]
      end_t = start_t+15
      block_n = np.array(np.where((raw.times>=start_t) & (raw.times<=end_t)))
      start_n = block_n[0,0]
      end_n = start_n+107
      y1 = oxyHb[start_n:end_n,c]
      stack_o = np.vstack([stack_o,y1])
      y2 = deoxyHb[start_n:end_n,c]
      stack_d = np.vstack([stack_d,y2])
   stack_o_mean = np.mean(stack_o, axis=0)
   stack_d_mean = np.mean(stack_d, axis=0)
   block_t = raw.times[(raw.times >= start_t) & (raw.times <= end_t)] - start_t
   plt.plot(block_t, stack_o_mean, color='red')
   plt.plot(block_t, stack_d_mean, color='blue')

def plot_channel_average_events():
   # find average for audio over 2 seconds
   stack_o = np.empty(shape=(0,14),dtype=object)
   for i in range(0,len(audio_t)):
      start_t = audio_t[i]
      end_t = start_t+2
      block_n = np.array(np.where((raw.times>=start_t) & (raw.times<=end_t)))
      start_n = block_n[0,0]
      end_n = start_n+14
      y1 = oxyHb[start_n:end_n,c]
      stack_o = np.vstack([stack_o,y1])
   stack_audio = np.mean(stack_o, axis=0)
   from statistics import mean
   avg_audio = mean(stack_audio)

   #find average for visual over 2 seconds
   stack_o = np.empty(shape=(0,15),dtype=object)
   for i in range(0,len(visual_t)):
      start_t = visual_t[i]
      end_t = start_t+2
      block_n = np.array(np.where((raw.times>=start_t) & (raw.times<=end_t)))
      start_n = block_n[0,0]
      end_n = start_n+15
      y1 = oxyHb[start_n:end_n,c]
      stack_o = np.vstack([stack_o,y1])
   stack_visual = np.mean(stack_o, axis=0)
   from statistics import mean
   avg_visual = mean(stack_visual)

   #find average for audiovisual over 2 seconds
   stack_o = np.empty(shape=(0,14),dtype=object)
   for i in range(0,len(audiovisual_t)):
      start_t = audiovisual_t[i]
      end_t = start_t+2
      block_n = np.array(np.where((raw.times>=start_t) & (raw.times<=end_t)))
      start_n = block_n[0,0]
      end_n = start_n+14
      y1 = oxyHb[start_n:end_n,c]
      stack_o = np.vstack([stack_o,y1])
   stack_audiovisual = np.mean(stack_o, axis=0)
   from statistics import mean
   avg_audiovisual = mean(stack_audiovisual)

   #find average for rest over 15 seconds
   stack_o = np.empty(shape=(0,14),dtype=object)
   for i in range(0,len(rest_t)):
      start_t = rest_t[i]
      end_t = start_t+2
      block_n = np.array(np.where((raw.times>=start_t) & (raw.times<=end_t)))
      start_n = block_n[0,0]
      end_n = start_n+14
      y1 = oxyHb[start_n:end_n,c]
      stack_o = np.vstack([stack_o,y1])
   stack_rest = np.mean(stack_o, axis=0)
   from statistics import mean
   avg_rest = mean(stack_rest)

   fig = plt.figure()
   ax = fig.add_axes([0, 0, 1, 1])
   condition = ['Audio', 'Visual', 'Audiovisual', 'Rest']
   values = [avg_audio, avg_visual, avg_audiovisual, avg_rest]
   color = ['#4477AA', '#EE7733','#CCBB44', 'black']
   ax.bar(condition, values, color=color)
   ax.spines["bottom"].set_position(("data", 0))

   import scipy.stats as stats
   t, p = stats.ttest_ind(stack_rest, stack_audio, equal_var=True, alternative='two-sided')
   p = format(p,".10f")
   print("Audio p = " + str(p))
   t, p = stats.ttest_ind(stack_rest, stack_visual, equal_var=True, alternative='two-sided')
   p = format(p,".10f")
   print("Visual p = " + str(p))
   t, p = stats.ttest_ind(stack_rest, stack_audiovisual, equal_var=True, alternative='two-sided')
   p = format(p,".10f")
   print("Audiovisual p = " + str(p))


