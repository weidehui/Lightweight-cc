import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sys
import math
from tqdm import tqdm
#file=['new-false-1e12','new-false-2','old-false-1e12','old-false-2','new-true-2','new-true-1e12','old-true-2','old-true-1e12','src-200']

def plotRes(file,kind):
    for ff in tqdm(file):
        filename=['result'+str(kind)+'/'+ff+'/pcc'+str(i)+'.json' for i in range(0,9)]
        fig, ax = plt.subplots(len(filename),3,figsize=(15,20))
        for i in range(0,len(filename)):
            with open(filename[i]) as f:
                data = json.load(f)
            time_data = [float(event["Time"]) for event in data["Events"][1:]]
            thpt_data = [float(event["Send Rate"]) for event in data["Events"][1:]]
            thpt_data2 = [float(event["Throughput"]) for event in data["Events"][1:]]
            delay_data = [float(event["Latency"]) for event in data["Events"][1:]]
            bw_data = [float(event["bandwidth"])*10000 for event in data["Events"][1:]]
            ax[i,0].plot(time_data, thpt_data)
            ax[i,0].fill_between(time_data, 0, bw_data,facecolor='linen')
            ax[i,0].set_ylabel("Send Rate", fontsize=12)
            ax[i,0].set_xlabel("Monitor Interval", fontsize=12)
            ax[i,1].plot(time_data, thpt_data2)
            ax[i,1].fill_between(time_data, 0, bw_data,facecolor='linen')
            ax[i,1].set_ylabel("Throughput", fontsize=12)
            ax[i,1].set_xlabel("Monitor Interval", fontsize=12)
            ax[i,2].scatter(time_data, delay_data)
            ax[i,2].set_ylabel("Delay", fontsize=12)
            ax[i,2].set_xlabel("Monitor Interval", fontsize=12)
        fig.savefig('result'+str(kind)+'/'+ff+'/res.jpg', bbox_inches='tight', pad_inches=0.2)