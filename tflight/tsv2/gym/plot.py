import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sys
import math
for i in range(1,2):
    #filename = 'pcc.json'
    filename = 'data/pcc0.json'
    #filename = 'pcc_env_log_run_'+str(i)+'00.json'
    data = {}
    with open(filename) as f:
        data = json.load(f)
    time_data = [float(event["Time"]) for event in data["Events"][1:]]
    thpt_data = [float(event["Send Rate"]) for event in data["Events"][1:]]
    delay_data = [float(event["Latency"]) for event in data["Events"][1:]]
    bw_data = [float(event["bandwidth"])*10000 for event in data["Events"][1:]]
    fig, ax = plt.subplots(2)
    ax[0].plot(time_data, thpt_data)
    ax[0].fill_between(time_data, 0, bw_data,facecolor='linen')

    ax[1].scatter(time_data, thpt_data)

    ax[0].set_ylabel("Throughput", fontsize=12)
    ax[0].set_xlabel("Monitor Interval", fontsize=12)
    fig.suptitle("Summary Graph ")
    fig.savefig('data/2.jpg', bbox_inches='tight', pad_inches=0.2)

