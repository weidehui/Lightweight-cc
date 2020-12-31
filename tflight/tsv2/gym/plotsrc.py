import json
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import sys
import math
for i in range(1,2):
    #filename = 'pcc.json'
    filename = 'data/pcc-c-very/pcc0.json'
    #filename = 'pcc_env_log_run_'+str(i)+'00.json'
    data = {}
    with open(filename) as f:
        data = json.load(f)
    time_data = [float(event["Time"]) for event in data["Events"][1:]]
    thpt_data = [float(event["Throughput"]) for event in data["Events"][1:]]
    delay_data = [float(event["Latency"]) for event in data["Events"][1:]]
    bw_data = [float(event["bandwidth"])*10000 for event in data["Events"][1:]]
    fig, ax = plt.subplots()
    ax.plot(time_data, bw_data)
    ax.set_ylabel("Bandwith", fontsize=12)
    ax.set_xlabel("Monitor Interval", fontsize=12)
    fig.suptitle("Summary Graph ")
    fig.savefig('data/src.jpg', bbox_inches='tight', pad_inches=0.2)

#filename2 = 'pcc_env_log_run_2000.json'
#data2 = {}
#with open(filename2) as f:
#    data2 = json.load(f)
#time_data2 = [float(event["Time"]) for event in data2["Events"][1:]]
#thpt_data2 = [float(event["Throughput"]) for event in data2["Events"][1:]]
#ax.plot(time_data2, thpt_data2)
#xmin = int(math.floor(min(time_data)))
#xmax = int(math.ceil(max(time_data)))
#ax.set_xlim(xmin, xmax)
#new_xticks = range(xmin, xmax, 10)
#ax.set_xticks(new_xticks)
#formatter = ticker.FuncFormatter(lambda x, pos: x - xmin)
#ax.xaxis.set_major_formatter(formatter)
#fig_w, fig_h = fig.get_size_inches()
