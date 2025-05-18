import re
import matplotlib.pyplot as plt
import json
import numpy as np

# Path to your log file
log_file_path = [r'C:\Users\lakshay\Desktop\detection\Co-DETR\train\20250502_061346.log.json',r'C:\Users\lakshay\Desktop\detection\Co-DETR\train\20250504_200241.log.json',r'C:\Users\lakshay\Desktop\detection\Co-DETR\train\20250506_232955.log.json',r'C:\Users\lakshay\Desktop\detection\Co-DETR\train\20250507_180824.log.json',r'C:\Users\lakshay\Desktop\detection\Co-DETR\train\20250508_125620.log.json',r'C:\Users\lakshay\Desktop\detection\Co-DETR\train\20250511_213123.log.json',r'C:\Users\lakshay\Desktop\detection\Co-DETR\train\20250513_024351.log.json',r'C:\Users\lakshay\Desktop\detection\Co-DETR\train\20250516_205702.log.json']

data = []
for log_file in log_file_path:
    with open(log_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))

count = -1
loss = []
loss_val = []
for entry in data:
    if 'epoch' in entry and 'loss' in entry:
        if entry['epoch']>len(loss):
            if len(loss):
                loss[len(loss)-1] = sum(loss[len(loss)-1])/len(loss[len(loss)-1])
            loss.append([entry['loss']])
        else:
            loss[entry['epoch']-1].append(entry['loss'])
loss[len(loss)-1] = sum(loss[len(loss)-1])/len(loss[len(loss)-1])
np.save(r'C:\Users\lakshay\Desktop\detection\Co-DETR\loss.py', np.array(loss))
print(np.array(loss_val))

plt.figure(figsize=(20,20))
plt.plot(np.array(loss))
plt.xlabel('epochs')
plt.ylabel('loss')
plt.savefig(r"C:\Users\lakshay\Desktop\detection\Co-DETR\loss_fig.png")