import numpy as np
import time
import sys

class FrameTimer:
    def __init__(self,label,buffer_size=100,verbose=False):
        self.index = 0
        self.fps = 0.0
        self.frame_time = 0.0
        self.frame_rms = 0.0
        self.buff = np.zeros(buffer_size)
        self.buffer_size = buffer_size
        self.label = label
        self.verbose = verbose
        
    def tick(self):
        self.buff[self.index] = time.time()
        self.index = self.index + 1
        if self.index==self.buffer_size:
            # buffer full--compute
            dt = np.diff(self.buff)
            self.frame_time = dt.mean()
            self.frame_rms = dt.std()
            self.fps = 1.0/self.frame_time
            self.index=0
            if self.verbose:
                print('%s: %0.1f (ms) %0.1f (ms std) %0.1f (fps)'%(self.label,1000.*self.frame_time,1000.*self.frame_rms,self.fps))


class BlockTimer:
    def __init__(self,timer_label):
        self.timer_label = timer_label
        self.tick_dict = {}
        self.labels = []

    def tick(self,label):
        if label in self.labels:
            self.tick_dict[label].append(time.time())
        else:
            self.labels.append(label)
            self.tick_dict[label] = [time.time()]

    def tock(self):
        #print(self.timer_label)
        ragged_arr = []
        for label in self.labels:
            ragged_arr.append(self.tick_dict[label])

        smallest_len = np.inf
        for vec in ragged_arr:
            if len(vec)<smallest_len:
                smallest_len = len(vec)

        arr = []
        for vec in ragged_arr:
            arr.append(vec[:smallest_len])

        arr = np.array(arr)

        darr = np.diff(arr,axis=0)
        dt = np.mean(darr,axis=1)
        for idx in range(1,len(self.labels)):
            lab1 = self.labels[idx-1]
            lab2 = self.labels[idx]
            print('%s -> %s: %0.3f ms'%(lab1,lab2,1000.*dt[idx-1]))
        print()
