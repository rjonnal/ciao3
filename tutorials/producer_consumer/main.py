import threading
import time
import logging
import random
import queue
import numpy as np
from dearpygui import core, simple

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)

BUF_SIZE = 10
N = 256




class ProducerThread(threading.Thread):
    def __init__(self, q, group=None, target=None, name=None,
                 args=(), kwargs=None, verbose=None):
        super(ProducerThread,self).__init__()
        self.target = target
        self.name = name
        self.q = q
        self.t0 = time.time()
        self.steps = 0
        self.fps = 0.0
    def run(self):
        while True:
            if not self.q.full():
                item = np.round(np.random.rand(N,N)).astype(np.uint8)
                f = np.fft.fft2(item)
                self.q.put(item)
                logging.debug('Putting ' + '%0.1f'%item[0,0]
                              + ' : ' + str(self.q.qsize()) + ' items in queue'
                              + ' : ' + '%0.1f fps'%self.fps)
                self.steps = self.steps + 1
                self.dt = time.time()-self.t0
                self.fps = float(self.steps)/self.dt
                
        return

class ConsumerThread(threading.Thread):
    def __init__(self, q, group=None, target=None, name=None,
                 args=(), kwargs=None, verbose=None):
        super(ConsumerThread,self).__init__()
        self.target = target
        self.name = name
        self.q = q
        self.t0 = time.time()
        self.steps = 0
        self.fps = 0.0
        
    def run(self):
        while True:
            if not self.q.empty():
                item = self.q.get()
                logging.debug('Getting ' + '%0.1f'%item[0,0]
                              + ' : ' + str(self.q.qsize()) + ' items in queue'
                              + ' : ' + '%0.1f fps'%self.fps)
                self.steps = self.steps + 1
                self.dt = time.time()-self.t0
                self.fps = float(self.steps)/self.dt

if __name__ == '__main__':

    systems = []
    for k in range(1):
        q = queue.Queue(BUF_SIZE)
        p = ProducerThread(q,name='producer %02d'%k)
        c = ConsumerThread(q,name='consumer %02d'%k)
        systems.append((p,c,q))


    def save_callback(sender, data):
        print("Save Clicked")

    with simple.window("Example Window"):
        core.add_text("Hello world")
        core.add_button("Save", callback=save_callback)
        core.add_input_text("string")
        core.add_slider_float("float")

    core.start_dearpygui()
        
    for idx,system in enumerate(systems):
        logging.debug('Starting system %d.'%idx)
        system[0].start()
        time.sleep(.1)
        system[1].start()
        time.sleep(.1)
        
