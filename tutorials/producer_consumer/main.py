import threading
import time
import logging
import random
import queue
import numpy as np

logging.basicConfig(level=logging.DEBUG,
                    format='(%(threadName)-9s) %(message)s',)

BUF_SIZE = 10

class ProducerThread(threading.Thread):
    def __init__(self, q, group=None, target=None, name=None,
                 args=(), kwargs=None, verbose=None):
        super(ProducerThread,self).__init__()
        self.target = target
        self.name = name
        self.q = q
        
    def run(self):
        while True:
            if not self.q.full():
                item = np.sum(np.random.randn(1000000))
                self.q.put(item)
                logging.debug('Putting ' + '%0.0f'%item  
                              + ' : ' + str(self.q.qsize()) + ' items in queue')
                time.sleep(random.random())
        return

class ConsumerThread(threading.Thread):
    def __init__(self, q, group=None, target=None, name=None,
                 args=(), kwargs=None, verbose=None):
        super(ConsumerThread,self).__init__()
        self.target = target
        self.name = name
        self.q = q

    def run(self):
        while True:
            if not self.q.empty():
                item = self.q.get()
                logging.debug('Getting ' + '%0.0f'%item
                              + ' : ' + str(self.q.qsize()) + ' items in queue')
                time.sleep(random.random())
        return

if __name__ == '__main__':

    systems = []
    for k in range(10):
        q = queue.Queue(BUF_SIZE)
        p = ProducerThread(q,name='producer %02d'%k)
        c = ConsumerThread(q,name='consumer %02d'%k)
        systems.append((p,c,q))

    for idx,system in enumerate(systems):
        logging.debug('Starting system %d.'%idx)
        system[0].start()
        time.sleep(.1)
        system[1].start()
        time.sleep(.1)
        
