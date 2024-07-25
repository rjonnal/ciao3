from ciao3.components import centroid
import numpy as np
from matplotlib import pyplot as plt

spots = np.random.rand(128,128)*1000
spots = spots.astype(np.int16)
sb_x_vec = np.arange(10,120,10)
sb_y_vec = np.ones(sb_x_vec.shape)*64
sb_x_vec = sb_x_vec.astype(float)
sb_y_vec = sb_y_vec.astype(float)
sb_bg_vec = np.ones(sb_x_vec.shape)*5.0
spots = spots + 100
sb_half_width_p = 4
iterations = 2
iteration_step_px_p = 1
x_out = np.zeros(sb_x_vec.shape).astype(float)
y_out = np.zeros(sb_x_vec.shape).astype(float)
mean_intensity = np.zeros(sb_x_vec.shape).astype(float)
maximum_intensity = np.zeros(sb_x_vec.shape).astype(float)
minimum_intensity = np.zeros(sb_x_vec.shape).astype(float)
num_threads_p = 1
centroid.compute_centroids(spots,sb_x_vec,sb_y_vec,sb_bg_vec,sb_half_width_p,iterations,iteration_step_px_p,x_out,y_out,mean_intensity,maximum_intensity,minimum_intensity,num_threads_p)
plt.imshow(spots,cmap='gray')
plt.plot(sb_x_vec,sb_y_vec,'r+')
plt.plot(x_out,y_out,'yo')
plt.show()
