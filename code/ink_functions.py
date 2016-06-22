# Some useful functions for imaging data
import numpy as np
import pylab as plt
class ImagingGrid(object):
	def __init__(self,n_rows = 6,width=50.0,drop = 3.0,time_between_scans = 1.25,n_scans = 2281):
		self.n_rows = n_rows
		self.width = 50.0
		self.drop = 3.0
		self.time_between_scans = 1.25
		self.n_scans = n_scans

		

		self.compute_coords()

	def compute_coords(self):
		self.rowcoord = []
		self.colcoord = []
		self.total_distance = (self.width + self.drop) * self.n_rows
		self.robot_speed = 1.0*self.total_distance/(self.n_scans*self.time_between_scans)
		for spec_id in range(self.n_scans+1):
		    time_elapsed = spec_id * self.time_between_scans
		    distance_traveled = time_elapsed * self.robot_speed
		    row_id = distance_traveled // (self.width + self.drop)
		    current_row = row_id * self.drop
		    current_col = distance_traveled % (self.width + self.drop)
		    if current_col > self.width:
		        current_row += current_col - self.width
		        current_col = self.width
		    if int(row_id) % 2 == 1:
		        current_col = self.width - current_col
		    self.rowcoord.append(self.n_rows*self.drop - current_row)
		    self.colcoord.append(current_col)

	def plot(self,vals = None,max_marker_size = 50.0,color = 'r',figsize=(10,10)):
		max_val = 0.0
		for v in vals:
			if vals[v] > max_val:
				max_val = vals[v]
		min_to_plot = 1.0
		plt.figure(figsize=figsize)
		plt.plot(self.colcoord,self.rowcoord,'k.')
		for v in vals:
			x = self.colcoord[int(v)]
			y = self.rowcoord[int(v)]
			m_size = max_marker_size*vals[v]/max_val
			if m_size >= min_to_plot:
				plt.plot(x,y,color+'o',markersize=max_marker_size*vals[v]/max_val)

	def plot_topic_dict(topic_dict,label_thresh=0.05):
	    plt.figure(figsize=(30,10))
	    for word in topic_dict:
	        mass = float(word)
	        plt.plot([mass,mass],[0,topic_dict[word]],'k',linewidth=2)
	        if topic_dict[word] > label_thresh:
	            plt.text(mass,topic_dict[word],word,fontsize=24)

