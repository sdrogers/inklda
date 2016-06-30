# Some useful functions for imaging data
import numpy as np
import pylab as plt
import plotly as plotly
from plotly.graph_objs import *
plotly.offline.init_notebook_mode()


class ImagingGrid(object):
	def __init__(self,n_rows = 6,width=50.0,drop = 3.0,time_between_scans = 1.25,n_scans = 2281):
		self.n_rows = n_rows
		self.width = width
		self.drop = drop
		self.time_between_scans = time_between_scans
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

	def plot_linear(self,vals):
		f = plt.figure(figsize=(20,10))
		cols = ['r','b','g']
		for i,topic in enumerate(vals):
			for doc in topic:
				plt.plot(int(doc),topic[doc],cols[i]+'o')
				# plt.plot([int(doc),int(doc)],[0,topic[doc]],cols[i])
		return f



def plot_topic_dict(topic_dict,label_thresh=0.02):
    plt.figure(figsize=(30,10))
    for word in topic_dict:
        mass = float(word)
        plt.plot([mass,mass],[0,topic_dict[word]],'k',linewidth=2)
        if topic_dict[word] > label_thresh:
            plt.text(mass,topic_dict[word],word,fontsize=14)

def plot_topic_dict_plotly(topic_dict,thresh=0.001):
	x = []
	y = []
	for word in topic_dict:
		if topic_dict[word] >= thresh:
			x.append(float(word))
			y.append(topic_dict[word])

	data1 = Bar(
		x = x,
		y = y,
		marker=dict(
                line=dict(
                    color='rgb(8,48,107)',
                    width=1),
            ),
		hoverinfo='none',
		)
	data2 = Scatter(
		x=x,
		y=y,
		mode='markers',
		hoverinfo = 'xy',
	)

	layout = Layout(
		hovermode='closest',
		)

	plotly.offline.iplot([data1,data2],[layout])


def plot_topic_dict_diff_plotly(topic_dict,topic_dict_2,names=None):
	x = []
	y = []
	for word in topic_dict:
		x.append(float(word))
		y.append(topic_dict[word])

	
	plotly.offline.init_notebook_mode()


	data1 = Bar(
		x = x,
		y = y,
		marker=dict(
                line=dict(
                    width=1),
            ),
		)
	data2 = Scatter(
		x=x,
		y=y,
		mode='markers',
		name = names[0],
		text=["Mass: {}".format(a) for a in x],
		marker=dict(
			color="rgb(0,255,0)")
	)

	x2 = []
	y2 = []
	for word in topic_dict_2:
		x2.append(float(word))
		y2.append(topic_dict_2[word])



	data3 = Bar(
		x = x2,
		y = [-a for a in y2],
		marker=dict(
                line=dict(
                    width=1),
            ),
		)
	data4 = Scatter(
		x=x2,
		y=[-a for a in y2],
		mode='markers',
		name = names[1],
		text=["Mass: {}".format(a) for a in x2],
		marker = dict(
			color = "rgb(255,0,0)"
			)
	)

	plotly.offline.iplot([data1,data2,data3,data4])