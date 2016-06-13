import numpy as np

# class Word(object):
# 	def __init__(self,name,index):
# 		self.name = name
# 		self.index = index

class LDA(object):
	def __init__(self,corpus,K=20,alpha=1,beta=1):
		self.corpus = corpus
		self.K = K
		self.alpha = alpha
		self.beta = beta
		self.collect_words()
		self.initialise()

	def collect_words(self):
		self.words = []
		self.nwords = 0
		self.ndocs = len(self.corpus)
		docpos = 0
		self.doc_index = {}
		self.word_index = {}
		for doc in self.corpus:
			self.doc_index[doc] = docpos
			docpos += 1
			for word in self.corpus[doc]:
				if not word in self.word_index:
					self.word_index[word] = self.nwords
					self.nwords += 1

		


	def initialise(self):
		self.Z = {}
		self.doc_topic_counts = np.zeros((self.K,self.ndocs),np.int) + self.alpha
		self.topic_word_counts = np.zeros((self.K,self.nwords),np.int) + self.beta
		self.topic_totals = np.zeros((self.K),np.int) + self.beta
		self.total_words = 0
		self.word_counts = {}
		for word in self.word_index:
			self.word_counts[word] = 0

		for doc in self.corpus:
			self.Z[doc] = {}
			di = self.doc_index[doc]
			for word in self.corpus[doc]:
				wi = self.word_index[word]
				count = self.corpus[doc][word]
				self.total_words += count
				self.word_counts[word] += count
				self.Z[doc][word] = []
				for c in range(count):
					topic = np.random.randint(self.K)
					self.topic_totals[topic] += 1
					self.Z[doc][word].append(topic)
					self.doc_topic_counts[topic,di] += 1
					self.topic_word_counts[topic,wi] += 1

		# Output things
		self.post_sample_count = 0.0
		self.post_mean_theta = np.zeros((self.K,self.ndocs),np.float)
		self.post_mean_topics = np.zeros((self.K,self.nwords),np.float)


	def gibbs_iteration(self,n_samples = 1,verbose = True,burn = True):
		# Does one gibbs step
		for sample in range(n_samples):
			if verbose:
				print "Sample {} of {} (Burn is {})".format(sample,n_samples,burn)
			for doc in self.corpus:
				di = self.doc_index[doc]
				for word in self.corpus[doc]:
					wi = self.word_index[word]
					for i,instance in enumerate(self.Z[doc][word]):
						current_topic = instance
						self.doc_topic_counts[current_topic,di] -= 1
						self.topic_word_counts[current_topic,wi] -= 1
						self.topic_totals[current_topic] -= 1

						# Re-sample
						p_topic = 1.0*self.topic_word_counts[:,wi] / self.topic_totals
						p_topic *= self.doc_topic_counts[:,di]
						p_topic = 1.0*p_topic / p_topic.sum()
						new_topic = np.random.choice(self.K,p=p_topic)

						self.Z[doc][word][i] = new_topic

						self.doc_topic_counts[new_topic,di] += 1
						self.topic_word_counts[new_topic,wi] += 1
						self.topic_totals[new_topic] += 1

		if not burn:
			self.post_sample_count += 1.0
			for doc in self.corpus:
				di = self.doc_index[doc]
				tcounts = self.doc_topic_counts[:,di]
				self.post_mean_theta[:,di] += np.random.dirichlet(tcounts)
			for topic in range(self.K):
				wcounts = self.topic_word_counts[topic,:]
				self.post_mean_topics[topic,:] += np.random.dirichlet(wcounts)
			
	def get_post_mean_theta(self):
		return self.post_mean_theta / self.post_sample_count
	def get_post_mean_topics(self):
		return self.post_mean_topics / self.post_sample_count

	def get_mass_plot(self,topic_id):
		pmeantopics = self.get_post_mean_topics()
		m = []
		probs = []
		for word in self.word_index:
			m.append(float(word))
			probs.append(pmeantopics[topic_id,self.word_index[word]])

		m_probs = zip(m,probs)
		m_probs = sorted(m_probs,key=lambda x: x[0])
		m,probs = zip(*m_probs)
		return np.array(m),np.array(probs)


	def plot_topic(self,topic_id,nrows = 10,ncols = 10):

		image_array = np.zeros((nrows,ncols),np.float)
		for doc in self.corpus:
			di = self.doc_index[doc]
			if self.post_sample_count == 0:
				tprobs = self.doc_topic_counts[:,di]
				tprobs = tcounts / 1.0*tcounts.sum()
			else:
				tprobs = self.get_post_mean_theta()
			(r,c) = doc
			image_array[r,c] = tprobs[topic_id,di]

		return image_array



