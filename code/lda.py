import numpy as np
from scipy.special import psi as psi
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

	def get_topic_as_dict(self,topic_id,thresh = 0.001):
		pmt = self.get_post_mean_topics()
		top = {}
		for word in self.word_index:
			pos = self.word_index[word]
			if pmt[topic_id,pos] >= thresh:
				top[word] = pmt[topic_id,pos]
		return top

	def get_topic_as_doc_dict(self,topic_id,thresh = 0.001):
		pmth = self.get_post_mean_theta()
		top = {}
		for doc in self.doc_index:
			pos = self.doc_index[doc]
			if pmth[topic_id,pos] >= thresh:
				top[doc] = pmth[topic_id,pos]
		return top
		
	def get_topic_as_tuples(self,topic_id,thresh = 0.001):
		pmth = self.get_post_mean_topics()
		top = []
		for word in self.word_index:
			pos = self.word_index[word]
			if pmth[topic_id,pos] >= thresh:
				top.append((word,pmth[topic_id,pos]))

		return sorted(top,key = lambda x: x[1], reverse=True)



class LDA_Feature_Extractor(object):
	def __init__(self,filename,use_scans = 'even',tol = 50, min_intense = 500, min_occurance = 5, max_occurance = 200,min_mass = 50.0,max_mass = 300.0,min_doc_word_instances = 5,max_doc_word_instances = 200):
		self.tol = tol
		self.min_intense = min_intense
		self.min_occurance = min_occurance
		self.max_occurance = max_occurance
		self.filename = filename
		self.min_mass = min_mass
		self.max_mass = max_mass
		self.use_scans = use_scans
		self.min_doc_word_instances = min_doc_word_instances
		self.max_doc_word_instances = max_doc_word_instances


	def make_corpus(self):
		import pymzml
		total_peaks = 0
		self.word_masses = []
		self.word_names = []
		self.instances = []
		self.total_m = []
		run = pymzml.run.Reader(self.filename,MS1_Precision = 5e-6)
		self.corpus = {}
		spec_pos = 0
		for spectrum in run:
			if self.use_scans == 'even' and spec_pos % 2 == 1:
				spec_pos += 1
				continue
			if self.use_scans == 'odd' and spec_pos % 2 == 0:
				spec_pos += 1
				continue
			new_doc = {}
			max_i = 3000.0 
			min_i = 1e10
			for m,i in spectrum.peaks:
				if i >= self.min_intense and m >= self.min_mass and m <= self.max_mass:
					word = None
					if len(self.word_masses) == 0:
						self.word_masses.append(m)
						self.word_names.append(str(m))
						self.instances.append(1)
						self.total_m.append(m)
						word = str(m)
					else:
						idx = np.abs(m - np.array(self.word_masses)).argmin()
						if not self.hit(m,self.word_masses[idx],self.tol):
							self.word_masses.append(m)
							self.word_names.append(str(m))
							self.instances.append(1)
							self.total_m.append(m)
							word = str(m)
						else:
							self.total_m[idx] += m
							self.instances[idx] += 1
							# self.word_masses[idx] = self.total_m[idx]/self.instances[idx]
							# self.word_names[idx] = str(self.word_masses[idx])
							word = self.word_names[idx]
					if word in new_doc:
						new_doc[word] += i
					else:
						new_doc[word] = i
					if i < min_i:
						min_i = i

			# to_remove = []
			# for word in new_doc:
			# 	if new_doc[word] > max_i:
			# 		new_doc[word] = max_i
			# 	new_doc[word] -= min_i
			# 	new_doc[word] /= (max_i - min_i)
			# 	new_doc[word] *= 100.0
			# 	new_doc[word] = int(new_doc[word])
			# 	if new_doc[word] == 0:
			# 		to_remove.append(word)

			# for word in to_remove:
			# 	del new_doc[word]

			self.corpus[str(spec_pos)] = new_doc
			spec_pos += 1
			if spec_pos % 100 == 0:
				print "Spectrum {} ({} words)".format(spec_pos,len(self.word_names))


		print "Found {} documents".format(len(self.corpus))

		word_counts = {}
		for doc in self.corpus:
			for word in self.corpus[doc]:
				if word in word_counts:
					word_counts[word] += 1
				else:
					word_counts[word] = 1



		to_remove = []
		for word in word_counts:
			if word_counts[word] < self.min_doc_word_instances:
				to_remove.append(word)
			if word_counts[word] > self.max_doc_word_instances:
				to_remove.append(word)


		print "Removing {} words".format(len(to_remove))

		for doc in self.corpus:
			for word in to_remove:
				if word in self.corpus[doc]:
					del self.corpus[doc][word]


	def make_nominal_corpus(self):
		import pymzml
		self.word_names = []
		self.word_masses = []
		self.word_names = []
		run = pymzml.run.Reader(self.filename,MS1_Precision = 5e-6)
		self.corpus = {}
		spec_pos = 0
		for spectrum in run:
			doc = str(spec_pos)
			self.corpus[doc] = {}
			for m,i in spectrum.peaks:
				if m >= self.min_mass and m <= self.max_mass and i >= self.min_intense:
					word = str(np.floor(m))
					if not word in self.word_names:
						self.word_names.append(word)
						self.word_masses.append(float(word))

					if word in self.corpus[doc]:
						self.corpus[doc][word] += i
					else:
						self.corpus[doc][word] = i
			spec_pos += 1



	def hit(self,m1,m2,tol):
	    if 1e6*abs(m1-m2)/m1 < tol:
	        return True
	    else:
	        return False



class VariationalLDA(object):
	def __init__(self,corpus,K = 20):
		self.corpus = corpus
		self.n_docs = len(self.corpus)
		self.word_index = self.find_unuique_words()
		print "Object created with {} documents".format(self.n_docs)
		self.n_words = len(self.word_index)

		self.make_doc_word_matrix()
		print "Document - word matrix created"
		print self.word_matrix
		self.K = K
		self.alpha = 1.0*np.ones(self.K)
		self.eta = 0.1 # Smoothing parameter for beta

	def run_vb(self,n_its = 1,verbose=True):
		print "Initialising"
		self.init_vb()
		print "Starting iterations"
		for it in range(n_its):
			diff = self.vb_step()
			if verbose:
				print "Iteration {} (change = {})".format(it,diff)



	def vb_step(self):
		temp_beta = self.e_step()
		temp_beta /= temp_beta.sum(axis=1)[:,None]
		total_difference = (np.abs(temp_beta - self.beta_matrix)).sum()
		self.beta_matrix = temp_beta
		return total_difference
		# self.m_step()

	def e_step(self):
		temp_beta = np.zeros((self.K,self.n_words)) + self.eta
		for doc in self.corpus:
			d = self.doc_index[doc]
			temp_gamma = np.zeros(self.K) + self.alpha
			for word in self.corpus[doc]:
				w = self.word_index[word]
				self.phi_matrix[doc][word] = self.beta_matrix[:,w]*np.exp(psi(self.gamma_matrix[d,:])).T
				# for k in range(self.K):
				# 	self.phi_matrix[doc][word][k] = self.beta_matrix[k,w]*np.exp(scipy.special.psi(self.gamma_matrix[d,k]))
				self.phi_matrix[doc][word] /= self.phi_matrix[doc][word].sum()
				temp_gamma += self.phi_matrix[doc][word]*self.corpus[doc][word]
				temp_beta[:,w] += self.phi_matrix[doc][word] * self.corpus[doc][word]
			# self.phi_matrix[d,:,:] = (self.beta_matrix * self.word_matrix[d,:][None,:] * (np.exp(scipy.special.psi(self.gamma_matrix[d,:]))[:,None])).T
			# self.phi_matrix[d,:,:] /= self.phi_matrix[d,:,:].sum(axis=1)[:,None]
			# self.gamma_matrix[d,:] = self.alpha + self.phi_matrix[d,:,:].sum(axis=0)
			self.gamma_matrix[d,:] = temp_gamma
		return temp_beta

	def m_step(self):
		for k in range(self.K):
			self.beta_matrix[k,:] = self.eta + (self.word_matrix * self.phi_matrix[:,:,k]).sum(axis=0)
		self.beta_matrix /= self.beta_matrix.sum(axis=1)[:,None]

	def find_unuique_words(self):
		word_index = {}
		pos = 0
		for doc in self.corpus:
			for word in self.corpus[doc]:
				if not word in word_index:
					word_index[word] = pos
					pos += 1
		print "Found {} unique words".format(len(word_index))
		return word_index

	def make_doc_word_matrix(self):
		self.word_matrix = np.zeros((self.n_docs,self.n_words),np.int)
		self.doc_index = {}
		doc_pos = 0
		for doc in self.corpus:
			for word in self.corpus[doc]:
				word_pos = self.word_index[word]
				self.word_matrix[doc_pos,word_pos] = self.corpus[doc][word]
			self.doc_index[doc] = doc_pos
			doc_pos += 1

	def init_vb(self):
		# self.gamma_matrix = np.zeros((self.n_docs,self.K),np.float) + 1.0
		# self.phi_matrix = np.zeros((self.n_docs,self.n_words,self.K))
		self.phi_matrix = {}
		self.gamma_matrix = np.zeros((self.n_docs,self.K))
		for doc in self.corpus:
			self.phi_matrix[doc] = {}
			for word in self.corpus[doc]:
				self.phi_matrix[doc][word] = np.zeros(self.K)
			d = self.doc_index[doc]
			self.gamma_matrix[d,:] = self.alpha + 1.0*sum(self.word_matrix[d,:])/self.K
		# # Normalise this to sum to 1
		# self.phi_matrix /= self.phi_matrix.sum(axis=2)[:,:,None]

		# Initialise the betas
		self.beta_matrix = np.random.rand(self.K,self.n_words)
		self.beta_matrix /= self.beta_matrix.sum(axis=1)[:,None]


	def get_topic_as_doc_dict(self,topic_id,thresh = 0.001,normalise=False):
		top = {}
		mat = self.gamma_matrix
		if normalise:
			mat = self.get_expect_theta()

		for doc in self.doc_index:
			pos = self.doc_index[doc]
			if mat[pos,topic_id] >= thresh:
				top[doc] = mat[pos,topic_id]
		return top

	def get_topic_as_dict(self,topic_id):
		top = {}
		for word in self.word_index:
			top[word] = self.beta_matrix[topic_id,self.word_index[word]]
		return top

	def get_expect_theta(self):
		e_theta = self.gamma_matrix.copy()
		e_theta /= e_theta.sum(axis=1)[:,None]
		return e_theta

	def get_beta(self):
		return self.beta_matrix.copy()



