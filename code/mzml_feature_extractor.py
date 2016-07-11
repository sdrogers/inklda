
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