import numpy as np
from pathlib import Path

DOWNLOAD_DIR = Path('/', 'opt', 'resources')
GLOVE_DIR = DOWNLOAD_DIR.joinpath('embeddings', 'glove')
GLOVE_FILE = GLOVE_DIR.joinpath('glove.840B.300d.txt')

def embeddings(sents):
	'''
		sents: numpy array of sentences to compute embeddings
				
		returns: numpy 2-D array of embeddings;
		length of single embedding is arbitrary, but have to be
		consistent across the whole result
	'''
	where = {}
	words = set()

	result = np.zeros((sents.shape[0], 300), dtype=np.float)
	count = np.zeros((sents.shape[0], 1))

	for idx, sent in enumerate(sents):
		chars = [',', ':', '/', '(', ')', '?', '!', '.', '"', "$", '“', '”', '#', ';', '%']

		for c in chars:
			sent = sent.replace(c, ' ' + c + ' ')
	
		sent = sent.split(' ')
		for word in sent:
			if word != '':
				if word not in where:
					where[word] = []
				where[word].append(idx)
			

	print(where)

	line_count = 0

	glove_file = open(GLOVE_FILE)
	for line in glove_file:
		line = line[:-1].split(' ')
		word = line[0]
		vec = np.array(line[1:], dtype=np.float)
		words.add(word)
		if word in where:
			for idx in where[word]:
				result[idx] += vec
				count[idx][0] += 1
		line_count += 1
		if line_count % 10000 == 0:
			print ('line_count: ' + str(line_count))

	result /= count

	print(result)

	for word in where:
		if not word in words:
			print ('Not found word \'' + word + '\'')
	
	return result

def test_run():
	embeddings(np.array(['This is a very good sentence, my friend.',
						'Birdie lives in the United States.',
						'\"He is not a suspect anymore.\" John said.'])
