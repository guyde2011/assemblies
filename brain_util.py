"""
Some utilities to make working with this library easier.
"""

import pickle

def sim_save(file_name, obj):
	"""
	Save obj to disc (could be Brain object, list of saved winners, etc) as file_name
	"""
	with open(file_name,'wb') as f:
		pickle.dump(obj, f)

def sim_load(file_name):
	"""
	Load object from file 'file_name'
	"""
	with open(file_name,'rb') as f:
		return pickle.load(f)

def overlap(a,b):
	"""
	Compute item overlap between two lists viewed as sets.
	"""
	return len(set(a) & set(b))

def get_overlaps(winners_list,base,percentage=False):
	"""
	Compute overlap of each list of winners in winners_list
	with respect to a specific winners set, namely winners_list[base]
	"""
	overlaps = []
	base_winners = winners_list[base]
	k = len(base_winners)
	for i in range(len(winners_list)):
		o = overlap(winners_list[i],base_winners)
		if percentage:
			overlaps.append(float(o)/float(k))
		else:
			overlaps.append(o)
	return overlaps

