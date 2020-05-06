from typing import Union

class Stimulus:
	# Beta may be static
	def __init__(self, beta, n):
		self.n = n
		self.beta = beta
		# need to add specific k neurons?
		pass

	def __hash__(self):
		return hash(self)


class Area:
	def __init__(self, beta, n, k=None):
		self.beta = beta
		self.n = n
		self.k = k

		if k is None:
			self.k = math.sqrt(n)

	def __hash__(self):
		return hash(self)


BrainPart = Union[Area, Stimulus]

