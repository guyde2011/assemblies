import math
from typing import List


class BrainPart:
	def __init__(self, part_type: str, *args, **kwargs):
		init = {'Area': self._init_area, 'Stimulus': self._init_stimulus, 'OutputArea': self._init_area}
		self.part_type = part_type
		init[self.part_type](*args, **kwargs)

	def _init_area(self, n: int, k: int, beta: float = 0.01):
		self.beta: float = beta
		self.n: int = n
		self.k: int = k
		self.support: List[int] = [0] * self.n
		self.winners: List[int] = []
		self.new_winners: List[int] = []

		if k < 0:
			self.k = math.sqrt(n)

	# Beta may be static
	def _init_stimulus(self, n: int, beta: float):
		self.n = n
		self.beta = beta

	@classmethod
	def create_area(cls, n: int, k: int, beta: float = 0.01):
		cls('Area', n, k, beta)

	@classmethod
	def create_stimulus(cls, n: int, beta: float):
		cls('Stimulus', n, beta)

	@classmethod
	def create_outputarea(cls, n: int, beta: float):
		cls('OutputArea', n, beta)

	def __repr__(self):
		attrs = []
		for attr in [n, k, beta]:
			if hasattr(self, attr):
				attrs.append(attr)
		return f"{self.part_type}(" + ','.join([str(self.attr) for attr in attrs]) + ")"


class Connection:
	def __init__(self, source: BrainPart, dest: BrainPart, synapses=None):
		self.source = source
		self.dest = dest
		self.synapses = synapses

	@property
	def beta(self):
		if self.source.part_type == 'Stimulus':
			return self.dest.beta
		return self.source.beta

	def __getitem__(self, key):
		return self.synapses[key]

	def __setitem__(self, key, value):
		self.synapses[key] = value

	def __repr__(self):
		return f"Connection({self.synapses!r})"


__all__ = ['BrainPart', 'Connection']
