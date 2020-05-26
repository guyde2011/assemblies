from .connectome import Connectome


class LazyConnectome(Connectome):
	def __init__(self, p: float, brain_parts=None, connections=None):
		super(self, LazyConnectome).__init__(brain_parts, connections)
		self.p = p
