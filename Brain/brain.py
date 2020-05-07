from typing import Dict, List

from .Connectome import BrainPart, Connectome


# Library Ext team:
class Assembly:
    pass


class Brain:
    T = 10
    """
    Represents a simulated brain, with it's connectome which holds the areas, stimuli, and all the synapse weights.
    The brain updates by selecting a subgraph of stimuli and areas, and activating only those connections.
    The brain object works with a general connectome, which export an unified api for how the connections between the
    parts of the brain should be used. In case of need, one should extend the Connectome API as he would like to make
    the implementation of the brain easier/better. Note that the brain implementation shouldn't depends on the
    underlying implementation of the connectome.
    
    Attributes:
        connectome: The full connectome of the brain, hold all the connections between the brain parts.
        active_connectome: The current active subconnectome of the brain. Gives a nice way of supporting inhibit, disinhibit.   
	
    """

    def __init__(self, connectome: Connectome):
        self.connectome: Connectome = connectome
        self.active_connectome: Dict[BrainPart, List[BrainPart]]
	
	def next_round(self, active_connections :Dict[BrainPart, BrainPart]):
		return self.connectome.next_round(active_connections)
	
	def add_brain_part(self, brain_part :BrainPart):
		return self.connectome.add_brain_part(brain_part)

	next_round.__doc__ = Connectome.next_round.__doc__
	add_brain_part.__doc__ = Connectome.add_brain_part.__doc__

    # Library Ext for research:
    def project(self, x: Assembly, area: Area) -> Assembly:
        pass

    def reciprocal_project(self, x: Assembly, area: Area) -> Assembly:
        pass

    def association(self, x: Assembly, y: Assembly):
        pass

    def merge(self, x: Assembly, y: Assembly, area: Area) -> Assembly:
	    pass
