from brain import *
from typing import Iterable, Union, Optional
from copy import deepcopy


Projectable = Union['Assembly', 'NamedStimulus']


class NamedStimulus(object):    # hi
    """ 
    acts as a buffer between our implementation and brain.py, as the relevant
    functions there use naming to differentiate between areas
    """
    def __init__(self, name, stimulus):
        self.name = name
        self.stimulus = stimulus

    def __repr__(self) -> str:
        return f"Stimulus({self.name})"


class Assembly(object):
    """
    the main assembly object. according to our implementation, the main data of an assembly
    is his parents. we also keep a name for simulation puposes.
    """
    def __init__(self, parents: Iterable[Projectable], area_name: str, name: str, t: int = 1 ):
        self.parents: List[Projectable] = list(parents)
        self.area_name: str = area_name
        self.name: str = name

        """
        set default repeat amount (how many times to repeat each function) for
        this assembly object.
        """
        self.t = t

    def __repr__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)


    def repeat_t_times(self, func):
        """
        decorator function for repeating functions with a generic t
        t can be either the object default (passed during init),
        or a custom int for the specific function.
        """
        def repeater(*args, **kwargs):
            if "t" in kwargs: t = kwargs["t"]
            else: t = self.t
            for _ in range(t-1):
                func(*args)
            return func(*args)
        return repeater


    @staticmethod
    def fire_many(brain: Brain, projectables: Iterable[Projectable], area_name: str):
        """
        params:
        -the relevant brain object, as project is dependent on the brain instance.
        -a list of object which are projectable (Stimuli, Areas...) which will be projected
        -the target area's name.
        
        This function works by creating a "Parent tree", (Which is actually a directed acyclic graph) first, 
        and then by going from the top layer, which will consist of stimuli, and traversing down the tree 
        while firing each layer to areas its relevant descendant inhibit.
        
        For example, "firing" an assembly, we will first climb up its parent tree (Assemblies of course may have
        multiple parents, such as results of merge. Then we iterate over the resulting list in reverse, while firing
        each layer to the relevant areas, which are saved in a dictionary format:
        The thing we will project: areas to project it to
        """
        
        layers: List[Dict[Projectable, List[str]]] = [{stuff: [area_name] for stuff in projectables}]
        while any(isinstance(ass, Assembly) for ass in layers[-1]):
            prev_layer: Iterable[Assembly] = (ass for ass in layers[-1].keys() if not isinstance(ass, NamedStimulus))
            current_layer: Dict[Projectable, List[str]] = {}
            for ass in prev_layer:
                for parent in ass.parents:
                    current_layer[parent] = current_layer.get(ass, []) + [ass.area_name]

            layers.append(current_layer)

        layers = layers[::-1]
        for layer in layers:
            stimuli_mappings: Dict[str, List[str]] = {stim.name: areas
                                                          for stim, areas in
                                                          layer.items() if isinstance(stim, NamedStimulus)}

            area_mappings: Dict[str, List[str]] = {}
            for ass, areas in filter(lambda pair: isinstance(pair[0], Assembly), layer.items()):
                area_mappings[ass.area_name] = area_mappings.get(ass.area_name, []) + areas

            brain.project(stimuli_mappings, area_mappings)

    @repeat_t_times
    def project(self, brain: Brain, area_name: str) -> 'Assembly':
        """
        a simple case of project many, with only one projectable parameter
        """
        projected_assembly: Assembly = Assembly([self], area_name, f"project({self.name}, {area_name})")
        Assembly.fire_many(brain, [self], area_name)
        return projected_assembly

    @repeat_t_times
    def reciprocal_project(self, brain: Brain, area_name: str) -> 'Assembly':
        projected_assembly: Assembly = self.project(brain, area_name)
        Assembly.fire_many(brain, [projected_assembly], self.area_name)
        return projected_assembly

    @staticmethod
    @repeat_t_times
    def merge(brain: Brain, assembly1: 'Assembly', assembly2: 'Assembly', area_name: str) -> 'Assembly':
        """
        we create an "artificial" new assembly with x, y as parents, and then project_many
        to its area. this will create the effect of projecting stimultaneously, as described in the paper.
        """
        merged_assembly: Assembly = Assembly([assembly1, assembly2], area_name,
                                             f"merge({assembly1.name}, {assembly2.name}, {area_name})")
        # TODO: Decide one of the two - Consult Edo Arad
        Assembly.fire_many(brain, [assembly1, assembly2], area_name)
        # OR: Assembly.fire_many(assembly1.brain, assembly1.parents + assembly2.parents)
        return merged_assembly


    @staticmethod
    @repeat_t_times
    def associate(brain: Brain, assembly1: 'Assembly', assembly2: 'Assembly'):
        assert(assembly1.area_name == assembly2.area_name, "Areas are not the same")
        Assembly.merge(brain, assembly1, assembly2, assembly1.area_name)

    @staticmethod
    def get_reads(brain: Brain, possible_assemblies: Iterable['Assembly'], area_name: str) -> Dict['Assembly', float]:
        """
        simulate the calculus up to a certain point on a copy of the original brain, and return the correlation
        of the parameters of the neurons after the effects of project with those of the original neurons.
        the purpose is to check how quickly an assembly stabilizes.
        ranks the list of assemblies with their correlations after the simulation.
        """
        original_area: Area = brain.areas[area_name]
        brain_copy: Brain = deepcopy(brain)
        brain_copy.disinhibit()  # TODO: Implement (Shani's & Adi Dinerstein's groups)

        assembly_reads: Dict['Assembly', float] = {}

        for ass in possible_assemblies:
            if ass.area_name != area_name:
                continue

            Assembly.fire_many(brain_copy, ass.parents, area_name)
            simulated_area = brain_copy.areas[area_name]
            simulated_ass_neurons = simulated_area.winners
            original_neurons = original_area.winners
            # TODO: Review scoring method, because assembly may "rotate"
            assembly_reads[ass] = len(set(simulated_ass_neurons).intersection(original_neurons)) / len(original_neurons)

        return assembly_reads

    @staticmethod
    def read(brain: Brain, possible_assemblies: Iterable['Assembly'], area_name: str) -> 'Assembly':
        """
        simply return the most "stabilized" assembly, meaning the one with highest correlation.
        """
        return max(Assembly.get_reads(brain, possible_assemblies, area_name))

