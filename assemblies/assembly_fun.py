from brain import *
from typing import Iterable, Union, Optional
from copy import deepcopy


Projectable = Union['Assembly', 'NamedStimulus']


class NamedStimulus(object):
    def __init__(self, name, stimulus):
        self.name = name
        self.stimulus = stimulus

    def __repr__(self) -> str:
        return f"Stimulus({self.name})"


class Assembly(object):
    def __init__(self, parents: Iterable[Projectable], area_name: str, name: str):
        self.parents: List[Projectable] = list(parents)
        self.area_name: str = area_name
        self.name: str = name

    def __repr__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash(self.name)

    @staticmethod
    def fire_many(brain: Brain, projectables: Iterable[Projectable], area_name: str):
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

    def project(self, brain: Brain, area_name: str) -> 'Assembly':
        projected_assembly: Assembly = Assembly([self], area_name, f"project({self.name}, {area_name})")
        Assembly.fire_many(brain, [self], area_name)
        return projected_assembly

    def reciprocal_project(self, brain: Brain, area_name: str) -> 'Assembly':
        projected_assembly: Assembly = self.project(brain, area_name)
        Assembly.fire_many(brain, [projected_assembly], self.area_name)
        return projected_assembly

    @staticmethod
    def merge(brain: Brain, assembly1: 'Assembly', assembly2: 'Assembly', area_name: str) -> 'Assembly':
        merged_assembly: Assembly = Assembly([assembly1, assembly2], area_name,
                                             f"merge({assembly1.name}, {assembly2.name}, {area_name})")
        # TODO: Decide one of the two - Consult Edo Arad
        Assembly.fire_many(brain, [assembly1, assembly2], area_name)
        # OR: Assembly.fire_many(assembly1.brain, assembly1.parents + assembly2.parents)
        return merged_assembly

    @staticmethod
    def associate(brain: Brain, assembly1: 'Assembly', assembly2: 'Assembly'):
        assert(assembly1.area_name == assembly2.area_name, "Areas are not the same")
        Assembly.merge(brain, assembly1, assembly2, assembly1.area_name)

    @staticmethod
    def get_reads(brain: Brain, possible_assemblies: Iterable['Assembly'], area_name: str) -> Dict['Assembly', float]:
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
        return max(Assembly.get_reads(brain, possible_assemblies, area_name))
