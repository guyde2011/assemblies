from assemblies.assembly_fun import Projectable, Brain, Assembly, Stimulus, Area
from typing import Iterable, List, Dict


class ReadRecursive:

    name = 'default'

    @staticmethod
    def fire(mapping: Dict[Projectable, List[Projectable]], *, brain: Brain = None):
        """
        Fire an assembly
        TODO: Tomer, this is an helper function for you right? If so, move to where relevant
        TODO: Karidi, make sure brain binding here makes sense
        :param brain:
        :param mapping:
        :return:
        """
        # TODO: Ido & Eyal, this is supposed to be mapping into BrainPart not str, update if needed (?)
        # I updated the signature
        for p in mapping:
            for t in mapping[p]:
                # TODO: Ido & Eyal, handle the case the projectable is an assembly (?)
                brain.inhibit(p, t)
        brain.next_round()
        for p in mapping:
            for t in mapping[p]:
                brain.disinhibit(p, t)

    @staticmethod
    def fire_many(brain: Brain, projectables: Iterable[Projectable], area: Area):
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

        layers: List[Dict[Projectable, List[Area]]] = [{projectable: [area] for projectable in projectables}]
        while any(isinstance(projectable, Assembly) for projectable in layers[-1]):
            prev_layer: Iterable[Assembly] = (ass for ass in layers[-1].keys() if not isinstance(ass, Stimulus))
            current_layer: Dict[Projectable, List[Area]] = {}
            for ass in prev_layer:
                for parent in ass.parents:
                    current_layer[parent] = current_layer.get(ass, []) + [ass.area]

            layers.append(current_layer)

        layers = layers[::-1]
        for layer in layers:
            stimuli_mappings: Dict[Stimulus, List[Area]] = {stim: areas
                                                            for stim, areas in
                                                            layer.items() if isinstance(stim, Stimulus)}

            assembly_mapping: Dict[Area, List[Area]] = {}
            for ass, areas in filter(lambda t: (lambda assembly, _: isinstance(assembly, Assembly))(*t), layer.items()):
                assembly_mapping[ass.area] = assembly_mapping.get(ass.area, []) + areas

            mapping = {**stimuli_mappings, **assembly_mapping}
            ReadRecursive.fire(mapping, brain=brain)

    @staticmethod
    def read(assembly: Assembly, *, brain: Brain):
        ReadRecursive.fire_many(brain, assembly.parents, assembly.area)
        return brain.winners[assembly.area]
