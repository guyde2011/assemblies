
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
        for ass in layer:
            if isinstance(ass, Assembly):
                ass.update_support(brain.areas[ass.area_name].winners)


def read_recursive(assembly):
    Assembly.fire_many(assembly.brain, [assembly.parents], assembly.area_name)
    return assembly.area.winners


read_recurive.name = 'recursive'