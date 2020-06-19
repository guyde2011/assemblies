from brain import Brain
import logging
from assemblies.assembly_fun import Assembly, NamedStimulus
from typing import List

"""
DEPRECATED
view simulations/basic_assemblies.py instead
"""

def assemblies_simulation(n=1000000, k=1000, p=0.1, beta=0.05, t=100) -> Brain:
    logging.basicConfig(level=logging.INFO)
    brain: Brain = Brain(p)
    brain.add_stimulus("stimulus1", k)
    brain.add_stimulus("stimulus2", k)
    area_names: List[str] = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P"]
    for name in area_names:
        brain.add_area(name, n, k, beta)

    for _ in range(t):
        brain.project({"stimulus1": [area_names[0]]}, {})
        brain.project({"stimulus2": [area_names[1]]}, {})

    assemblies: List[Assembly] = [
        Assembly([NamedStimulus("stimulus1", brain.stimuli["stimulus1"])], area_names[0], "assembly1"),
        Assembly([NamedStimulus("stimulus2", brain.stimuli["stimulus2"])], area_names[1], "assembly2")
    ]

    assemblies.append(assemblies[0].project(brain, area_names[2]))
    assemblies.append(assemblies[1].project(brain, area_names[3]))

    for _ in range(t):
        assemblies[0].project(brain, area_names[2])
        assemblies[1].project(brain, area_names[3])

    winners_after_stabilization: set = set()
    for _ in range(t // 10):
        assemblies[0].project(brain, area_names[2])
        assemblies[1].project(brain, area_names[3])
        winners_after_stabilization = winners_after_stabilization.union(brain.areas[area_names[2]].winners)

    print(f"Assembly winners has stabilized to a set of size {len(winners_after_stabilization) / k} * k")

    assemblies.append(Assembly.merge(brain, assemblies[0], assemblies[1], area_names[4]))
    for _ in range(t):
        Assembly.merge(brain, assemblies[0], assemblies[1], area_names[4])

    try:
        print(f"Assembly in area of merged area is {Assembly.identify(brain, assemblies, area_names[4]).name}")
    except Exception:
        print("Read is unimplemented!")

    for _ in range(t):
        brain.project({"stimulus2": [area_names[0]]}, {})
    assemblies.append(Assembly([NamedStimulus("stimulus2", brain.stimuli["stimulus2"])], area_names[0], "assembly3"))

    Assembly.associate(brain, assemblies[0], assemblies[5])
    for _ in range(t):
        Assembly.associate(brain, assemblies[0], assemblies[5])

    try:
        print(f"Reads in area {area_names[0]} are {Assembly.get_reads(brain, assemblies, area_names[0])}")
    except Exception:
        print("Read is unimplemented!")

    return brain


assemblies_simulation()
