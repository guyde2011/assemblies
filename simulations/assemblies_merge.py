import gc
import time

from brain import Area, Stimulus, BrainRecipe, bake
from brain import Connectome
from assemblies import Assembly

import matplotlib.pyplot as plt

"""
Parameter Selection
"""
# Number of samples per graph point
AVERAGING_SIZE = 25
# Size of Stimulus
STIMULUS_SIZE = 100
# Size of areas
AREA_SIZE = 1000

# MERGE_STABILIZATIONS = (0, 1, 2, 3)
# REPEATS = (1, 10, 25, 50, 100, 250)
# MERGE_STABILIZATIONS = (0, 1, 2, 3, 10, 25)
# REPEATS = (1, 5, 10, 25)
TESTS = (
    (1, (1, 3, 5, 10, 25, 100, 250)),
    (3, (1, 3, 5, 10, 25, 100)),
    (5, (1, 3, 5, 10, 25, 100)),
    (10, (1, 3, 5, 10, 25, 100)),
    (25, (1, 3, 5, 10, 25, 50)),
    (50, (1, 3, 5, 10, 25, 50)),
    (100, (1, 3, 5, 10, 25)),
)


# Create graph for presenting the results
plt.title('Assemblies Merge')
plt.xlabel('t (Repeat Parameter)')
plt.ylabel('Overlap %')

# Create basic brain model
stimulus = Stimulus(STIMULUS_SIZE)
area1 = Area(AREA_SIZE)
area2 = Area(AREA_SIZE)
area3 = Area(AREA_SIZE)
area4 = Area(AREA_SIZE)
assembly1 = Assembly([stimulus], area1)
assembly2 = Assembly([stimulus], area2)

begin_time = time.time()
for merge_stabilization, repeats in TESTS:
    recipe = BrainRecipe(area1, area2, area3, area4, stimulus, assembly1, assembly2)
    # Define assembly out of recipe,
    # that way merge can done manually!
    assembly3 = (assembly1 + assembly2) >> area3

    with recipe:
        # Manual merge process by interleaved projects
        for _ in range(merge_stabilization):
            assembly1 >> area3
            assembly2 >> area3

        # Stabilize connections between Assembly 3 and Area 4
        assembly3.project(area4)

    # Dictionary for storing results
    overlap_per_repeat = {}
    for t in repeats:
        print(f"Beginning simulation with merge_stabilization={merge_stabilization}, t={t}:")

        # Averaging loop
        values = []
        for _ in range(AVERAGING_SIZE):
            # Create brain from recipe
            with bake(recipe, 0.1, Connectome, train_repeat=t, effective_repeat=3) as brain:
                def overlap(A, B):
                    assert len(A) == len(B)
                    return len(set(A).intersection(set(B))) / len(A)

                # Project assembly for the first time
                assembly3 >> area4
                # Store winners
                first_winners = area4.winners
                # Project assembly for the second time
                assembly3 >> area4
                print(assembly3.identify(preserve_brain=True))
                print(Assembly.read(area4, brain=brain))
                # Store winners
                second_winners = area4.winners

                # Compute the overlap between first and second projection winners
                values.append(overlap(first_winners, second_winners) * 100)

            gc.collect()

        # Compute average overlap
        overlap_value = sum(values) / len(values)
        overlap_per_repeat[t] = overlap_value

        print(f"\tOverlap: {overlap_value}%")

    # Add current observations to graph
    x, y = zip(*overlap_per_repeat.items())
    plt.plot(x, y, label=f"{merge_stabilization} Merge Stabilizations")

end_time = time.time()
print(f"Simulation took {end_time - begin_time}ms")

# Present and save graph
plt.legend()
plt.show()
