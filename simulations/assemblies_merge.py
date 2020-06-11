import gc

from brain import Area, Stimulus, BrainRecipe, bake, NonLazyConnectome
from assemblies import Assembly
from utils.i_love_my_ram import protec_ram

import matplotlib.pyplot as plt

# Protect RAM from program using up all memory
# Allows program to use only half of free memory
protec_ram(0.75)

# Create graph for presenting the results
plt.title('Assemblies Merge')
plt.xlabel('t (Repeat Parameter)')
plt.ylabel('Overlap %')
# Number of samples per graph point
AVERAGING_SIZE = 5

# Size of Stimulus
STIMULUS_SIZE = 100
# Size of areas
AREA_SIZE = 1000

# Create basic brain model
stimulus = Stimulus(STIMULUS_SIZE)
area1 = Area(AREA_SIZE)
area2 = Area(AREA_SIZE)
area3 = Area(AREA_SIZE)
area4 = Area(AREA_SIZE)
assembly1 = Assembly([stimulus], area1)
assembly2 = Assembly([stimulus], area2)

# MERGE_STABILIZATIONS = (0, 1, 2, 3)
# REPEATS = (1, 10, 25, 50, 100, 250)
MERGE_STABILIZATIONS = (0, 1, 2, 3, 10, 25, 50)
REPEATS = (1, 10, 25)

for merge_stabilization in MERGE_STABILIZATIONS:
    recipe = BrainRecipe(area1, area2, area3, area4, stimulus, assembly1, assembly2)
    # Define assembly out of recipe,
    # that way merge can done manually!
    assembly3 = (assembly1 + assembly2) >> area3

    with recipe:
        # Manual merge process by interleaved projects
        for _ in range(merge_stabilization):
            assembly1 >> area3
            assembly2 >> area3

    # Dictionary for storing results
    overlap_per_repeat = {}
    for t in REPEATS:
        print(f"Beginning simulation with merge_stabilization={merge_stabilization}, t={t}:")

        # Averaging loop
        values = []
        for _ in range(AVERAGING_SIZE):
            # Create brain from recipe
            with bake(recipe, 0.1, NonLazyConnectome, repeat=t) as brain:
                def overlap(A, B):
                    assert len(A) == len(B)
                    return len(set(A).intersection(set(B))) / len(A)

                # Project assembly for the first time
                assembly3.project(area4, iterations=1)
                # Store winners
                first_winners = area4.winners
                # Project assembly for the second time
                assembly3.project(area4, iterations=1)
                # Store winners
                second_winners = area4.winners

                # Compute the overlap between first and second projection winners
                values.append(overlap(first_winners, second_winners) * 100)

            del brain
            gc.collect()

        # Compute average overlap
        overlap_value = sum(values) / len(values)
        overlap_per_repeat[t] = overlap_value

        print(f"\tOverlap: {overlap_value}%")

    # Add current observations to graph
    x, y = zip(*overlap_per_repeat.items())
    plt.plot(x, y, label=f"{merge_stabilization} Merge Stabilizations")

# Present and save graph
plt.legend()
plt.show()
