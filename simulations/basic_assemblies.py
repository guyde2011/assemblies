from brain import Area, Stimulus, BrainRecipe, bake, NonLazyConnectome
from assemblies import Assembly

STIMULUS_SIZE = 1000
AREA_SIZE = 1000

stimulus = Stimulus(STIMULUS_SIZE, 0.05)
area1 = Area(AREA_SIZE)
area2 = Area(AREA_SIZE)
area3 = Area(AREA_SIZE)
area4 = Area(AREA_SIZE)
assembly1 = Assembly([stimulus], area1)
assembly2 = Assembly([stimulus], area2)
recipe = BrainRecipe(area1, area2, area3, area4, stimulus, assembly1, assembly2)

with recipe:
    assembly3 = (assembly1 + assembly2) >> area3

print("Beginning simulation")

with bake(recipe, 0.1, NonLazyConnectome, t=10 ** 4) as brain:
    def overlap(A, B):
        assert len(A) == len(B)
        return len(set(A).intersection(set(B))) / len(A)

    assembly3 >> area4
    first_winners = area4.winners
    # print("Winners: " + str(sorted(first_winners)))
    assembly3 >> area4
    second_winners = area4.winners
    # print("Winners: " + str(sorted(second_winners)))
    print(f"Overlap: {overlap(first_winners, second_winners) * 100}%")

    assembly3 = (assembly1 + assembly2).merge(area3)
    assembly1.associate(assembly3)
