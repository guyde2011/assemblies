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

assembly3 = (assembly1 + assembly2) >> area3

with recipe:
    # assembly4 = assembly3 >> area1
    # assembly1.associate(assembly3)
    pass

print("Beginning simulation")

with bake(recipe, 0.1, NonLazyConnectome, t=1) as brain:
    assembly3 >> area4
    print("Winners: " + str(sorted(area4.winners)))
    assembly3 >> area4
    print("Winners: " + str(sorted(area4.winners)))

    assembly3 = (assembly1 + assembly2).merge(area3)
    assembly1.associate(assembly3)
