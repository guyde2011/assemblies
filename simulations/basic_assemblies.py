from brain import Area, Stimulus, BrainRecipe, bake, NonLazyConnectome
from assemblies import Assembly

stim = Stimulus(100, 0.05)
area1 = Area(100)
area2 = Area(200)
area3 = Area(300)
assembly1 = Assembly([stim], area1)
assembly2 = Assembly([stim], area2)
recipe = BrainRecipe(area1, area2, stim, assembly1, assembly2)

assembly4 = Assembly([assembly1,assembly2],area1)
assembly3 = Assembly([assembly2,assembly1],area1)
print(assembly4 == assembly3)

with recipe:
    assembly3 = (assembly1 + assembly2).merge(area3)
    assembly4 = assembly3 >> area1
    assembly1.associate(assembly3)

print("Beginning simulation")

with bake(recipe, 0.1, NonLazyConnectome, 100) as brain:
    print("t is " + str(brain.t))
    assembly3 >> area1
    assembly4 >> area1
    assembly3 = (assembly1 + assembly2).merge(area3)
    assembly1.associate(assembly3)
