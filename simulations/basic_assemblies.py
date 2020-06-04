from brain import Area, Stimulus, BrainRecipe, bake, NonLazyConnectome
from assemblies import Assembly

stim = Stimulus(100, 0.05)
area1 = Area(100)
area2 = Area(200)
area3 = Area(300)
assembly1 = Assembly([stim], area1, 100, t=1)
assembly2 = Assembly([stim], area2, 100, t=1)
recipe = BrainRecipe(area1, area2, stim, assembly1, assembly2)

with recipe:
    assembly3 = (assembly1 + assembly2).merge(area3)
    assembly4 = assembly3 >> area1
    assembly1.associate(assembly3)

print("Beginning simulation")

with bake(recipe, 0.1, NonLazyConnectome) as brain:
    assembly3 >> area1
    assembly4 >> area1
    assembly3 = (assembly1 + assembly2).merge(area3)
    assembly1.associate(assembly3)
