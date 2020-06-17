from brain.components import *
from brain.connectome.multithreaded_connectome import MultithreadedConnectome
from brain.connectome import Connectome

from . import profile


a1 = Area(n=10000, k=100, beta=0.05)
a2 = Area(n=10000, k=100, beta=0.05)
s1 = Stimulus(n=100, beta=0.05)

nlc = Connectome(areas=[a1, a2], stimuli=[s1], p=0.05)

nlc.project({s1: [a1]})
nlc.project({a1: [a2, a1]})
nlc.project({a2: [a1, a2]})


@profile
def time_normal():

    yield 'many_projects'
    for i in range(100):
        nlc.project({a1: [a2]})
