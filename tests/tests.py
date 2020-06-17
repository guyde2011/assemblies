
from brain import *
from brain.connectome import *


def test_area_in_brain():
    """test for non lazy connectome"""
    brain = Connectome(p=0, initialize=True)
    a = Area(n=3, k=1, beta=0.1)
    brain.add_area(a)
    assert a in brain.areas


def test_stimulus_in_brain():
    """test for non lazy brain"""
    brain = Connectome(p=0, initialize=True)
    s = Stimulus(n=3, beta=0.1)
    brain.add_stimulus(s)
    assert s in brain.stimuli


def test_init_connectomes_area():
    """test for non lazy brain"""
    brain = Connectome(p=0, initialize=True)
    a = Area(n=3, k=1, beta=0.1)
    brain.add_area(a)
    assert all([all([brain.connections[a, a][i, j] == 0 for i in range(3)]) for j in range(3)])
    assert brain.connections[a, a].beta == 0.1
    brain = Connectome(p=1)
    a = Area(n=3, k=1, beta=0.1)
    brain.add_area(a)
    assert all([all([brain.connections[a, a][i, j] == 1 for i in range(3)]) for j in range(3)])
    assert brain.connections[a, a].beta == 0.1


def test_init_connectomes_stimulus():
    """test for non lazy brain"""
    brain = Connectome(p=0, initialize=True)
    a = Area(n=3, k=1, beta=0.1)
    brain.add_area(a)
    s = Stimulus(n=2, beta=0.1)
    brain.add_stimulus(s)
    assert all([all([brain.connections[s, a][i, j] == 0 for i in range(2)]) for j in range(3)])
    assert brain.connections[s, a].beta == 0.1
    brain = Connectome(p=1)
    a = Area(n=3, k=1, beta=0.1)
    brain.add_area(a)
    s = Stimulus(n=2, beta=0.1)
    brain.add_stimulus(s)
    assert all([all([brain.connections[s, a][i, j] == 1 for i in range(2)]) for j in range(3)])
    assert brain.connections[s, a].beta == 0.1


def test_project_winners():
    brain = Connectome(p=0, initialize=True)
    a = Area(n=2, k=1, beta=0.1)
    brain.add_area(a)
    b = Area(n=2, k=1, beta=0.1)
    brain.add_area(b)
    s = Stimulus(n=1, beta=0.1)
    brain.add_stimulus(s)
    brain.connections[a, b][0, 0] = 1
    brain.connections[a, b][0, 1] = 0
    brain.connections[a, b][1, 0] = 0
    brain.connections[a, b][1, 1] = 0
    brain.connections[s, a][0, 0] = 1
    brain.connections[s, a][0, 1] = 0
    brain.project({s: [a], a: [b]})
    assert brain.winners[a] == [0]


# Supposed to test whether or not the code crashes with no winners
def test_project_no_winners():
    brain = Connectome(p=0, initialize=True)
    a = Area(n=2, k=1, beta=0.1)
    brain.add_area(a)
    b = Area(n=2, k=1, beta=0.1)
    brain.add_area(b)
    s = Stimulus(n=1, beta=0.1)
    brain.add_stimulus(s)
    brain.connections[a, b][0, 0] = 1
    brain.connections[a, b][0, 1] = 0
    brain.connections[a, b][1, 0] = 0
    brain.connections[a, b][1, 1] = 0
    brain.connections[s, a][0, 0] = 1
    brain.connections[s, a][0, 1] = 0
    brain.project({s: [a], a: [b]})


def test_project_connectomes():
    brain = Connectome(p=0, initialize=True)
    a = Area(n=2, k=1, beta=0.1)
    b = Area(n=2, k=1, beta=0.1)
    brain.add_area(a)
    brain.add_area(b)
    s = Stimulus(n=1, beta=0.1)
    brain.add_stimulus(s)
    brain.connections[a, b][0, 0] = 1
    brain.connections[a, b][0, 1] = 0
    brain.connections[a, b][1, 0] = 0
    brain.connections[a, b][1, 1] = 0
    brain.connections[s, a][0, 0] = 1
    brain.connections[s, a][0, 1] = 0
    brain.project({s: [a]})

    print(brain.connections[a, b])
    brain.project({a: [b]})

    print(brain.connections[a, b])
    assert abs(brain.connections[a, b][0, 0] - 1.1) < 0.0001


# Supposed to test whether or not the code crashes with different n's
def test_project_different_n():
    brain = Connectome(p=0.5, initialize=True)
    a = Area(n=3, k=1, beta=0.1)
    brain.add_area(a)
    b = Area(n=2, k=1, beta=0.1)
    brain.add_area(b)
    s = Stimulus(n=2, beta=0.1)
    brain.add_stimulus(s)
    brain.project({s: [a], a: [b]})


# Supposed to test whether or not the code crashes with different k's
def test_project_different_k():
    brain = Connectome(p=0.5, initialize=True)
    a = Area(n=30, k=5, beta=0.1)
    b = Area(n=30, k=3, beta=0.1)
    brain.add_area(a)
    brain.add_area(b)
    s = Stimulus(n=3, beta=0.1)
    brain.add_stimulus(s)
    brain.project({s: [a], a: [b]})


'''
def test_project_area_winners():
    """test for lazy brain"""
    brain = LazyBrain(p=0.01)
    brain.add_area(name='a', n=10000, k=100, beta=0.05)
    brain.add_area(name='b', n=10000, k=100, beta=0.05)
    brain.add_stimulus(name='s', k=100)

    brain.project({'s': ['a']}, {})
    brain.project({'s': ['a']}, {'a': ['b']})

    a = brain.areas['a']
    b = brain.areas['b']
    assert len(a.winners) == a.k
    assert len(set(a.winners)) == a.k
    assert len(b.winners) == b.k
    assert len(set(b.winners)) == b.k


def test_project_num_first_winners_lazy():
    """test for lazy brain"""
    brain = LazyBrain(p=0.5)
    brain.add_area(name='a', n=50, k=7, beta=0.1)
    brain.add_area(name='b', n=50, k=7, beta=0.1)
    brain.add_stimulus(name='s', k=7)
    brain.project({'s': ['a']}, {})
    assert brain.areas['a'].num_first_winners == 7
'''
