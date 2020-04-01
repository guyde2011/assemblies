from brain import *
from non_lazy_brain import *
import numpy as np
# ____// NON LAZY TESTS //____


def test_area_in_brain():
    brain = NonLazyBrain(p=0)
    brain.add_area(name='a', n=3, k=1, beta=0.1)
    assert 'a' in brain.areas.keys()
    assert 'a' in brain.connectomes.keys()


def test_stimulus_in_brain():
    brain = NonLazyBrain(p=0)
    brain.add_stimulus(name='s', k=2)
    assert 's' in brain.stimuli.keys()


def test_init_connectomes_area():
    brain = NonLazyBrain(p=0)
    brain.add_area(name='a', n=3, k=1, beta=0.1)
    assert all([all([brain.connectomes['a']['a'][i][j] == 0 for i in range(3)]) for j in range(3)])
    assert brain.areas['a'].area_beta['a'] == 0.1
    brain = NonLazyBrain(p=1)
    brain.add_area(name='a', n=3, k=1, beta=0.1)
    assert all([all([brain.connectomes['a']['a'][i][j] == 1 for i in range(3)]) for j in range(3)])
    assert brain.areas['a'].area_beta['a'] == 0.1


def test_init_connectomes_stimulus():
    brain = NonLazyBrain(p=0)
    brain.add_area(name='a', n=3, k=1, beta=0.1)
    brain.add_stimulus(name='s', k=2)
    assert all([all([brain.stimuli_connectomes['s']['a'][i][j] == 0 for i in range(2)]) for j in range(3)])
    assert brain.areas['a'].stimulus_beta['s'] == 0.1
    brain = NonLazyBrain(p=1)
    brain.add_area(name='a', n=3, k=1, beta=0.1)
    brain.add_stimulus(name='s', k=2)
    assert all([all([brain.stimuli_connectomes['s']['a'][i][j] == 1 for i in range(2)]) for j in range(3)])
    assert brain.areas['a'].stimulus_beta['s'] == 0.1


def test_project_support_size():
    for p in [0, 0.3, 0.5, 1]:
        support_size = 0
        brain = NonLazyBrain(p)
        brain.add_area(name='a', n=3, k=1, beta=0.1)
        brain.add_stimulus(name='s', k=2)
        assert brain.areas['a'].support_size == 0
        for _ in range(3):
            brain.project(stim_to_area={'s': ['a']}, area_to_area={})
            support_size += brain.areas['a'].num_first_winners
            assert support_size == brain.areas['a'].support_size


def test_project_winners():
    brain = NonLazyBrain(p=0)
    brain.add_area(name='a', n=2, k=1, beta=0.1)
    brain.add_area(name='b', n=2, k=1, beta=0.1)
    brain.add_stimulus(name='s', k=1)
    brain.connectomes['a']['b'][0][0] = 1
    brain.connectomes['a']['b'][0][1] = 0
    brain.connectomes['a']['b'][1][0] = 0
    brain.connectomes['a']['b'][1][1] = 0
    brain.stimuli_connectomes['s']['a'][0][0] = 1
    brain.stimuli_connectomes['s']['a'][0][1] = 0
    brain.project({'s': ['a']}, {'a': ['b']})
    return brain.areas['a'].winners == [0]


def test_project_num_first_winners():
    brain = NonLazyBrain(p=0)
    brain.add_area(name='a', n=2, k=1, beta=0.1)
    brain.add_area(name='b', n=2, k=1, beta=0.1)
    brain.add_stimulus(name='s', k=1)
    brain.connectomes['a']['b'][0][0] = 1
    brain.connectomes['a']['b'][0][1] = 0
    brain.connectomes['a']['b'][1][0] = 0
    brain.connectomes['a']['b'][1][1] = 0
    brain.stimuli_connectomes['s']['a'][0][0] = 1
    brain.stimuli_connectomes['s']['a'][0][1] = 0
    brain.project({'s': ['a']}, {'a': ['b']})
    assert brain.areas['a'].num_first_winners == 1


def test_project_no_winners():
    brain = NonLazyBrain(p=0)
    brain.add_area(name='a', n=2, k=1, beta=0.1)
    brain.add_area(name='b', n=2, k=1, beta=0.1)
    brain.add_stimulus(name='s', k=1)
    brain.connectomes['a']['b'][0][0] = 1
    brain.connectomes['a']['b'][0][1] = 0
    brain.connectomes['a']['b'][1][0] = 0
    brain.connectomes['a']['b'][1][1] = 0
    brain.stimuli_connectomes['s']['a'][0][0] = 1
    brain.stimuli_connectomes['s']['a'][0][1] = 0
    brain.project({'s': ['a']}, {'a': ['b']})
    assert True


def test_project_connectomes():
    brain = NonLazyBrain(p=0)
    brain.add_area(name='a', n=2, k=1, beta=0.1)
    brain.add_area(name='b', n=2, k=1, beta=0.1)
    brain.add_stimulus(name='s', k=1)
    brain.connectomes['a']['b'][0][0] = 1
    brain.connectomes['a']['b'][0][1] = 0
    brain.connectomes['a']['b'][1][0] = 0
    brain.connectomes['a']['b'][1][1] = 0
    brain.stimuli_connectomes['s']['a'][0][0] = 1
    brain.stimuli_connectomes['s']['a'][0][1] = 0
    brain.project({'s': ['a']}, {})
    brain.project({}, {'a': ['b']})
    assert abs(brain.connectomes['a']['b'][0][0] - 1.1) < 0.0001


def test_project_different_n():
    brain = NonLazyBrain(p=0)
    brain.add_area(name='a', n=3, k=1, beta=0.1)
    brain.add_area(name='b', n=2, k=1, beta=0.1)
    brain.add_stimulus(name='s', k=1)
    brain.project({'s': ['a']}, {'a': ['b']})


def test_project_different_k():
    brain = NonLazyBrain(p=0)
    brain.add_area(name='a', n=3, k=1, beta=0.1)
    brain.add_area(name='b', n=3, k=2, beta=0.1)
    brain.add_stimulus(name='s', k=1)
    brain.project({'s': ['a']}, {'a': ['b']})
