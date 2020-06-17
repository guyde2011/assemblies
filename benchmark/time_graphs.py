from brain.components import *
from brain.Connectome import Connectome
from brain.Connectome import NonLazySparseConnectome
from brain.Connectome import NonLazyConnectomeOriginal
from brain.Connectome import NonLazyConnectomeRandomMatrix

import matplotlib.pylab as plt
import os
import time
from pathlib import Path
import math


def create_gen(cls):
    """
    Create connectome generator.
    The generator runs a basic connectome run, yieding after the initialization,
    the first projects and 25 additional projects, allowing step-by-step exectution of the sections,
    which enables timing.
    :param cls: Connectome class
    :return: The described connectome gen
    """
    def gen(a1, a2, s1):
        inst = cls(areas=[a1, a2], stimuli=[s1], p=0.05)
        yield 'init_connectome'
        inst.project({s1: [a1]})
        inst.project({a1: [a2]})
        yield 'first_projects'
        for i in range(25):
            inst.project({a1: [a2]})
        yield 'many_projects'
    gen.__name__ = cls.__name__
    return gen


gens = [
        create_gen(Connectome),
        create_gen(NonLazyConnectomeOriginal),
        create_gen(NonLazySparseConnectome),
        create_gen(NonLazyConnectomeRandomMatrix),
        ]


def _timed(gen, a1, a2, s1, N=20):
    """ Time every call to the generator """
    d = {}
    for i in range(N):
        g = gen(a1, a2, s1)
        durations = {}
        while True:
            try:
                before = int(time.time() * 1000)
                section = next(g)
                after = int(time.time() * 1000)
                durations[section] = (after - before)
            except StopIteration:
                 d[i] = durations
                 break
    durations = {}
    for section in d[0]:
        durations[section] = sum([d[i][section] for i in range(N)]) / N
    return durations


def _sample_timing(gen):
    # dict(n: durations)
    d = {}
    print(f"Creating sample for {gen.__name__}")
    for i in range(1, 11):
        a1 = Area(n=(1000 * i), k=math.floor(math.sqrt(1000 * i)), beta=0.05)
        a2 = Area(n=(1000 * i), k=math.floor(math.sqrt(1000 * i)), beta=0.05)
        s1 = Stimulus(n=math.floor(math.sqrt(1000 * i)), beta=0.05)
        d[(2000 * i)] = _timed(gen, a1, a2, s1)
    return d


def _sample_by_section(gen):
    sample = _sample_timing(gen)

    sample_by_section = {}
    for section in ['init_connectome', 'first_projects', 'many_projects']:
        sample_by_section[section] = {key: sample[key][section] for key in sample}
    return sample_by_section


def get_sample(classes=None):
    """
    Get time sample for all classes, can be later plotted.

    :param classes: classes to sample, if not given samples all classes
    :return: the time sample
    """
    sample = {}
    sample_gens = gens if not classes else filter(lambda gen: gen.__name__ in classes, gens)
    for gen in sample_gens:
        sample[gen.__name__] = _sample_by_section(gen)
    return sample


def create_graph(sample, section, path):
    """
    Create and save graph from sample, for a given section.

    :param sample: sample returned by get_sample
    :param section: section in sample
                   (one of 'init_connectome', 'first_projects', 'many_projects')
                   the graph will plot the execution time of this section
    :param path: path to save graph to
    """
    legend = []
    for gen in sample:
        lists = sorted(sample[gen][section].items())
        x, y = zip(*lists)
        plt.plot(x, y, 'o-.')
        plt.title(section)
        plt.xlabel('area size (#neurons)')
        plt.ylabel('time (milliseconds)')
        name = gen if gen != 'Connectome' else 'NonLazyConnectomeNew'
        legend.append(name)
    plt.legend(legend, loc='upper left')
    plt.savefig(f"{path}/{section}.jpg", dpi=300)
    plt.clf()


def _get_directory(root):
    """ Gets new path. """
    index = 1
    while True:
        path = root / str(index)
        if path.exists():
            index += 1
            continue
        path.mkdir(parents=True)
        return str(path)


def graphit(name=None, classes=None):
    """
    Graph given classes, save to graphs/name.pdf

    :param name: Name of graph,
                 if not supplied will be saved to an automatically chosen path
    :param classes: Connectome classes to measure
    """
    sample = get_sample(classes)
    root = Path(os.path.abspath('.')) / 'graphs'
    if name:
        (root / name).mkdir(parents=True, exist_ok=True)
    path = str(root / name) if name else _get_directory(root)
    create_graph(sample, 'first_projects', path)
    create_graph(sample, 'many_projects', path)
    print(f"Saved graphs to {path}")
