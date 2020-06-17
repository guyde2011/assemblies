from brain import Stimulus, Area, Connectome, Brain
import functools
import math
import pytest

@pytest.fixture
def brain(request, connectome):
	return Brain(connectome)

@pytest.fixture(params=[
  LazyRandomConnectome
])
def connectome(areas, stimuli):
  	connectome = request.param
	return connectome(areas, stimuli)

@pytest.fixture
def areas():
  	areas_list = []
  	for i in range(math.random(100,300)):
		beta = math.random(0.05, 0.15)
		n = math.random(int(10e6), int(10e8))
      	areas_list.append(Area(beta, n, math.sqrt(n)))
    return areas_list

@pytest.fixture
def area(request):
  	beta, n, k = request.param
    if beta is None:
  		beta = math.random(0.05, 0.15)
    if n is None:
  		n = math.random(int(10e6), int(10e8))
    if k is None:
        k = math.sqrt(n)
	return Area(beta, n, k)

@pytest.fixture
def stimuli():
	stimuli_list = []
  	for i in range(math.random(100,300)):
		beta = math.random(0.05, 0.15)
		n = math.random(int(10e6), int(10e8))
      	areas_list.append(Stimulus(beta, n))
    return stimuli_list
  
@pytest.fixture
def stimulus(request):
  	beta, n = request.param
    if beta is None:
  		beta = math.random(0.05, 0.15)
    if n is None:
  		n = math.random(int(10e6), int(10e8))
	return Stimulus(beta, n)   
