![build status](https://travis-ci.org/BrainProjectTau/Brain.svg?branch=master)
[![coverage](https://img.shields.io/badge/coverage-404-lightgrey)](https://img.shields.io/badge/coverage-404-lightgrey)
[![codesize](https://img.shields.io/github/languages/code-size/Assemblies-Performance/assemblies)](https://img.shields.io/github/languages/code-size/Assemblies-Performance/assemblies)
[![laziness](https://img.shields.io/badge/laziness-0-brightgreen)](https://img.shields.io/badge/laziness-0-brightgreen)
[![performance](https://img.shields.io/badge/%D7%A9%D7%A0%D7%9E%D7%A8%D7%95%D7%A4%D7%A8%D7%A2%D7%A4-100%25-ff69b4)](https://img.shields.io/badge/%D7%A9%D7%A0%D7%9E%D7%A8%D7%95%D7%A4%D7%A8%D7%A2%D7%A4-100%25-ff69b4)
[![assemblies](https://img.shields.io/badge/assemblies-0-red)](https://img.shields.io/badge/assemblies-0-red)
[![guyde](https://img.shields.io/badge/guyde-100%25-9cf)](https://github.com/guyde2011)
[![badges](https://img.shields.io/badge/badges-118%25-ffcc99)](https://img.shields.io/badge/badges-118%25-ffcc99)
[![meta](https://img.shields.io/badge/meta-1000000000000000000000%25-80dfff)](https://img.shields.io/badge/meta-1000000000000000000000%25-80dfff)
[![bananas](https://img.shields.io/badge/bananas-0-ffdb4d)](https://www.youtube.com/watch?v=aKn0HddzuWM)
[![exbananas](https://img.shields.io/badge/exbananas-1-yellow)](https://www.youtube.com/watch?v=vnciwwsvNcc)

## Installation

1. Clone the repository and enter it:

    ```sh
    $ git clone git@github.com:BrainProjectTau/Brain.git
    ...
    $ cd Brain/
    ```

2. Run the installation script and activate the virtual environment:

    ```sh
    $ ./scripts/install.sh
    ...
    $ source .env/bin/activate
    [Brain] $ # you're good to go!
    ```

3. To check that everything is working as expected, run the tests:


    ```sh
    $ pytest tests/
    ...
    ```

## Usage

The `Brain` packages provides the following classes:
    
- `Area`

    This class represents an Area in the brain.

    ```pycon
    >>> from Brain import Area
    >>> area = Area(beta = 0.1, n = 10 ** 7, k = 10 ** 4)
    ```

- `Assembly`
    
    This class represents an Assembly in the brain.
    
- `Connectome`
    
    Sub-package which holds the structre of the brain.
    The sub-package defines the following classes:
    
    - `Connectome`
        Abstract class which defines the API which a general connectome should have.
        This class should be inhereted and implemented.
        
        ```pycon
        >>> from Connectome import Connectome
        >>> class LazyConnectome(Connectome):
        >>>     #implementation of a specific connectome
        >>>> connectome = LazyConnectome()
        >>> area = Area(beta = 0.1, n = 10 ** 7, k = 10 ** 4)
        >>> connectome.add_area(area)
        ```
    - `NonLazyRandomConnectome` 
        Already implemented Connectome which by decides it's edge by chance.
        This Connectome doesn't use any kind of laziness.
       
       ```pycon
        >>> from Connectome import NonLazyRandomConnectome
        >>>> connectome = NonLazyRandomConnectome()
        >>> area = Area(beta = 0.1, n = 10 ** 7, k = 10 ** 4)
        >>> connectome.add_area(area)
        ```
    - `To be continued`
        More ways to implement a connectome can be applied simply by inhereting from Connectome and implementing it's API.
    
- `Brain`

    This class represents a simulated brain, with it's connectome which holds the areas, stimuli, and all the synapse weights.

    ```pycon
    >>> from Brain import Brain, NonLazyRandomConnectome, Area
    >>> connectome = NonLazyRandomConnectome()
    >>> area = Area(beta = 0.1, n = 10 ** 7, k = 10 ** 4)
    >>> connectome.add_area(area)
    >>> brain = Brain(connectome)
    ```
