import numpy as np


def fixed_mutation(individual, search_space, proba):
    """mutate(individual, proba)

        Mutate a given individual, using Searchspace neighborhood.
        Parameters
        ----------
        search_space : Search Space object, with a "labels" attribute and a "get_neighbor" method
        individual : list[{int, float, str}]
            Individual to mutate, in the mixed format.

        proba : float
            Probability to mutate a gene.

        Returns
        -------
        individual : list[{int, float, str}]
            Mutated individual

        """
    assert isinstance(individual, list)
    # For each dimension of a solution draw a probability to be muted
    for index, label in enumerate(search_space.labels):
        t = np.random.random()
        if t < proba:
            # Get a neighbor of the selected attribute
            individual[0][index] = search_space.get_neighbor(
                point=individual[0], attribute=label
            )
    return individual
