import multiprocessing
from typing import Callable, List

# Process count for multiprocessing
POOL_SIZE = 4


def process_map(f: Callable, args: List, packed: bool = False) -> List:
    """
    Maps an operation from conversions.py across multiple processes.
    :param f: The function to map, from conversions.py.
    :param args: The list of argument tuples to map over.
    :param packed: Whether the args list consists of packed argument tuples.
    :return: The list of outputs from the mapping of f over args.
    """
    with multiprocessing.Pool(POOL_SIZE) as p:
        if packed:
            return p.starmap(f, args)
        else:
            return p.map(f, args)
