from collections.abc import Callable, Iterable
from functools import partial
import multiprocessing as mp
from typing import TypeVar


T = TypeVar('T')
U = TypeVar('U')


def parallel_map(
    func: Callable[[T], U],
    values: Iterable[T],
    *,
    num_threads: int = 1,
    chunk_size: int = 1,
    ordered: bool = True,
    progress: bool = False,
) -> Iterable[U]:
    """Maps a function onto an iterable of values.
    Returns a lazy iterator producing results.
    If num_threads > 1, processes the results in parallel."""
    if num_threads == 1:
        # no parallelism, so just use normal map
        mapped: Iterable[U] = map(func, values)
    else:
        pool = mp.Pool(num_threads)
        mapper = partial(
            pool.imap if ordered else pool.imap_unordered,
            chunksize=chunk_size,
        )
        mapped = mapper(func, values)
    if progress:
        from tqdm import tqdm
        total = len(values) if hasattr(values, '__len__') else None  # type: ignore[arg-type]
        mapped = tqdm(mapped, total=total)
    return mapped
