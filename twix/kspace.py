import math
from itertools import product as iproduct
from typing import Tuple
try:
    import cPickle as pickle
except ImportError:
    import pickle


default_counter_order = ('ide',
                         'idd',
                         'idc',
                         'idb',
                         'ida',
                         'acquisition',
                         'echo',
                         'repetition',
                         'set',
                         'segment',
                         'partition',
                         'channel_id',
                         'slice',
                         'phase',
                         'line',
                        )
'''Default slowest-to-fastest order for counters in the k-space array'''


ordinal_counters = ('channel_id',)
'''Counters where we use the order rather than the actual value as an index
'''


class KSpaceSpec(object):
    '''Maps a K-space array to a sequential series of readouts

    Parameters
    ----------

    dim_info : sequence of tuples
        Each element is a tuple giving the name and size of each dimension.
        The last dimension should always be the "readout" dimension.

    center_indices : dict
        Map dim names to the center of k-space index values.

    idx_map : dict
        Map n-D K-space indices (minus the last index) to 1-D readout indices.

    ro_offset : int
        Offset for acquired readout when zero padding that dimension.
        If None, defaults to right-aligning the acquired data.
    '''
    def __init__(self, dim_info, center_indices, idx_map, ro_offset=None):
        if dim_info[-1][0] != 'readout':
            raise ValueError("Last dim should always be 'readout'")
        self._dim_info = tuple(dim_info)
        self._dim_names = tuple(d[0] for d in dim_info)
        self._shape = tuple(x[1] for x in dim_info)
        # TODO: check these indices make sense
        self._center_indices = center_indices
        self._idx_map = idx_map
        self._ro_offset = ro_offset

    @property
    def dim_names(self) -> Tuple[str, ...]:
        return self._dim_names

    @property
    def dim_info(self):
        return self._dim_info

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._shape

    @property
    def center_indices(self):
        return self._center_indices.copy()

    @property
    def ro_offset(self):
        return self._ro_offset

    def dim_size(self, dim_name: str) -> int:
        return self._shape[self._dim_names.index(dim_name)]

    def get_ro_idx(self, k_spc_loc):
        '''Get the sequential readout location for a k-Space location

        Parameters
        ----------
        k_spc_loc: dict or sequence of tuples
        '''
        if isinstance(k_spc_loc, dict):
            k_spc_loc = tuple(sorted(k_spc_loc.items()))
        else:
            k_spc_loc = tuple(sorted(k_spc_loc))
        res = self._idx_map.get(pickle.dumps(k_spc_loc,
                                             pickle.HIGHEST_PROTOCOL))
        if res is None:
            raise IndexError("No readout found for location: %s" % k_spc_loc)
        return res

    def pad_dim(self, dim_name, padded_size):
        new_dim_info = []
        # TODO: Warn if padded size doesn't make sense
        for name, size in self.dim_info:
            if name == dim_name:
                new_dim_info.append((name, padded_size))
            else:
                new_dim_info.append((name, size))
        self._dim_info = tuple(new_dim_info)
        self._shape = tuple(x[1] for x in self._dim_info)

    def get_chunk_info(self, fixed=None, bounds=None):
        '''Get information about a subset of k-space

        Parameters
        ----------
        fixed : dict
            Map dimension names to single fixed indices

        bounds : dict
            Map dimension names to tuples of lower/upper bounds

        Returns
        -------
        chunk_shape : tuple
            The shape of the k-space chunk

        ro_map : dict
            Maps 1-D indices of all needed readouts to k-space chunk indices
        '''
        if fixed is None:
            fixed = {}
        else:
            for dim_name in fixed:
                if dim_name not in self._dim_names:
                    raise ValueError("Unknown dimension: %s" % dim_name)
        if bounds is None:
            bounds = {}
        else:
            for dim_name in bounds:
                if dim_name not in self._dim_names:
                    raise ValueError("Unknown dimension: %s" % dim_name)

        # Figure out the chunk of the array we are considering
        lb = []
        ub = []
        out_shape = []
        padded_centers = {}
        for dim_name, dim_size in self._dim_info[:-1]:
            if dim_name in self._center_indices:
                # TODO: Is taking the ceiling here correct?
                padded_centers[dim_name] = int(math.ceil(dim_size / 2.0))
            if dim_name in fixed:
                fixed_val = fixed[dim_name]
                if not 0 <= fixed_val < dim_size:
                    raise IndexError("Fixed dim '%s' out of bounds" % dim_name)
                lb.append(fixed_val)
                ub.append(fixed_val + 1)
            elif dim_name in bounds:
                lower, upper = bounds[dim_name]
                lower = 0 if lower is None else lower
                upper = dim_size if upper is None else upper
                if not 0 <= lower < upper <= dim_size:
                    raise IndexError("Invalid bounds for dim '%s'" % dim_name)
                lb.append(lower)
                ub.append(upper)
                out_shape.append(upper - lower)
            else:
                lb.append(0)
                ub.append(dim_size)
                out_shape.append(dim_size)
        out_shape.append(self._dim_info[-1][1])

        # Build the map for any data that was actually acquired
        ro_map = {}
        for full_arr_idx in iproduct(*[range(l ,u) for l, u in zip(lb, ub)]):
            map_idx = []
            chunk_idx = []
            for dim_idx, (name, _) in enumerate(self.dim_info[:-1]):
                aidx = full_arr_idx[dim_idx]
                if name in padded_centers:
                    centering_offset = padded_centers[name] - self.center_indices[name]
                else:
                    centering_offset = 0
                map_idx.append((name, aidx - centering_offset))
                if name not in fixed:
                    chunk_idx.append(aidx - lb[dim_idx])
            map_idx = tuple(sorted(map_idx))
            ro_idx = self._idx_map.get(pickle.dumps(map_idx, pickle.HIGHEST_PROTOCOL))
            if ro_idx is not None:
                ro_map[ro_idx] = tuple(chunk_idx)
        return (out_shape, ro_map)
