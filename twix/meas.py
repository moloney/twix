"""Twix "meas" file handling

Inspired by and includes code from "vespa" (http://scion.duhs.duke.edu/vespa/)
"""

import os, struct, re, logging
from datetime import datetime
from collections import deque
from pathlib import Path
from typing import Iterable, List, Optional, Set

from packaging.version import parse as LooseVersion  # for syngo version comparision

try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np

from .util import BinaryHeader, _read_cstr, dotdict
from .mdh import Mdh
from .kspace import default_counter_order, ordinal_counters, KSpaceSpec


log = logging.getLogger(__name__)


MIN_OFFSET = 10240


class KSpaceSizeError(Exception):
    """Thrown if the computed k-space size is too small for the data"""


DEP_TYPES = ("SensMap", "ChannelMixing", "RFMap", "B0Map")


class Meas(object):
    """A dataset containing some meta data and one or more chunks of readout data (MDH)
    """

    def __init__(
        self,
        src_file,
        offset,
        length=None,
        meas_id=None,
        file_id=None,
        protocol=None,
        patient=None,
        version=None,
        cntr_order=None,
    ):
        self._src_file = src_file
        self._offset = offset
        self._length = length
        self._meas_id = meas_id
        self._file_id = file_id
        self._protocol = protocol
        self._patient = patient
        self._version = version
        self._cntr_order = cntr_order
        self._meta = None
        self._k_space_spec = None
        self._mdh_locs = []
        self._mdh_locs_complete = False
        self._ro_per_mdh = []

        # Use default ordering for counters if none was given
        if self._cntr_order is None:
            self._cntr_order = default_counter_order

        # Lineup with our file offset if needed
        if self._src_file.tell() != self._offset:
            self._src_file.seek(self._offset)

        # Read binary descriptors at front of header
        (header_size, n_evps) = struct.unpack("<2I", src_file.read(8))
        self._header_size = header_size
        self._n_evps = n_evps

        # Determine version automatically
        # TODO: Update once proper meta data parsing is included
        if self._version is None:
            vrs_regex = re.compile(r"syngo MR (?P<syngo_version>[A-Z][0-9]+)")
            for name, evp_data in self.meta:
                match = vrs_regex.search(evp_data)
                if match:
                    syngo_version = match.group("syngo_version")
                    if LooseVersion(syngo_version) < LooseVersion("D11"):
                        self._version = 1
                    else:
                        self._version = 2
                    break
            else:
                raise ValueError("Could not automatically determine version")
        if self._version not in (1, 2):
            raise ValueError("Unknown version: %s" % self._version)
        log.debug("Initialized version %d Meas", self._version)

    @property
    def meta(self):
        """The meta data associated with this measurement"""
        if self._meta is not None:
            return self._meta

        evp_offset = self._offset + 8
        if self._src_file.tell() != evp_offset:
            self._src_file.seek(evp_offset)
        log.debug("Reading meta data at offset: %d", evp_offset)
        evps = []
        for evp_idx in range(self._n_evps):
            name = _read_cstr(self._src_file)
            (evp_size,) = struct.unpack("<I", self._src_file.read(4))
            evp_data = self._src_file.read(evp_size).decode()
            evps.append((name, evp_data))
        curr_offset = self._src_file.tell()
        hdr_pad_size = self._offset + self._header_size - curr_offset
        assert hdr_pad_size >= 0
        if hdr_pad_size == 0:
            self._hdr_padding = b""
        else:
            self._hdr_padding = self._src_file.read(hdr_pad_size)
        # TODO: handle meta data parsing
        self._meta = evps
        return self._meta
    
    @property
    def length(self) -> int:
        if self._length is None:
            curr = self._src_file.tell()
            self._src_file.seek(0, os.SEEK_END)
            self._length = self._src_file.tell()
            self._src_file.seek(curr)
        return self._length


    def get_dependency_uids(self, dep_types: Iterable[str] = DEP_TYPES) -> Set[int]:
        """Get set of meas UID values for measurements this measurement depends on"""
        res = set()
        for meta_section, meta_str in self.meta:
            for dep_type in dep_types:
                pattern = rf'<ParamLong."l{dep_type}UID">\s*{{\s*([0-9]+)\s*}}'
                matches = re.findall(pattern, meta_str, flags=re.MULTILINE)
                for match in matches:
                    uid = int(match)
                    if uid != -1:
                        res.add(uid)
        return res

    def gen_mdhs(self, no_data: bool = False):
        """Generates MDHs (chunks of data) as stored in the file"""
        mdh_offset = self._offset + self._header_size
        mdh_idx = 0
        done = False
        while not done:
            if self._src_file.tell() != mdh_offset:
                self._src_file.seek(mdh_offset)
            mdh = Mdh.from_file(self._src_file, self._version, no_data)
            mdh_idx += 1
            if len(self._mdh_locs) < mdh_idx:
                self._mdh_locs.append(mdh_offset)
            mdh_offset += mdh.dma_length
            if mdh.is_last_acquisition:
                self._mdh_locs_complete = True
                done = True
            yield mdh

    def gen_readouts(self, no_data: bool = False, primary_only: bool = True):
        """Convienance method to loop over each RF channel from each MDH from `gen_mdhs`
        """
        for mdh in self.gen_mdhs(no_data):
            if mdh.rf_data is None or primary_only and not mdh.is_primary_data:
                continue
            for chan_data in mdh.rf_data:
                yield (mdh, chan_data)

    def get_k_space_spec(self) -> KSpaceSpec:
        """Get mapping from "counters" to k-space axes and their size

        This requires a full pass through the file so we can find all varying
        counters, and thus it can be slow on large files.
        """
        if self._k_space_spec != None:
            return self._k_space_spec
        # Do a first pass through the data and figure out which counters vary
        non_dupes = {}
        indices = deque()
        counter_sets = [set() for name in self._cntr_order]
        found_counters = None
        cntr_pack_fmt = None
        first_mdh = None
        n_readouts = 0
        for mdh, chan_data in self.gen_readouts(no_data=True):
            assert not mdh.is_last_acquisition
            n_readouts += 1
            if first_mdh is None:
                first_mdh = mdh
                samples_in_scan = first_mdh.hdr.samples_in_scan
                center_indices = {
                    "column": first_mdh.hdr.kspace_center_column,
                    "line": first_mdh.hdr.kspace_center_line,
                    "partition": first_mdh.hdr.kspace_center_partition,
                }
            else:
                if samples_in_scan != mdh.hdr.samples_in_scan:
                    raise ValueError(
                        "The samples_in_scan changed: %d vs %d"
                        % (samples_in_scan, mdh.hdr.samples_in_scan)
                    )
            cntr_vals = []
            for cidx, cntr in enumerate(self._cntr_order):
                cntr_val = mdh.hdr.get(cntr)
                if cntr_val is None and chan_data.channel_hdr is not None:
                    cntr_val = chan_data.channel_hdr.get(cntr)
                if cntr_val is None:
                    continue  # TODO: Exception or warning here?
                counter_sets[cidx].add(cntr_val)
                cntr_vals.append((cntr, cntr_val))
            n_cntrs = len(cntr_vals)
            if found_counters is None:
                found_counters = [x[0] for x in cntr_vals]
                cntr_pack_fmt = "%dH" % n_cntrs
            packed_cntrs = struct.pack(cntr_pack_fmt, *(x[1] for x in cntr_vals))
            indices.append((packed_cntrs, ro_idx))
            # TODO: More efficient way to handle duplicate counters?
            for idx, (name, val) in enumerate(cntr_vals):
                if not name in non_dupes:
                    non_dupes[name] = set()
                for name2, val2 in cntr_vals:
                    if val != val2 or name == name2:
                        non_dupes[name].add(name2)
        # Pull out info about the varying counters
        varying_set = set()
        varying_counters = []
        varying_values = {}
        ordinal_lists = {}
        for idx, counter_name in enumerate(self._cntr_order):
            if counter_name in ordinal_counters:
                count = len(counter_sets[idx])
            else:
                count = max(counter_sets[idx]) + 1
            if count > 1:
                if counter_name in ordinal_counters:
                    ordinal_lists[counter_name] = sorted(list(counter_sets[idx]))
                varying_set.add(counter_name)
                varying_counters.append((counter_name, count))
                varying_values[counter_name] = sorted(counter_sets[idx])
        # Remove any varying counters that are duplicates
        all_dupes = set()
        for name, _ in reversed(varying_counters):
            if name in all_dupes:
                continue
            dupes = varying_set.difference(non_dupes[name])
            all_dupes.update(dupes)
            if len(dupes) != 0:
                varying_counters = [
                    item for item in varying_counters if not item[0] in dupes
                ]
        # Basic shape_info and sanity check
        full_count = 1
        shape = []
        for dim_name, dim_size in varying_counters:
            full_count *= dim_size
            shape.append(dim_size)
        if full_count < n_readouts:
            raise KSpaceSizeError(
                "K-space shape %s is too small "
                "(%d readouts > %d possible indices"
                % (shape, n_readouts, full_count)
            )
        # Create map from our multi-dimensional k-space indices to the
        # sequential readout indices
        idx_map = {}
        while len(indices) >= 1:
            k_spc_idx = []
            packed_cntrs, ro_idx = indices.popleft()
            cntr_vals = struct.unpack(cntr_pack_fmt, packed_cntrs)
            for name, _ in varying_counters:
                cntr_idx = found_counters.index(name)
                if name in ordinal_counters:
                    idx_val = ordinal_lists[name].index(cntr_vals[cntr_idx])
                else:
                    idx_val = cntr_vals[cntr_idx]
                k_spc_idx.append((name, idx_val))
            k_spc_idx = tuple(sorted(k_spc_idx))
            idx_map[pickle.dumps(k_spc_idx, pickle.HIGHEST_PROTOCOL)] = ro_idx
        end_dt = datetime.now()
        # Build and return the KSpaceSpec object
        dim_info = varying_counters + [("readout", samples_in_scan)]
        self._k_space_spec = KSpaceSpec(dim_info, center_indices, idx_map)
        return self._k_space_spec

    def _insert_ro(self, readout, arr, arr_idx, ro_offset):
        full_len = arr.shape[-1]
        rdata = readout.data
        acq_len = len(rdata)
        if acq_len < full_len:
            # We have readout zero padding
            if ro_offset is None:
                ro_offset = full_len - acq_len
                tail_padding = 0
            else:
                tail_padding = (full_len - acq_len) - ro_offset
                if tail_padding < 0:
                    raise ValueError("Readoug offset is too large")
            if readout.eval_info_is_set("REFLECT"):
                arr[arr_idx][tail_padding:-ro_offset] = rdata[::-1]
            else:
                arr[arr_idx][ro_offset:-tail_padding] = rdata
        elif acq_len == full_len:
            if readout.eval_info_is_set("REFLECT"):
                arr[arr_idx] = rdata[::-1]
            else:
                arr[arr_idx] = rdata
        else:
            raise IndexError("Readout dim is too small for acquired data")

    def _fill_with_seek(self, spec, ro_map, arr):
        max_ro_per_mdh = np.cumsum(self._ro_per_mdh) - 1
        ro_indices = sorted(ro_map.keys())
        ro_buf = None
        last_idx = None
        for ro_idx in ro_indices:
            if ro_buf is None or ro_idx > last_idx:
                mdh_idx = np.searchsorted(max_ro_per_mdh, ro_idx)
                offset = self._mdh_locs[mdh_idx]
                if self._src_file.tell() != offset:
                    self._src_file.seek(offset)
                ro_buf = self._readout_class.from_file(self._src_file)
                if mdh_idx == 0:
                    first_idx = 0
                else:
                    first_idx = max_ro_per_mdh[mdh_idx - 1] + 1
                last_idx = max_ro_per_mdh[mdh_idx]
            buf_idx = ro_idx - first_idx
            arr_idx = ro_map[ro_idx]
            self._insert_ro(ro_buf[buf_idx], arr, arr_idx, spec.ro_offset)

    def _fill_seq(self, spec, ro_map, arr):
        ro_indices = deque(sorted(ro_map.keys()))
        curr_ro_idx = ro_indices.popleft()
        for ro_idx, readout in enumerate(self.gen_readouts()):
            if ro_idx == curr_ro_idx:
                arr_idx = ro_map[ro_idx]
                self._insert_ro(readout, arr, arr_idx, spec.ro_offset)
                if len(ro_indices) == 0:
                    break
                curr_ro_idx = ro_indices.popleft()

    def get_k_space(self, spec=None, fixed=None, bounds=None):
        """Get (some of) the k-space array

        You can fix some of the counters to only get a subset on the k-space
        array, and thus constrain memory use.
        """
        if spec is None:
            spec = self.get_k_space_spec()

        # Figure out the indices of the readouts we need
        out_shape, ro_map = spec.get_chunk_info(fixed, bounds)

        # Create a zeroed k-space array and fill in the acquired data
        k_spc = np.zeros(out_shape, dtype=np.complex64)
        if self._mdh_locs_complete:
            self._fill_with_seek(spec, ro_map, k_spc)
        else:
            self._fill_seq(spec, ro_map, k_spc)

        return k_spc

    def write(self, dest_file, zero_padding: bool = False):
        """Write data to a file object"""
        start = dest_file.tell()
        dest_file.write(struct.pack("<2I", self._header_size, self._n_evps))
        for evp_name, evp_data in self.meta:
            dest_file.write(evp_name.encode() + b"\x00")
            evp_bytes = evp_data.encode()
            dest_file.write(struct.pack("<I", len(evp_bytes)))
            dest_file.write(evp_bytes)
        if zero_padding:
            dest_file.write(b"\x00" * len(self._hdr_padding))
        else:
            dest_file.write(self._hdr_padding)
        log.debug("Writing MDHs start at offset: %d", dest_file.tell())
        for mdh in self.gen_mdhs():
            mdh.write(dest_file, zero_padding)
        padding = self.length - (dest_file.tell() - start)
        if padding:
            log.debug("Found padding at end of data set")
            dest_file.write(b"\x00" * padding)


MEAS_RECORD = BinaryHeader(
    {
        "meas_id": "I",
        "file_id": "I",
        "offset": "Q",
        "length": "Q",
        "patient": "64s",
        "protocol": "64s",
    }
)


class MeasFile(object):
    """A single Twix file, which may include one or more measurements.

    Usually the last measurement is the actual experiment while any earlier
    measurements are some sort of preparation/calibration.
    """

    def __init__(self, src, version=None):
        if isinstance(src, str):
            self._src_file = open(src, "rb")
        elif isinstance(src, Path):
            self._src_file = src.open("rb")
        else:
            self._src_file = src
        self._meas: List[Meas] = []
        self._meas_records: Optional[List[dotdict]] = None
        # On V1 files, "test" will be the header size and thus never zero while on V2
        # files it will always be zero
        (test,) = struct.unpack("<I", self._src_file.read(4))
        if test != 0:
            self._meas.append(
                Meas(
                    self._src_file,
                    0,
                    None,
                    version=version,
                )
            )
        else:
            if version is not None:
                assert version == 2
            version = 2
            (n_meas,) = struct.unpack("<I", self._src_file.read(4))
            self._meas_records = [
                MEAS_RECORD.read(self._src_file) for _ in range(n_meas)
            ]
            for meas_record in self._meas_records:
                self._meas.append(
                    Meas(
                        self._src_file,
                        meas_record.offset,
                        meas_record.length,
                        meas_record.meas_id,
                        meas_record.file_id,
                        meas_record.protocol.rstrip(b"\0"),
                        meas_record.patient.rstrip(b"\0"),
                        version=version,
                    )
                )

    @property
    def n_meas(self) -> int:
        """The number of measurements in this file"""
        return len(self._meas)
    
    def get_meas(self, meas_idx: int = -1) -> Meas:
        """Get the measurment with the given index"""
        return self._meas[meas_idx]

    def get_meta(self, meas_idx: int = -1):
        """Get meta data from the measurement at `meas_idx`"""
        return self._meas[meas_idx].meta
    
    def get_missing_dep_uids(self, meas_idx: int = -1):
        """Get list of UIDs for missing dependencies of last measurement"""
        # TODO: Need to get MeasUID of embedded measurements and exclude those
        return self._meas[meas_idx].get_dependency_uids()

    def gen_mdhs(self, no_data: bool = False, meas_idx: int = -1):
        for mdh in self._meas[meas_idx].gen_mdhs():
            yield mdh

    def get_k_space_spec(self, meas_idx: int = -1):
        """Get info about the shape of the full k-space array"""
        return self._meas[meas_idx].get_k_space_spec()

    def get_k_space(self, spec=None, fixed=None, bounds=None, meas_idx: int = -1):
        return self._meas[meas_idx].get_k_space(spec, fixed, bounds)

    def save(self, dest_path, prepend=None, zero_padding: bool = False):
        """Save to a file, optionally prepending one or more pre scan datasets"""
        version = self._meas[0]._version
        if prepend and version == 1:
            raise ValueError("Can't prepend data with V1 format")
        with open(dest_path, "wb") as out_f:
            if version == 1:
                assert len(self._meas) == 1
                self._meas[0].write(out_f)
            elif version == 2:
                if prepend is None:
                    prepend = []
                meas_out = prepend + self._meas
                # Determine offsets to data sets
                if not prepend:
                    out_records = self._meas_records
                else:
                    out_records = []
                    # Using MIN_OFFSET is probably not required but does help our output
                    # better match the files written by the console
                    curr_offset = max(MIN_OFFSET, 8 + (len(meas_out) * MEAS_RECORD.size))
                    for pre_meas in prepend:
                        record = dotdict(
                            {
                                "meas_id": pre_meas._meas_id,
                                "file_id": pre_meas._file_id,
                                "offset": curr_offset,
                                "length": pre_meas.length,
                                "patient": pre_meas._patient,
                                "protocol": pre_meas._protocol,
                            }
                        )
                        out_records.append(record)
                        curr_offset += record.length
                        # It seems like data sets must be aligned at 512 byte boundaries
                        tail_padding = 512 - (curr_offset % 512)
                        if tail_padding != 512:
                            curr_offset += tail_padding
                    for record in self._meas_records:
                        record = record.copy()
                        record.offset = curr_offset
                        curr_offset += record.length
                        tail_padding = 512 - (curr_offset % 512)
                        if tail_padding != 512:
                            curr_offset += tail_padding
                        out_records.append(record)
                # Write the file
                out_f.write(struct.pack("<2I", 0, len(meas_out)))
                for record in out_records:
                    MEAS_RECORD.write(record, out_f)
                pad_len = out_records[0].offset - out_f.tell()
                assert pad_len >= 0
                out_f.write(b"\x00" * pad_len)
                for meas_idx, meas in enumerate(meas_out):
                    curr_offset = out_f.tell()
                    pad_len = out_records[meas_idx].offset - curr_offset
                    assert pad_len >= 0
                    out_f.write(b"\x00" * pad_len)
                    curr_offset += pad_len
                    log.debug("Writing meas dataset at offset: %d", curr_offset)
                    meas.write(out_f, zero_padding)
                tail_padding = 512 - (out_f.tell() % 512)
                if tail_padding != 512:
                    out_f.write(b"\x00" * tail_padding)
            else:
                raise ValueError("Invalid version")
