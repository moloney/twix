"""Parsing of PMU (phsyio measurement unit) data"""
import logging
from dataclasses import dataclass
import struct
from typing import Dict

import numpy as np

from .util import BinaryHeader, dotdict


log = logging.getLogger(__name__)


PMU_BLOCK_HEADER = BinaryHeader({
    'timestamp0': 'I',
    'timestamp': 'I',
    'packet_no': 'I',
    'duration': 'I',
})


PMU_SET_HEADER = BinaryHeader({"magic": 'I', "period": 'I'})


PMU_MAGIC = {
    "END": 0x01FF0000,
    "ECG1": 0x01010000,
    "ECG2": 0x01020000,
    "ECG3": 0x01030000,
    "ECG4": 0x01040000,
    "PULS": 0x01050000,
    "RESP": 0x01060000,
    "EXT1": 0x01070000,
    "EXT2": 0x01080000
}


MAGIC_PMU = dict(reversed(item) for item in PMU_MAGIC.items())


class UnknownPacketDataError(Exception):
    pass


class InvalidPmuData(Exception):
    pass


@dataclass
class PmuData:
    """Capture physio measurement unit packets"""

    block_hdr: dotdict

    set_hdrs: Dict[str, dotdict]

    signals: Dict[str, np.ndarray]

    triggers: Dict[str, np.ndarray]
    
    padding: bytes
    
    def encode(self) -> bytes:
        res = bytearray()
        res.extend(PMU_BLOCK_HEADER.encode(self.block_hdr))
        duration = self.block_hdr.duration
        for pmu_type, set_hdr in self.set_hdrs.items():
            res.extend(PMU_SET_HEADER.encode(set_hdr))
            n_pts = duration // set_hdr.period
            out_data = np.empty((2, n_pts), dtype=np.uint16)
            out_data[0, :] = self.signals[pmu_type] * 4096
            out_data[1, :] = self.triggers[pmu_type]
            res.extend(out_data.T.tobytes())
        log.debug("Writing padding '%s' at end of PMU", self.padding)
        res.extend(self.padding)
        return res

    @classmethod
    def decode(klass, data: bytes) -> "PmuData":
        offset = 0
        next_offset = offset + PMU_BLOCK_HEADER.size
        block_hdr = PMU_BLOCK_HEADER.decode(data[offset:next_offset])
        offset = next_offset
        set_hdrs = {}
        signal = {}
        trigger = {}
        n_bytes = len(data)
        found_end = False
        while n_bytes - offset >= 4:
            if struct.unpack('<I', data[offset:offset+4])[0] == PMU_MAGIC["END"]:
                found_end = True
                offset += 4
                break
            next_offset = offset + PMU_SET_HEADER.size
            set_hdr = PMU_SET_HEADER.decode(data[offset:next_offset])
            offset = next_offset
            try:
                pmu_type = MAGIC_PMU[set_hdr.magic]
            except KeyError:
                raise InvalidPmuData(f"Unknown magic number: {set_hdr.magic}")
            assert pmu_type not in set_hdrs
            set_hdrs[pmu_type] = set_hdr
            n_pts = block_hdr.duration // set_hdr.period
            next_offset = offset + (n_pts * 4)
            data = np.frombuffer(data[offset:next_offset], dtype=np.uint16)
            offset = next_offset
            data = data.reshape((n_pts, 2)).T
            signal[pmu_type] = data[0].astype(float) / 4096
            trigger[pmu_type] = data[1].astype(bool)
        if not found_end:
            log.warning("Didn't fine END marker in PMU packet")
        padding = data[offset:]
        if padding:
            logging.debug("Got padding '%s' at end of PMU block", padding)
        return klass(block_hdr, set_hdrs, signal, trigger, padding)
