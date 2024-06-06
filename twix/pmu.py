"""Parsing of PMU (phsyio measurement unit?) data"""
import logging
from dataclasses import dataclass
import struct
from typing import Dict

import numpy as np

from .util import BinaryHeader, dotdict


log = logging.getLogger(__name__)


PACKET_HEADER = BinaryHeader({
    'packet_size': 'I',
    'id': '52s',
    'swapped': 'I',
})


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

    packet_hdr: dotdict

    block_hdr: dotdict

    set_hdrs: Dict[str, dotdict]

    signals: Dict[str, np.ndarray]

    triggers: Dict[str, np.ndarray]
    
    padding: bytes

    @property
    def size(self) -> int:
        return PACKET_HEADER.size + self.packet_hdr.packet_size
    
    def to_file(self, dest_file) -> None:
        start = dest_file.tell()
        PACKET_HEADER.write(self.packet_hdr, dest_file)
        if self.block_hdr:
            PMU_BLOCK_HEADER.write(self.block_hdr, dest_file)
            duration = self.block_hdr.duration
            for pmu_type, set_hdr in self.set_hdrs.items():
                PMU_SET_HEADER.write(set_hdr, dest_file)
                n_pts = duration // set_hdr.period
                out_data = np.empty((2, n_pts), dtype=np.uint16)
                out_data[0, :] = self.signals[pmu_type] * 4096
                out_data[1, :] = self.triggers[pmu_type]
                dest_file.write(out_data.T.tobytes())
        log.debug(
            "Writing padding '%s' at end of PMU at offset: %d", 
            self.padding, 
            dest_file.tell()
        )
        dest_file.write(self.padding)
        bytes_written = dest_file.tell() - start
        assert bytes_written == self.size

    @classmethod
    def from_file(klass, src_file) -> "PmuData":
        start = src_file.tell()
        packet_hdr = PACKET_HEADER.read(src_file)
        if not packet_hdr.id.startswith(b'PMU'):
            padding = src_file.read(packet_hdr.packet_size)
            return PmuData(packet_hdr, dotdict(), {}, {}, {}, padding)
        block_hdr = PMU_BLOCK_HEADER.read(src_file)
        set_hdrs = {}
        signal = {}
        trigger = {}
        bytes_left = packet_hdr.packet_size - PMU_BLOCK_HEADER.size
        while bytes_left >= PMU_SET_HEADER.size:
            set_hdr = PMU_SET_HEADER.read(src_file)
            try:
                pmu_type = MAGIC_PMU[set_hdr.magic]
            except KeyError:
                raise InvalidPmuData(f"Unknown magic number: {set_hdr.magic}")
            assert pmu_type not in set_hdrs
            set_hdrs[pmu_type] = set_hdr
            bytes_left -= PMU_SET_HEADER.size
            if pmu_type == "END":
                log.debug("Got PMU 'END' with duration = %d", set_hdr.duration)
                break
            n_pts = block_hdr.duration // set_hdr.period
            n_bytes = n_pts * 4
            if n_bytes > bytes_left:
                raise InvalidPmuData("Not enough data")
            data = np.frombuffer(src_file.read(n_bytes), dtype=np.uint16)
            data = data.reshape((n_pts, 2)).T
            signal[pmu_type] = data[0].astype(float) / 4096
            trigger[pmu_type] = data[1].astype(bool)
            bytes_left -= n_bytes
        pad_offset = src_file.tell()
        padding = src_file.read(
            PACKET_HEADER.size + packet_hdr.packet_size - (src_file.tell() - start)
        )
        assert struct.unpack('<I', padding)[0] == PMU_MAGIC["END"]
        if padding:
            logging.debug(
                "Got padding '%s' at end of PMU block at offset: %d", 
                padding, 
                pad_offset,
            )
        return klass(packet_hdr, block_hdr, set_hdrs, signal, trigger, padding)
