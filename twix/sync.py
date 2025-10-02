"""Handle "sync_data" packets"""
import logging, enum
from dataclasses import dataclass
from typing import Union

from .util import BinaryHeader, dotdict
from .pmu import PmuData


log = logging.getLogger(__name__)


PACKET_HEADER = BinaryHeader({
    'packet_size': 'I',
    'id': '52s',
    'swapped': 'I',
})


class PacketType(enum.Enum):
    UNKNOWN = 0
    PMU = 1


@dataclass
class SyncPacket:
    """Handle 'sync_data' packets"""
    
    hdr: dotdict

    ptype: PacketType

    data: Union[bytes, PmuData]

    @property
    def size(self) -> int:
        return PACKET_HEADER.size + self.hdr.packet_size
    
    def to_file(self, dest_file) -> None:
        PACKET_HEADER.write(self.hdr, dest_file)
        if self.ptype == PacketType.UNKNOWN:
            dest_file.write(self.data)
        else:
            dest_file.write(self.data.encode())
    
    @classmethod
    def from_file(klass, src_file) -> "SyncPacket":
        hdr = PACKET_HEADER.read(src_file)
        data = src_file.read(hdr.packet_size)
        if hdr.id.startswith(b'PMU'):
            ptype = PacketType.PMU
            data = PmuData.decode(data)
        else:
            ptype = PacketType.UNKNOWN
        return SyncPacket(hdr, ptype, data)
