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
            log.debug("Writing PMU data at offset %d", dest_file.tell())
            dest_file.write(self.data.encode())
    
    @classmethod
    def from_file(klass, src_file) -> "SyncPacket":
        start = src_file.tell()
        hdr = PACKET_HEADER.read(src_file)
        data = src_file.read(hdr.packet_size)
        if hdr.id.startswith(b'PMU'):
            ptype = PacketType.PMU
            log.debug("Decoding %d bytes into PMU data", hdr.packet_size)
            data = PmuData.decode(data)
        else:
            ptype = PacketType.UNKNOWN
        return SyncPacket(hdr, ptype, data)
