"""Parsing of chunks of data from a `Meas` dataset"""
from dataclasses import dataclass
from functools import reduce
from typing import Optional, List

import numpy as np

from .util import BinaryHeader, dotdict
from .pmu import PmuData


MDH_HEADER_V1 =  BinaryHeader({
    'dma_info' : 'I',
    'meas_uid' : 'i',
    'scan_count' : 'I',
    'timestamp' : 'I',
    'pmu_timestamp' : 'I',
    'eval_info_mask' : 'Q',
    'samples_in_scan' : 'H',
    'used_channels' : 'H',
    'line' : 'H',
    'acquisition' : 'H',
    'slice' : 'H',
    'partition' : 'H',
    'echo' : 'H',
    'phase' : 'H',
    'repetition' : 'H',
    'set' : 'H',
    'segment' : 'H',
    'ida' : 'H',
    'idb' : 'H',
    'idc' : 'H',
    'idd' : 'H',
    'ide' : 'H',
    'pre' : 'H',
    'post' : 'H',
    'kspace_center_column' : 'H',
    'coil_select' : 'H',
    'readout_off_center' : 'f',
    'time_since_last_rf' : 'I',
    'kspace_center_line' : 'H',
    'kspace_center_partition' : 'H',
    'ice_parameters': '4H',
    'free_parameters' : '4H',
    'sagittal_pos' : 'f',
    'coronal_pos' : 'f',
    'transverse_pos' : 'f',
    'orient_quat' : '4f',
    'channel_id' : 'H',
    'table_pos_negative' : 'H',
})


MDH_HEADER_V2 =  BinaryHeader({
    'dma_info' : 'I',
    'meas_uid' : 'i',
    'scan_count' : 'I',
    'timestamp' : 'I',
    'pmu_timestamp' : 'I',
    'system_type' : 'H',
    'table_pos_delay' : 'H',
    'table_pos_x' : 'i',
    'table_pos_y' : 'i',
    'table_pos_z' : 'i',
    'unused1' : 'I',
    'eval_info_mask' : 'Q',
    'samples_in_scan' : 'H',
    'used_channels' : 'H',
    'line' : 'H',
    'acquisition' : 'H',
    'slice' : 'H',
    'partition' : 'H',
    'echo' : 'H',
    'phase' : 'H',
    'repetition' : 'H',
    'set' : 'H',
    'segment' : 'H',
    'ida' : 'H',
    'idb' : 'H',
    'idc' : 'H',
    'idd' : 'H',
    'ide' : 'H',
    'pre' : 'H',
    'post' : 'H',
    'kspace_center_column' : 'H',
    'coil_select' : 'H',
    'readout_off_center' : 'f',
    'time_since_last_rf' : 'I',
    'kspace_center_line' : 'H',
    'kspace_center_partition' : 'H',
    'sagittal_pos' : 'f',
    'coronal_pos' : 'f',
    'transverse_pos' : 'f',
    'orient_quat' : '4f',
    'ice_parameters': '24H',
    'free_parameters' : '4H',
    'application_counter' : 'H',
    'application_mask' : 'H',
    'checksum' : 'I',
})


CHANNEL_HEADER = BinaryHeader({
    'type_and_len' : 'I',
    'meas_uid' : 'i',
    'scan_count' : 'I',
    'unused2' : 'I',
    'sequence_time' : 'I',
    'unused3' : 'I',
    'channel_id' : 'H',
    'unused4' : 'H',
    'checksum' : 'I',
})


EVAL_INFO_FLAGS = [
    'ACQEND',
    'RTFEEDBACK',
    'HPFEEDBACK',
    'ONLINE',
    'OFFLINE',
    'SYNCDATA',
    'UNKNOWN6',
    'UNKNOWN7',
    'LASTSCANINCONCAT',
    'UNKNOWN9',
    'RAWDATACORRECTION',
    'LASTSCANINMEAS',
    'SCANSCALEFACTOR',
    '2NDHADAMARPULSE',
    'REFPHASESTABSCAN',
    'PHASESTABSCAN',
    'D3FFT',
    'SIGNREV',
    'PHASEFFT',
    'SWAPPED',
    'POSTSHAREDLINE',
    'PHASCOR',
    'PATREFSCAN',
    'PATREFANDIMASCAN',
    'REFLECT',
    'NOISEADJSCAN',
    'SHARENOW',
    'LASTMEASUREDLINE',
    'FIRSTSCANINSLICE',
    'LASTSCANINSLICE',
    'TREFFECTIVEBEGIN',
    'TREFFECTIVEEND',
    'MDS_REF_POSITION',
    'SLC_AVERAGED'
    'TAGFLAG1',
    'CT_NORMALIZE',
    'SCAN_FIRST',
    'SCAN_LAST',
    'UNKNOWN38',
    'UNKNOWN39',
    'FIRST_SCAN_IN_BLADE',
    'LAST_SCAN_IN_BLADE',
    'LAST_BLADE_IN_TR',
    'UNKNOWN43',
    'PACE',
    'RETRO_LASTPHASE',
    'RETRO_ENDOFMEAS',
    'RETRO_REPEATTHISHEARTBEAT',
    'RETRO_REPEATPREVHEARTBEAT',
    'RETRO_ABORTSCANNOW',
    'RETRO_LASTHEARTBEAT',
    'RETRO_DUMMYSCAN',
    'RETRO_ARRDETDISABLED',
    'B1_CONTROLLOOP',
    'SKIP_ONLINE_PHASCOR',
    'SKIP_REGRIDDING',
]
'''Bit flags from 'eval_info_mask' in readout header'''


SUPLEMENT_DATA_FLAGS = [
    'ACQEND',
    'RTFEEDBACK',
    'HPFEEDBACK',
    'SYNCDATA',
    'REFPHASESTABSCAN',
    'PHASESTABSCAN',
    'PHASCOR',
    'NOISEADJSCAN',
]
'''Flags from eval_info_mask that indicate this is reference/calibration data
'''


SUPLEMENT_DATA_MASK = reduce(
    lambda mask, flag: mask | (1 << EVAL_INFO_FLAGS.index(flag)),
    SUPLEMENT_DATA_FLAGS,
    0,
)
'''Mask for eval_info_mask flags that indicate this is ref/calibration data'''


def eval_info_is_set(eval_info_mask: int, flag_name: str) -> bool:
    '''Check if the given flag is set in the `eval_info_mask`'''
    return bool(eval_info_mask & (1 << EVAL_INFO_FLAGS.index(flag_name)))


@dataclass
class RfChannelData:
    """Complex readout data from a single channel"""
    channel_id: int

    data: Optional[np.ndarray] = None

    channel_hdr: Optional[dotdict] = None


# TODO: Need to figure out potential padding in V1 files based on dma_length
class Mdh:
    """Capture an "MDH" (subset of a 'meas' dataset)"""
    def __init__(
        self, 
        hdr: dotdict, 
        rf_data: Optional[List[RfChannelData]] = None, 
        pmu_data: Optional[PmuData] = None, 
        version: int = 2
    ):
        self.hdr = hdr
        self.rf_data = rf_data
        self.pmu_data = pmu_data
        self._version = version

    @property
    def version(self) -> int:
        return self._version

    @property
    def dma_length(self) -> int:
        first16 = self.hdr.dma_info & 0xFFFF
        next8 = (self.hdr.dma_info & 0xFF0000) >> 16
        return first16 + (next8 * 2**16)
    
    @property
    def size(self) -> int:
        if self._version == 1:
            res = MDH_HEADER_V1.size
            res += self.hdr.samples_in_scan * 8
        else:
            res = MDH_HEADER_V2.size
            if self.rf_data is not None:
                res += (
                    len(self.rf_data) 
                    * (CHANNEL_HEADER.size + self.hdr.samples_in_scan * 8)
                )
            else:
                res += self.pmu_data.size
        return res
            
    @property
    def eval_info_flags(self) -> List[str]:
        '''Return human readable list of flags set in eval_info_mask'''
        return [x for x in EVAL_INFO_FLAGS if eval_info_is_set(self.hdr.eval_info_mask, x)]
    
    @property
    def is_primary_data(self) -> bool:
        """Check if this contains actual data vs callibration/reference data
        """
        # TODO: This was developed while our eval_info mask handling had some
        #       bugs, and thus needs to be reevaluated
        if (
            eval_info_is_set(self.hdr.eval_info_mask, 'PATREFANDIMASCAN') 
            or eval_info_is_set(self.hdr.eval_info_mask, 'PATREFSCAN')
        ):
            return True
        else:
            return (self.hdr.eval_info_mask & SUPLEMENT_DATA_MASK) == 0

    @property
    def is_last_acquisition(self) -> bool:
        """Returns True if this is the last acquisition
        """
        return eval_info_is_set(self.hdr.eval_info_mask, 'ACQEND')
    
    def write(self, dest_file):
        # TODO: Need to consider padding here in general
        if self._version == 1:
            MDH_HEADER_V1.write(self.hdr, dest_file)
            dest_file.write(self.rf_data[0].data.tobytes())
        else:
            MDH_HEADER_V2.write(self.hdr, dest_file)
            if self.pmu_data is not None:
                self.pmu_data.to_file(dest_file)
                # TODO: Might need to pad out to dma_length
            else:
                for chan_data in self.rf_data:
                    CHANNEL_HEADER.write(chan_data.channel_hdr, dest_file)
                    dest_file.write(chan_data.data.tobytes())
    
    @classmethod
    def from_file(cls, src_file, version=2, no_data=False) -> "Mdh":
        """Construct by reading from `src_file`"""
        if version == 1:
            hdr = MDH_HEADER_V1.read(src_file)
        else:
            hdr = MDH_HEADER_V2.read(src_file)
        if version == 1:
            pmu_data = None
            data_count = 2 * hdr.samples_in_scan
            data_size = 4 * data_count    
            if no_data:
                rf_data = [RfChannelData(hdr.channel_id)]
                src_file.seek(data_size, 1)
            else:
                rf_data = [RfChannelData(
                    hdr.channel_id,
                    np.frombuffer(src_file.read(data_size),
                                  dtype=np.float32,
                                  count=data_count).view(np.complex64)
                )]
        else:
            assert version == 2            
            if eval_info_is_set(hdr.eval_info_mask, "SYNCDATA"):
                assert hdr.used_channels == 0
                rf_data = None
                pmu_data = PmuData.from_file(src_file)
            else:
                pmu_data = None
                rf_data = []
                for _ in range(hdr.used_channels):
                    chan_hdr = CHANNEL_HEADER.read(src_file)
                    if no_data:
                        data = None
                        src_file.seek(hdr.samples_in_scan * 8, 1)
                    else:
                        data_count = 2 * hdr.samples_in_scan
                        data = np.frombuffer(src_file.read(4 * data_count),
                                            dtype=np.float32,
                                            count=data_count).view(np.complex64)
                    rf_data.append(RfChannelData(chan_hdr.channel_id, data, chan_hdr))
        return cls(hdr, rf_data, pmu_data, version)
