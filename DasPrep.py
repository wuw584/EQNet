import numpy as np
from scipy import signal
from nptdms import TdmsFile
import h5py
import segyio

def read_das(fname, **kwargs):
    if fname.lower().endswith('.tdms'):
        return _read_das_tdms(fname, **kwargs)
    
    elif fname.lower().endswith('.h5'):
        return _read_das_h5(fname, **kwargs)
    
    elif fname.lower().endswith(('.segy', 'sgy')):
        return _read_das_segy(fname, **kwargs)
    
    else:
        print('DAS data format not supported.')

    
def _read_das_h5(fname, metadata=False):
    with h5py.File(fname, 'r') as h5_file:
        # 获取数据集
        dataset = h5_file['Acquisition/Raw[0]/RawData']
        nch, nt = dataset.shape  # 直接从数据shape获取通道数
        
        # 获取属性（带默认值）
        dx = h5_file['Acquisition'].attrs.get('SpatialSamplingInterval', 10.0)
        GL = h5_file['Acquisition'].attrs.get('GaugeLength', 10.0)
        fs = h5_file['Acquisition/Raw[0]'].attrs.get('OutputDataRate', 100.0)
        dt = 1.0 / fs
        
        if metadata:
            # 获取时间数据（如果存在）
            time_arr = h5_file.get('Acquisition/Raw[0]/RawDataTime', None)
            if time_arr is not None:
                dt = np.diff(time_arr[:]).mean()/1e6
                nt = len(time_arr)
            
            return {
                'dt': dt,
                'nt': nt,
                'dx': dx,
                'nch': nch,
                'GL': GL,
                'fs': fs,
                'headers': dict(h5_file['Acquisition'].attrs)
            }
        else:
            return dataset[()]
    
    
    
def _read_das_tdms(fname, **kwargs):
    
    ### https://nptdms.readthedocs.io/en/stable/quickstart.html
    
    tdms_file = TdmsFile.read(fname) 
    nch = len(tdms_file['Measurement'])
    metadata = kwargs.pop('metadata', False)

    if metadata:
        dt = 1./tdms_file.properties['SamplingFrequency[Hz]']
        dx = tdms_file.properties['SpatialResolution[m]']
        nt = len(tdms_file['Measurement']['0'])
        GL = tdms_file.properties['GaugeLength']
        headers = tdms_file.properties
        return {'dt': dt, 
                'nt': nt,
                'dx': dx, 
                'nch': nch,
                'GL': GL,
                'headers': headers}
    else:
        ch1 = kwargs.pop('ch1', 0)
        ch2 = kwargs.pop('ch2', nch)
        data = np.asarray([tdms_file['Measurement'][str(i)] for i in range(ch1, ch2)])
        return data

    
def _read_das_segy(fname, **kwargs):
    
    # https://github.com/equinor/segyio-notebooks/blob/master/notebooks/basic/02_segy_quicklook.ipynb
    
    metadata = kwargs.pop('metadata', False)
    
    with segyio.open(fname, ignore_geometry=True) as segy_file:
    
        nch = segy_file.tracecount
        
        if metadata:
            dt = segyio.tools.dt(segy_file) / 1e6
            nt = segy_file.samples.size
            return {'dt': dt, 
                    'nt': nt,
                    'nch': nch}
        else:   
            ch1 = kwargs.pop('ch1', 0)
            ch2 = kwargs.pop('ch2', nch)
            data = segy_file.trace.raw[ch1:ch2]
            return data
    
    
def das_preprocess(data_in):
    data_out = signal.detrend(data_in)
    data_out = data_out - np.median(data_out, axis=0) 
    return data_out

def tapering(data, alpha):
    nt = data.shape[1]
    window = signal.windows.tukey(nt, alpha)
    data = data * window[None, :]
    return data


def bandpass(data, dt, fl, fh):
    sos = signal.butter(6, [fl, fh], 'bp', fs=1/dt, output='sos')
    data = signal.sosfiltfilt(sos, data, axis=1)
    return data


def highpass(data, dt, fl):
    sos = signal.butter(6, fl, 'hp', fs=1/dt, output='sos')
    data = signal.sosfiltfilt(sos, data, axis=1)
    return data

def lowpass(data, dt, fh):
    sos = signal.butter(6, fh, 'lp', fs=1/dt, output='sos')
    data = signal.sosfiltfilt(sos, data, axis=1)
    return data
