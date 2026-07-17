"""F0 determination."""
# %%
import matplotlib.pyplot as plt
import numpy as np
import h5py

import console
from console.service.acquisition_manager import AcquisitionControlManager
from console.utilities.sequences.spectrometry import fid

num_samples     = 8192
acq_bandwidth   = 40e3
signal_coil     = 0         ### use consistant variables pls, thnx, Ruben

class F0():

    @staticmethod
    def fwhm_hz(freq_axis, spectrum):
        """Full width at half maximum of a magnitude spectrum, in Hz."""
        mag = np.abs(spectrum)
        half_max = mag.max() / 2
        above = np.where(mag >= half_max)[0]
        return freq_axis[above[-1]] - freq_axis[above[0]]

    def run(self):

        seq = fid.constructor(
            rf_duration=600e-6, # Duration of the RF pulse
            dead_time=2e-3, # Time between RF pulse and data acquisition
            num_samples= num_samples, # Number of data points to collect
            acq_bandwidth= acq_bandwidth, # Acquisition bandwidth
            flip_angle=np.pi/2, # Flip angle in degrees
        )

        initial_f0 = console.parameter.larmor_frequency

        print(f"ADC duration: {1e3 * num_samples / acq_bandwidth} ms")
        print(f"Initial f0: {console.parameter.larmor_frequency} Hz")


        with AcquisitionControlManager() as mngr:
            mngr.acquisition.set_sequence(sequence=seq, parameter=console.parameter)
            scan_data   = mngr.acquisition.run().receive_data[0]
            
            signal      = scan_data.processed_data
            signal_fft  = np.fft.fftshift(np.fft.fft(np.fft.fftshift(signal)))
            time_axis   = np.arange(scan_data.num_samples)*scan_data.dwell_time*1e3
            freq_axis   = np.fft.fftshift(np.fft.fftfreq(n=scan_data.num_samples,d=scan_data.dwell_time))
            
            
            freq_offset = freq_axis[np.argmax(np.abs(signal_fft[signal_coil,:]))]
            new_f0      = initial_f0 + freq_offset
            fwhm        = F0.fwhm_hz(freq_axis, signal_fft[signal_coil, :])


            print(f"Frequency offset: {freq_offset} Hz")
            print(f"New f0: {new_f0} Hz")
            print(f"FWHM: {fwhm:.1f} Hz")


            console.parameter.larmor_frequency = new_f0
            
            mngr.acquisition.set_sequence(sequence=seq, parameter=console.parameter)
            acq_data   = mngr.acquisition.run()
            scan_data = acq_data.receive_data[0]
            
            signal      = scan_data.processed_data
            signal_fft  = np.fft.fftshift(np.fft.fft(np.fft.fftshift(signal)))
            time_axis   = np.arange(scan_data.num_samples)*scan_data.dwell_time*1e3
            freq_axis   = np.fft.fftshift(np.fft.fftfreq(n=scan_data.num_samples,d=scan_data.dwell_time))

def main():
   f0 = F0()
   f0.run() 

if __name__ == "__main__":
    main()

    
# %% check an old fid 
"""
filename = '/home/openimaging/nexus-console/2026-02-18-session/2026-02-18-140032-fid/rx_data.h5'

with h5py.File(filename, 'r') as f:
    data = f['receive_data/0/processed_data'][:]
signal      = data
signal_fft  = np.fft.fftshift(np.fft.fft(np.fft.fftshift(signal)))
time_axis   = np.arange(scan_data.num_samples)*scan_data.dwell_time*1e3
freq_axis   = np.fft.fftshift(np.fft.fftfreq(n=scan_data.num_samples,d=scan_data.dwell_time))
ax[0].plot(time_axis, np.abs(signal[signal_coil]), label = "Initial")
ax[1].plot(freq_axis, np.abs(signal_fft[signal_coil]), label = "Initial")
plt.show()
"""
