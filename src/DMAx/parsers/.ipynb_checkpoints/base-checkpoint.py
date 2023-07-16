"""Module defining base dynamic mechanical analysis data output parser."""

import re
import os
import subprocess
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from mastercurves import MasterCurve
from mastercurves.transforms import Multiply
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class DMADataParser:
    def __init__(self, calc_dir):
        self.calc_dir = calc_dir
    
    def pressureparser(self, dielectric=False, noise_filter_run=False, output_xy=True, plot=False, plot_title="Pressure vs runtime", pressdir=None, apply_gaussian_filter=False, gaussian_sigma=6, force_pressure_remove=True, **kwargs):
        simulation_dir = kwargs.get("simulation_dir", self.calc_dir)
        # Function to generate a pressure.txt file with human-readable values for the pressures in all directions over the runtime of the simulation
        press_dict = {"x": "Pxx", "y": "Pyy", "z": "Pzz"}
        os.chdir(simulation_dir)
        if pressdir != None:
            pressdir = pressdir
            if dielectric or noise_filter_run:
                pressure_mult = 0.101325
            else:
                pressure_mult = -0.101325
        else:
            if dielectric:
                grepPressDir = subprocess.check_output("grep 'Field direction = ' in.*", shell=True).decode("utf-8").split()
                pressdir = grepPressDir[5]
                pressure_mult = 0.101325
            elif noise_filter_run:
                pressdir = None
                pressure_mult = 0.101325
            else:
                grepPressDir = subprocess.check_output("grep 'wiggle' in.*", shell=True).decode("utf-8").split()
                pressdir = grepPressDir[5]
                pressure_mult = -0.101325
        if "output.txt" not in os.listdir() or force_pressure_remove:
            log_file = [i for i in os.listdir(simulation_dir) if i[-3:] == "log"][0]
            with open(log_file, 'r') as file:
                lines = file.readlines()
            start_NVT_prod = [i for i, line in enumerate(lines) if 'Step          Time          TotEng         KinEng          Temp          PotEng' in line][0]
            end_NVT_prod = [i for i, line in enumerate(lines) if 'Loop time of' in line][-1]
            if start_NVT_prod and end_NVT_prod:
                NVT_prod_lines = lines[start_NVT_prod:end_NVT_prod]
                NVT_prod_lines = [line for line in NVT_prod_lines if "SHAKE" not in line and "Bond" not in line]
                NVT_prod_data = [line.split() for line in NVT_prod_lines]
                try:
                    NVT_prod_df = pd.DataFrame(NVT_prod_data[1:], columns=NVT_prod_data[0])
                except:
                    print("The simulation {} did not run successfully".format(simulation_dir))
                    return None, None
                NVT_prod_df.to_csv('output.txt', sep='\t', index=False)
        runtime = np.array(NVT_prod_df["Time"].astype(float))
        if noise_filter_run:
            pressures = [np.array(NVT_prod_df[pressdir].astype(float)) * pressure_mult for pressdir in press_dict.values()] #in MPa
            pressure = np.mean(pressures, axis=0)
        else:
            pressure = np.array(NVT_prod_df[press_dict[pressdir]].astype(float)) * pressure_mult #in MPa
        if apply_gaussian_filter:
            pressure = gaussian_filter1d(pressure, gaussian_sigma)
        if plot:
            plt.title(plot_title)
            plt.xlabel("Runtime (fs)")
            plt.ylabel("Pressure (MPa)")
            plt.ticklabel_format(style="scientific", scilimits=(0,4))
            plt.scatter(runtime, pressure, s=1)
        if output_xy:
            return runtime, pressure
    
    def fit_sin(self, runtime, pressure, dielectric=False, plot_pressure=False, plot_title="Pressure vs runtime (+ Scipy Fitting)", fit_color='red', original_signal_color='green', do_prints=False, **kwargs):
        '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
        simulation_dir = kwargs.get("simulation_dir", self.calc_dir)
        os.chdir(simulation_dir)
        if runtime is None and pressure is None:
            return {"Fit Curve": None, "Predicted Frequency": None, "Real Phi": None, "Loss Tangent": None, "Storage Modulus": None, "Loss Modulus": None, "MAE": None, "RMSE": None, "R2": None}      
        runtime = np.array(runtime)
        pressure = np.array(pressure)
        ff = np.fft.fftfreq(len(runtime), (runtime[1]-runtime[0]))   # assume uniform spacing
        if dielectric:
            guess_freq = 2 * 1 / float(subprocess.check_output("grep 'periodfs' in.*", shell=True).decode("utf-8").split()[3])
        else:
            guess_freq = 1 / float(subprocess.check_output("grep 'wiggle' in.*", shell=True).decode("utf-8").split()[8])
        guess_w = 2 * np.pi * guess_freq
        guess_amp = np.std(pressure) * 2.**0.5
        guess_offset = np.mean(pressure)
        guess = np.array([guess_amp, 0., guess_offset])

        def sinfunc(t, A, p, c):  return A * np.sin(guess_w * t + p) + c
        popt, pcov = sp.optimize.curve_fit(sinfunc, runtime, pressure, p0=guess)
        A, p, c = popt
        fitfunc = lambda t: A * np.sin(guess_w * t + p) + c
        y = fitfunc(runtime)
        if plot_pressure:
            plt.scatter(runtime, pressure, s=1, label="Pressure data")
            plt.ticklabel_format(style="scientific", scilimits=(0,4))
            plt.title(plot_title)
            plt.xlabel("Runtime (fs)")
            plt.ylabel("Pressure (MPa)")
            plt.plot(runtime, y, fit_color, label="Scipy fitted curve")
            plt.plot(runtime, A * np.sin(guess_w * runtime) + c, original_signal_color, label="Original Field Applied", linestyle="--")
            plt.legend()
        sin_offset = p
        mae, rmse, r2 = mean_absolute_error(pressure, y), mean_squared_error(pressure, y, squared=False), r2_score(pressure, y)
        if do_prints:
            print('Scipy Fitted Parameters:')
            #print(tpe_best)
            print("Scipy Angular Frequency (w): " + str(guess_w))
            print("Scipy Offset (a0): " + str(c))
            print("Scipy Amplitude (a1): " + str(A))
            print("Scipy Predicted Frequency (f): " + str(guess_w / (2 * np.pi) * 1e6) + " GHz")
            print("Scipy Real 'phi': " + str(sin_offset))
            print("Scipy Loss Tangent is: " + str(np.tan(sin_offset)))
            print("MAE, RMSE, R2: {}, {}, {}".format(mae, rmse, r2))
        return {"Original Curve": pressure, "Fit Curve": y, "Predicted Frequency": guess_w / (2 * np.pi) * 1e6, "Real Phi (rad)": sin_offset, "Real Phi (deg)": sin_offset * 180 / np.pi, "Loss Tangent": np.tan(sin_offset), "Storage Modulus": np.cos(sin_offset), "Loss Modulus": np.sin(sin_offset), "MAE": mae, "RMSE": rmse, "R2": r2}