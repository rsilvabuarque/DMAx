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

def frequency_dirname_parser(frequency):
    freq_range_dict = {"Hz": 1e0, "KHz": 1e3, "MHz": 1e6, "GHz": 1e9, "THz": 1e12, "PHz": 1e15, "EHz": 1e18}
    for freq_mag, freq in freq_range_dict.items():
        dirname_val = frequency / freq
        if 1 <= dirname_val < 1000:
            return "{}{}".format(dirname_val, freq_mag)

def sort_dict_by_frequency(dict_to_sort):
    # Define a dictionary mapping frequency unit suffixes to their corresponding multipliers
    unit_multipliers = {'Hz': 1, 'kHz': 1e3, 'MHz': 1e6, 'GHz': 1e9, 'THz': 1e12}
    
    # Create a list of tuples where each tuple contains the numerical value of the frequency and the original key-value pair
    sorted_list = sorted([(float(k[:-3]) * unit_multipliers[k[-3:]], k, v) for k, v in dict_to_sort.items()])
    
    # Create a new dictionary with the sorted key-value pairs
    sorted_dict = {k: v for _, k, v in sorted_list}
    
    return sorted_dict


def plot_error_bars(*data_dicts):
    fig_width = len(data_dicts) * 5 # set width of figure based on number of data_dicts
    fig, axs = plt.subplots(1, len(data_dicts), figsize=(fig_width, 5), squeeze=False) # create one subplot for each data_dict
    yaxis_list = ["Storage Modulus", "Loss Modulus", "Loss Tangent"]
    for i, data_dict in enumerate(data_dicts):
        x_values = list(data_dict.keys())
        y_values = list(data_dict.values())

        for j, y in enumerate(y_values):
            mean = sum(y) / len(y)
            stdev = (sum([(val-mean)**2 for val in y]) / len(y))**0.5
            max_error = max(y) - min(y)
            axs[0, i].errorbar(x_values[j], mean, yerr=stdev, capsize=5, capthick=1, elinewidth=1, marker='s', markersize=5)
            axs[0, i].text(x_values[j], mean + stdev + 0.1*max_error, f"Max Error: {max_error:.4f}", ha='center', fontsize=6)

            # Add data points to error bars
            for k, val in enumerate(y):
                axs[0, i].plot([x_values[j]], [val], 'ko', markersize=2)

        axs[0, i].set_xlabel('Frequencies')
        axs[0, i].set_ylabel(yaxis_list[i])
    plt.suptitle("Storage Modulus, Loss Modulus, and Loss Tangent Error Analysis")
    plt.show()

        
class DMADataParser:
    def __init__(self, calc_dir):
        self.calc_dir = calc_dir
    
    def pressureparser(self, output_xy=True, plot=False, plot_title="Pressure vs runtime", pressdir=None, apply_savgol_filter=False, savgol_windowlength=10, savgol_polyorder=5, apply_gaussian_filter=False, gaussian_sigma=6, **kwargs):
        simulation_dir = kwargs.get("simulation_dir", self.calc_dir)
        # Function to generate a pressure.txt file with human-readable values for the pressures in all directions over the runtime of the simulation
        press_dict = {"x": "Pxx", "y": "Pyy", "z": "Pzz"}
        os.chdir(simulation_dir)
        if pressdir != None:
            pressdir = pressdir
        else:
            grepPressDir = subprocess.check_output("grep 'remap v' in.*", shell=True).decode("utf-8").split()
            pressdir = grepPressDir[5]
        if "pressure.txt" not in os.listdir():
            part1 = "i=`grep '{}' -n *.log | sed -e 's/:/  /g' | ".format(press_dict[pressdir])
            subprocess.call(part1 + "awk '{print $1}'`", shell=True)
            subprocess.call("awk -v i=$i '{if ( FNR > i && NF == 16) print $0}' *.log > pressure.txt", shell=True)
            #subprocess.call("sed -i 1d pressure.txt", shell=True)
        #df = pd.read_csv("pressure.txt", header=None, delimiter=r"\s+")
        df = pd.read_csv("pressure.txt", skiprows=[0], header=None, delim_whitespace=True)
        timestep = df[13][0]
        if pressdir == "x":
            item = 7
        elif pressdir == "y":
            item = 8
        else:
            item = 9
        runtime = np.array(df[0]) * timestep
        pressure = np.array(df[item]) * - 0.101325 #in MPa
        if apply_savgol_filter:
            pressure = savgol_filter(pressure, savgol_windowlength, savgol_polyorder)
        if apply_gaussian_filter:
            pressure = gaussian_filter1d(pressure, gaussian_sigma)
        if plot:
            plt.title(plot_title)
            plt.xlabel("Runtime (fs)")
            plt.ylabel("Pressure (MPa)")
            plt.ticklabel_format(style="scientific", scilimits=(0,4))
            plt.scatter(runtime, pressure, s=1)
        if output_xy:
            return runtime, pressure, simulation_dir
    
    def fit_sin(self, tt, yy, plot_pressure=False, plot_title="Pressure vs runtime for DMA (+ Scipy Fitting)", plot_color='red', do_prints=False, **kwargs):
        '''Fit sin to the input time sequence, and return fitting parameters "amp", "omega", "phase", "offset", "freq", "period" and "fitfunc"'''
        simulation_dir = kwargs.get("simulation_dir", self.calc_dir)
        os.chdir(simulation_dir)
        tt = np.array(tt)
        yy = np.array(yy)
        ff = np.fft.fftfreq(len(tt), (tt[1]-tt[0]))   # assume uniform spacing
        Fyy = abs(np.fft.fft(yy))
        guess_freq = 1 / float(subprocess.check_output("grep 'wiggle' in.*", shell=True).decode("utf-8").split()[8])
        guess_w = 2 * np.pi * guess_freq
        #guess_freq = abs(ff[np.argmax(Fyy[1:])+1])   # excluding the zero frequency "peak", which is related to offset
        guess_amp = np.std(yy) * 2.**0.5
        guess_offset = np.mean(yy)
        guess = np.array([guess_amp, 0., guess_offset])

        def sinfunc(t, A, p, c):  return A * np.sin(guess_w * t + p) + c
        popt, pcov = sp.optimize.curve_fit(sinfunc, tt, yy, p0=guess)
        A, p, c = popt
        fitfunc = lambda t: A * np.sin(guess_w * t + p) + c
        y = fitfunc(tt)
        if plot_pressure:
            plt.scatter(tt, yy, s=1, label="Pressure data")
            plt.ticklabel_format(style="scientific", scilimits=(0,4))
            plt.title(plot_title)
            plt.xlabel("Runtime (fs)")
            plt.ylabel("Pressure (MPa)")
            plt.plot(tt, y, plot_color, label="Scipy fitted curve")
            plt.legend()
        sin_offset = p
        mae, rmse, r2 = mean_absolute_error(yy, y), mean_squared_error(yy, y, squared=False), r2_score(yy, y)
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
        #return {"amp": A, "omega": w, "phase": p, "offset": c, "freq": f, "period": 1./f, "fitfunc": fitfunc, "maxcov": np.max(pcov), "rawres": (guess,popt,pcov)}
        return {"Fit Curve": y, "Predicted Frequency": guess_w / (2 * np.pi) * 1e6, "Real Phi": sin_offset, "Loss Tangent": np.tan(sin_offset), "Storage Modulus": np.cos(sin_offset), "Loss Modulus": np.sin(sin_offset), "MAE": mae, "RMSE": rmse, "R2": r2}

class DMAErrorDataParser(DMADataParser):
    def error_parser(self, add_savgol_filter=False, add_gaussian_filter=False):
        os.chdir(self.calc_dir)
        os.chdir("frequencies")
        if "noise_filter_run" in os.listdir("."):
            os.chdir("noise_filter_run")
            runtime_nfr, pressure_nfr = self.pressureparser(".", add_savgol_filter=add_savgol_filter, add_gaussian_filter=add_gaussian_filter)
            os.chdir("../")
            freq_directories = os.listdir(".").pop("noise_filter_run")
        else:
            freq_directories = os.listdir(".")
        unique_frequencies = [i.split("_")[0] for i in freq_directories]
        unique_frequencies = set(unique_frequencies)
        SM_dict = sort_dict_by_frequency({i:[] for i in unique_frequencies})
        LM_dict = sort_dict_by_frequency({i:[] for i in unique_frequencies})
        LT_dict = sort_dict_by_frequency({i:[] for i in unique_frequencies})
        for freq in freq_directories:
            frequency = freq.split("_")[0]
            os.chdir(freq)
            runtime, pressure, simulation_dir = self.pressureparser(add_savgol_filter=add_savgol_filter, add_gaussian_filter=add_gaussian_filter, **{"simulation_dir": "."})
            curve_fitting = self.fit_sin(runtime, pressure, **{"simulation_dir": simulation_dir})
            SM_dict[frequency].append(curve_fitting["Storage Modulus"])
            LM_dict[frequency].append(curve_fitting["Loss Modulus"])
            LT_dict[frequency].append(curve_fitting["Loss Tangent"])
            os.chdir("../")
        return SM_dict, LM_dict, LT_dict
    
    def error_plotter(self, SM_dict, LM_dict, LT_dict, title="DMA Error Margin Analysis"):
        plot_error_bars(SM_dict, LM_dict, LT_dict)
            
class DMAMasterCurveDataParser(DMADataParser):
    def master_curve_parser(self, add_savgol_filter=False, add_gaussian_filter=False):
        final_dict_storage, final_dict_loss, final_dict_tangent = {}, {}, {}
        os.chdir(self.calc_dir)
        os.chdir("temperatures")
        for temp_dir in os.listdir("."):
            temp = float(re.sub("[^0-9.]", "", temp_dir))
            final_dict_storage[temp], final_dict_loss[temp], final_dict_tangent[temp] = {}, {}, {}
            os.chdir(temp_dir)
            #if "noise_filter_run" in os.listdir("."):
             #   os.chdir("noise_filter_run")
              #  print(os.listdir("."))
               # runtime_nfr, pressure_nfr, calc_dir_nfr = self.pressureparser(".", pressdir="x", add_savgol_filter=add_savgol_filter, add_gaussian_filter=add_gaussian_filter)
                #os.chdir("../")
            freq_directories = os.listdir(".")
            freq_directories.remove("noise_filter_run")
            #else:
            for freq_dir in freq_directories:
                freq = float(re.sub("[^0-9.]", "", freq_dir)) * 1e9
                os.chdir(freq_dir)
                try:
                    runtime, press, calc_dir = self.pressureparser(apply_savgol_filter=add_savgol_filter, apply_gaussian_filter=add_gaussian_filter, **{"simulation_dir": "."})
                    results = self.fit_sin(runtime, press, **{"simulation_dir": calc_dir})
                    final_dict_storage[temp][freq] = results["Storage Modulus"]
                    final_dict_loss[temp][freq] = results["Loss Modulus"]
                    final_dict_tangent[temp][freq] = results["Loss Tangent"]
                    os.chdir("../")
                except:
                    final_dict_storage[temp][freq] = 0.99
                    final_dict_loss[temp][freq] = 0.1
                    final_dict_tangent[temp][freq] = 0.2
                    os.chdir("../")
            os.chdir("../")
        df_storage, df_loss, df_tangent = pd.DataFrame(final_dict_storage), pd.DataFrame(final_dict_loss), pd.DataFrame(final_dict_tangent)
        df_storage.columns.name, df_loss.columns.name, df_tangent.columns.name = 'Temperature (K)', 'Temperature (K)', 'Temperature (K)'
        df_storage.index.name, df_loss.index.name, df_tangent.index.name = 'Frequency (Hz)', 'Frequency (Hz)', 'Frequency (Hz)'
        df_storage, df_loss, df_tangent = df_storage.sort_index(axis=0), df_loss.sort_index(axis=0), df_tangent.sort_index(axis=0)  # sort rows
        df_storage, df_loss, df_tangent = df_storage.sort_index(axis=1), df_loss.sort_index(axis=1), df_tangent.sort_index(axis=1)  # sort columns
        return df_storage, df_loss, df_tangent
    
    def master_curve_plotter(self, *data):
        for data in data:
            x_data = [np.array(data.index) for i in data.columns]
            y_data = [np.array(data[i]) for i in data.columns]
            x_data = [np.log(xi) for xi in x_data]
            y_data = [np.log(ci) for ci in y_data]
            states = [i for i in np.array(data.columns)]
            mc = MasterCurve(fixed_noise = 0)
            mc.add_data(x_data, y_data, states)
            mc.add_htransform(Multiply())
            mc.add_vtransform(Multiply())
            mc.superpose()
            fig1, ax1, fig2, ax2, fig3, ax3 = mc.plot(colormap = lambda i: plt.cm.Blues_r(i/1.5))
            ax1.set_xscale('log')
            ax1.set_yscale('linear')
            ax2.set_xscale('log')
            ax2.set_yscale('linear')
            ax3.set_xscale('log')
            ax3.set_yscale('linear')
    

