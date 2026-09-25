import numpy as np
from enum import IntEnum, auto, Enum
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
from Settings import settings, Setting, parameter
from Func import func
from dataclasses import dataclass
from typing import Any
from functools import singledispatchmethod
from Simulation import Run, sim
from Extraction import extract, ResolutionMode
import shutil
from scipy.optimize import curve_fit

class FaStep(IntEnum):
    fluct = auto()
    smoothing = auto()
    stationary = auto()
    autocorr = auto()
    comparison = auto()

@dataclass
class FluctuationAnalysisResult:
    fluct_energy: np.ndarray
    fluct_data: np.ndarray
    fine: np.ndarray
    rough: np.ndarray
    stationary: np.ndarray
    energy_range: np.ndarray
    nld: np.ndarray
    settings: dict[Setting, Any]

class FluctuationAnalysisPlot:
    def __init__(self):
        self.linewidth = 1
        self.param_name: dict = {
            Setting.g_nEvent : "Events",
            Setting.exp_resolution : "Res."
        }

    def fluct_data(
            self,
            energy: list,
            fluct_data: list,
            *,
            hide_exact_data: bool = False,
            run: Run = None,
            label: str = "",
            log_scale: bool = False
            ) -> None:
        plt.step(energy, fluct_data, lw=self.linewidth, label=label)
        if log_scale:
            plt.yscale("log")
        if not hide_exact_data and run is not None:
            energy, data = extract.get_fluct_data(run)
            plt.plot(energy, data, color="grey", alpha=0.5)
        plt.xlabel("E / MeV")
        plt.ylabel("counts")
        if label != "":
            plt.legend()
        plt.title("Input spectrum")

    def smooth(
            self,
            energy: list,
            fluct_data: list,
            fine: list,
            rough: list,
            *,
            run: Run = None,
            label: str = "",
            log_scale: bool = False
            ) -> None:
        plt.plot(energy, fluct_data, label=label)
        plt.plot(energy, fine)
        plt.plot(energy, rough)
        if log_scale:
            plt.yscale("log")
        plt.xlabel("E in MeV")
        plt.ylabel("counts")
        if label != "":
            plt.legend()
        plt.title("Rough/fine smoothing")

    def stationary(
            self,
            energy: list,
            stationary: list,
            *,
            run: Run = None,
            label: str = ""
            ) -> None:
        plt.xlabel("E in MeV")
        plt.ylabel("relative fluctuations")
        plt.title("Stationary spectrum")
        plt.ylim(0.5,1.5)
        plt.plot(energy, stationary, label=label)
        if label != "":
            plt.legend()

    def autocorrelation(self, eps_start, eps_end, eps_step, energy, full_data, save_path, file_name, series = True, *, interval_low = 5, interval_high = 6):
        epsilons = np.arange(eps_start,eps_end,eps_step)
        if not series:
            plt.figure()
        plt.plot(epsilons, [self.autocorr(eps, energy, full_data, interval_low, interval_high) for eps in epsilons])
        plt.xlabel("epsilon")
        plt.ylabel("Autocorr(epsilon)")
        plt.title("Autocorrelation function")
        plt.xlim(0,0.5) 
        if not series:
            plt.savefig(save_path / settings.folder_autocorr_name / file_name, dpi=300)
            plt.close()

    def pop_dist_comp(self, run: Run, data_energy, nld):
        if run is not None:
            pop_dist = sim.pop.dist_normed(data_energy, settings.Q_76Ga)
            scalar = 1e3/max(pop_dist)
            
            scaled_pop_dist = [prob * scalar for prob in pop_dist]
            plt.plot(data_energy, scaled_pop_dist, color="grey", alpha=0.8, linestyle="-.") #JUST TEMPORARY THE Q VALUE REMEMBER

    def comparison(
            self,
            energy_range: list,
            nld: list,
            data_energy: list,
            *,
            run: Run = None,
            label: str = ""
            ) -> None:
        self.pop_dist_comp(run, data_energy, nld)
        plt.plot(data_energy, [func.rho(e) for e in data_energy], color="blue")
        plt.scatter(energy_range, nld, marker="^", label=label)
        plt.ylim(1e-2,1e8)
        if label != "":
            plt.legend(loc="center left",bbox_to_anchor=(1.00, 0.5))
        plt.yscale("log")
        plt.title("CT model and extracted NLD")
        plt.xlabel("E / MeV")
        plt.ylabel("# levels per MeV")

    def helper_seriesplot(
            self,
            fa: FluctuationAnalysisResult,
            fa_step: FaStep,
            *,
            label: str = "no label",
            run: Run = None,
            fluct_data_hide_exact_data: bool = False,
            log_scale: bool = False
            ) -> None:
        fluct_energy = fa.fluct_energy
        fluct_data = fa.fluct_data
        fine = fa.fine
        rough = fa.rough
        stationary = fa.stationary
        energy_range = fa.energy_range
        extract_nld = fa.nld
        match fa_step:
            case FaStep.fluct:
                self.fluct_data(
                    fluct_energy,
                    fluct_data,
                    run=run,
                    label=label,
                    hide_exact_data=fluct_data_hide_exact_data,
                    log_scale=log_scale
                    )
            case FaStep.smoothing:
                self.smooth(
                    fluct_energy,
                    fluct_data,
                    fine,
                    rough,
                    run=run,
                    label=label,
                    log_scale=log_scale
                    )
            case FaStep.stationary:
                self.stationary(fluct_energy, stationary, run=run, label=label)
            case FaStep.autocorr:
                pass
            case FaStep.comparison:
                self.comparison(energy_range, extract_nld, fluct_energy, run=run, label=label)

    def init_folders(self) -> None:
        (settings.std_path/settings.folder_fluct_name).mkdir(exist_ok=True, parents=True)
        (settings.std_path/settings.folder_smoothing_name).mkdir(exist_ok=True, parents=True)
        (settings.std_path/settings.folder_stationary_name).mkdir(exist_ok=True, parents=True)
        (settings.std_path/settings.folder_autocorr_name).mkdir(exist_ok=True, parents=True)
        (settings.std_path/settings.folder_comparison_name).mkdir(exist_ok=True, parents=True)

    def clear_folders(self) -> None:
        shutil.rmtree(settings.std_path/settings.folder_fluct_name, ignore_errors=True)
        shutil.rmtree(settings.std_path/settings.folder_smoothing_name, ignore_errors=True)
        shutil.rmtree(settings.std_path/settings.folder_stationary_name, ignore_errors=True)
        shutil.rmtree(settings.std_path/settings.folder_autocorr_name, ignore_errors=True)
        shutil.rmtree(settings.std_path/settings.folder_comparison_name, ignore_errors=True)



    def label(self, iter_setting: Setting, value: Any):
        if func.isint(value):
            modified = f"{int(value):,}"
        elif func.isfloat(value):
            modified = str(int(value*1e3))+"keV"
        else:
            print("cant find label for that value type")
        if iter_setting in self.param_name:
            setting_name = self.param_name[iter_setting]
        else:   
            setting_name = iter_setting.name
        return setting_name+"="+modified

    def create(
        self,
        runs: list[Run],
        iter_setting: Setting,
        *,
        figsize: tuple[float, float] = (6,3.7),
        res_mode: ResolutionMode
        ) -> None:
        self.init_folders()
        run_fa = [fluc.fluctuation_analysis(run,res_mode=res_mode) for run in runs]
        for k in range(len(runs)):
            run = runs[k]
            fa = run_fa[k]
            for step in FaStep:
                plt.figure(figsize=figsize)
                self.helper_seriesplot(fa, step, label=self.label(iter_setting, run.settings[iter_setting]), run=run)
                plt.savefig(
                    settings.std_path / FluctuationAnalysis.fa_step_to_folder_name[step] / (self.label(iter_setting,run.settings[iter_setting])+".png"),
                    dpi = 300,
                    bbox_inches = "tight")
                plt.close()
        if len(runs) > 1:
            for step in FaStep:
                plt.figure(figsize=figsize)
                for k in range(len(runs)):
                    run = runs[k]
                    fa = run_fa[k]
                    self.helper_seriesplot(fa, step, label=self.label(iter_setting,run.settings[iter_setting]), run=run)
                plt.savefig(
                    settings.std_path / FluctuationAnalysis.fa_step_to_folder_name[step] / ("combined"+".png"),
                    dpi = 300,
                    bbox_inches = "tight")
                plt.close()

    def from_val_get_valid_file_name(self, val: float):
        pass



    def iter_analysis_param(
            self,
            run: Run,
            iter_setting: Setting,
            iter_range: list,
            *,
            plot_single: bool = True,
            figsize: tuple = (6,3.7),
            res_mode: ResolutionMode
            ) -> None:
        self.init_folders()
        if plot_single:
            for param in iter_range:
                param0 = parameter[iter_setting]
                parameter[iter_setting] = param
                fa = fluc.fluctuation_analysis(run,res_mode=res_mode)
                for step in FaStep:
                    plt.figure(figsize=figsize)
                    self.helper_seriesplot(fa, step, label=self.label(iter_setting,param), run=run)
                    plt.savefig(
                        settings.std_path / FluctuationAnalysis.fa_step_to_folder_name[step] / (self.label(iter_setting,param)+".png"),
                        dpi = 300,
                        bbox_inches = "tight")
                    plt.close()
                parameter[iter_setting] = param0
        for step in FaStep:
            plt.figure(figsize=figsize)
            for param in iter_range:
                param0 = parameter[iter_setting]
                parameter[iter_setting] = param
                fa = fluc.fluctuation_analysis(run,res_mode=res_mode)
                self.helper_seriesplot(fa, step, label=self.label(iter_setting,param), run=run, fluct_data_hide_exact_data=True)
                parameter[iter_setting] = param0
            plt.savefig(
                settings.std_path / FluctuationAnalysis.fa_step_to_folder_name[step] / ("combined"+".png"),
                dpi = 300,
                bbox_inches = "tight")
            plt.close()

    def create_by_fa(
            self,
            fa: FluctuationAnalysisResult,
            *,
            run: Run | list[Run] = None,
            file_name: str,
            log_scale: bool = False,
            figsize: tuple = (6,3.7)
            ) -> None:
        if isinstance(fa, FluctuationAnalysisResult):
            single = True
        else:
            single = False
        self.init_folders()
        for step in FaStep:
            plt.figure(figsize=figsize)
            if single:
                self.helper_seriesplot(fa, step, label=file_name, run=run, log_scale=log_scale)
            else:
                for k in range(len(fa)):
                    self.helper_seriesplot(fa, step, label=file_name, run=run, log_scale=log_scale)
            plt.savefig(settings.std_path / FluctuationAnalysis.fa_step_to_folder_name[step] / (file_name+".png"), dpi = 300, bbox_inches = "tight")
            plt.close()

class FluctuationAnalysis:
    fa_step_to_folder_name: dict[FaStep, str] = {
        FaStep.fluct : settings.folder_fluct_name,
        FaStep.smoothing : settings.folder_smoothing_name,
        FaStep.stationary : settings.folder_stationary_name,
        FaStep.autocorr : settings.folder_autocorr_name,
        FaStep.comparison : settings.folder_comparison_name
    }

    def __init__(self):
        self.plot = FluctuationAnalysisPlot()

    def get_smooth(self, fluct_data, bin_width, sigma_fine, sigma_rough):
        np_data = np.array(fluct_data)
        fine = gaussian_filter(np_data, sigma=sigma_fine/bin_width)
        rough = gaussian_filter(np_data, sigma=sigma_rough/bin_width)
        return fine, rough

    def get_interval2(self, energy, myData, lower, upper):
        mask = (energy >= lower) & (energy <= upper)
        return myData[mask]
    
    def autocorr(self, x, energy, full_data, lower, upper, *, print_cut = True):
        h1 = self.get_interval(energy, full_data, lower, upper)
        h2 = self.get_interval(energy, full_data, lower+x, upper+x)
        c1 = 0
        c2 = 0
        while h1.size > h2.size:
            c1 += 1
            h1 = h1[:-1]
        while h1.size < h2.size:
            c2 += 1
            h2 = h2[:-1]
        if print_cut:
            if c1 > 1:
                print("Autocorrelation 1: cut", c1, " out of ",len(h1))
            if c2 > 1:
                print("Autocorrelation 1: cut", c2, " out of ",len(h2))
        return np.mean(h1*h2)/(np.mean(h1)*np.mean(h2))

    def autocorr_zero(self, energy, stationary, Emin, Emax):
        d = self.get_interval2(energy, stationary, Emin, Emax)
        return np.mean(d**2) / (np.mean(d))**2

    def get_avg_level_spacing(self, energy, stationary, Emin, Emax, sigma):
        alpha = parameter[Setting.alpha_parameter]
        c_min_1 = self.autocorr_zero(energy, stationary, Emin, Emax) - 1
        return c_min_1 * 2 * sigma * np.sqrt(np.pi) / alpha

    def get_level_density(self, energy, stationary, Emin, Emax, sigma):
        return 1 / self.get_avg_level_spacing(energy, stationary, Emin, Emax, sigma)

    def get_energy_width(self, energy):
        return np.mean(np.diff(energy))

    def get_fit_slope(
            self,
            energy_range: list,
            nld: list
            ) -> tuple:
        #ways: directly fit exp func
        #or log data and fit line
        # model = lambda E, c, b, a, d: c*np.exp((E-b)/a) + d
        # p0 = []
        log_nld = [np.log10(nld_val) for nld_val in nld]
        model = lambda E, sl, off: sl*E + off
        opt, cov = curve_fit(model, energy_range, log_nld)
        slope = opt[0]
        return slope
        

    @singledispatchmethod
    def fluctuation_analysis(
        self,
        energy: list,
        fluct_data: list
        ) -> FluctuationAnalysisResult:

        bin_width = self.get_energy_width(energy)
        # sigma_fine = 0.003
        # sigma_rough = 0.005
        sigma_fine = parameter[Setting.exp_resolution] / 3
        sigma_rough = sigma_fine * 5

        fine, rough = self.get_smooth(fluct_data, bin_width, sigma_fine, sigma_rough)
        stationary = fine/rough

        E_start = parameter[Setting.analysis_E_start]
        E_end = parameter[Setting.analysis_E_end]
        sliding_window_shift = parameter[Setting.sliding_window_E_shift]
        E_step = parameter[Setting.analysis_E_step]
        energy_range = np.arange(E_start + E_step/2, E_end - E_step/2, sliding_window_shift)
        nld_list = []
        for E_val in energy_range:
            nld_list.append(
                self.get_level_density(energy, stationary, E_val - E_step/2, E_val + E_step/2, sigma_fine)
            )
        return FluctuationAnalysisResult(energy,fluct_data,fine,rough,stationary,energy_range,nld_list,parameter)

    @fluctuation_analysis.register
    def _(
        self,
        run: Run,
        *,
        res_mode: ResolutionMode
        ):
        energy, smeared_data = extract.spectrum_smeared(run,plot=False,exp_res=parameter[Setting.exp_resolution],res_mode=res_mode)
        return self.fluctuation_analysis(energy, smeared_data)

    def iterate(self, energy, fluct_data, nld_energy, nld, param, param_range):
        param0 = parameter[param]
        result = []
        for k in range(len(param_range)):
            print(Setting(param).name+": "+str(param_range[k]))
            parameter[param] = param_range[k]
            result.append(self.fluctuation_analysis(energy, fluct_data, nld_energy, nld))
        parameter[param] = param0
        return result

    def plot_iteration(self, fa_collection):
        for k in range(len(fa_collection)):
            self.plot_fluctuation_analysis(fa_collection[k],file_name=str(k))




fluc = FluctuationAnalysis()