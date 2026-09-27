import numpy as np
from enum import IntEnum, auto, Enum
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
from pathlib import Path
import subprocess
from Settings import settings, Setting, parameter
from Func import func, ResolutionMode
from dataclasses import dataclass
from typing import Any
from functools import singledispatchmethod
from Simulation import Run, sim
from Extraction import extract
import shutil
from scipy.optimize import curve_fit
import bisect

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
    fit_data: tuple
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
        #plt.title("Input spectrum")

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
        plt.xlim(4.5,5)
        plt.plot(energy, rough)
        if log_scale:
            plt.yscale("log")
        plt.xlabel("E in MeV")
        plt.ylabel("counts")
        if label != "":
            plt.legend()
        #plt.title("Rough/fine smoothing")

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
        #plt.title("Stationary spectrum")
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
        #plt.title("Autocorrelation function")
        plt.xlim(0,0.5) 
        if not series:
            plt.savefig(save_path / settings.folder_autocorr_name / file_name, dpi=300)
            plt.close()

    def pop_dist_comp(self, run: Run, data_energy, nld):
        if run is not None:
            pop_dist = sim.pop.dist_normed(data_energy, settings.Q_76Ga)
            scalar = 1e3/max(pop_dist)
            
            scaled_pop_dist = [prob * scalar/100 for prob in pop_dist]
            plt.plot(data_energy, scaled_pop_dist, color="grey", alpha=0.67, linestyle="-.", label="population") #JUST TEMPORARY THE Q VALUE REMEMBER

    def comparison(
            self,
            energy_range: list,
            nld: list,
            data_energy: list,
            *,
            run: Run = None,
            label: str = "",
            fit_data: tuple = None,
            ) -> None:
        self.pop_dist_comp(run, data_energy, nld)
        plt.plot(data_energy, [func.rho(e) for e in data_energy], color="blue", label="CT model")
        plt.scatter(energy_range, nld, marker="^", label="NLD for "+label)
        if fit_data is not None:
            exp_model = lambda E: 10 ** fit_data[0][1] * np.exp(fit_data[0][0]*E*np.log(10))
            plt.plot(energy_range, [exp_model(E) for E in energy_range], linestyle="dashed", color="red", alpha = 0.75, label="fit")
            # exp_model = lambda E: 10 ** fit_data[0][1] * np.exp(fit_data[0][0]*E*np.log(10))
            # plt.plot(energy_range, [exp_model(E) for E in energy_range], linestyle="dashed", color="red", alpha = 0.75)
        plt.xlim(0,7)
        plt.ylim(1e-2,1e8)
        if label != "":
            plt.legend()
            #plt.legend(loc="center left",bbox_to_anchor=(1.00, 0.5))
        plt.yscale("log")
        #plt.title("CT model and extracted NLD")
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
            log_scale: bool = False,
            fit_data: tuple = None
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
                self.comparison(energy_range, extract_nld, fluct_energy, run=run, label=label, fit_data=fit_data)

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

    def plot_values_set(self):
        plt.rcParams.update({
            # Font
            "font.family": "serif",
            "font.size": 10,

            # Axes
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,

            # Legend
            "legend.fontsize": 9,

            # Lines
            "lines.linewidth": 1.6,
            "lines.markersize": 5,

            # Figure
            "figure.dpi": 300,

            # Saving
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
        })

    def create(
        self,
        runs: list[Run],
        iter_setting: Setting,
        *,
        figsize: tuple[float, float] = (6,3),
        res_mode: ResolutionMode,
        plot_fit_line: bool = False
        ) -> None:
        self.init_folders()
        run_fa = [fluc.fluctuation_analysis(run,res_mode=res_mode) for run in runs]
        for k in range(len(runs)):
            run = runs[k]
            fa = run_fa[k]
            if plot_fit_line:
                pass
                #print(self.label(iter_setting, run.settings[iter_setting])+", deviation: ",fluc.deviation(fa))
            for step in FaStep:
                plt.figure(figsize=figsize)
                fit_data = fa.fit_data
                if not plot_fit_line:
                    fit_data = None
                self.helper_seriesplot(
                    fa,
                    step,
                    label=self.label(iter_setting, run.settings[iter_setting]),
                    run=run,
                    fit_data=fit_data
                    )
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
                    fit_data = fa.fit_data
                    if not plot_fit_line:
                        fit_data = None
                    self.helper_seriesplot(
                        fa,
                        step,
                        label=self.label(iter_setting,run.settings[iter_setting]),
                        run=run,
                        fit_data=fit_data
                        )
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
            res_mode: ResolutionMode,
            plot_fit_line: bool = False
            ) -> None:
        self.init_folders()
        if plot_single:
            for param in iter_range:
                param0 = parameter[iter_setting]
                parameter[iter_setting] = param
                fa = fluc.fluctuation_analysis(run,res_mode=res_mode)
                for step in FaStep:
                    plt.figure(figsize=figsize)
                    fit_data = fa.fit_data
                    if not plot_fit_line:
                        fit_data = None
                    self.helper_seriesplot(
                        fa,
                        step,
                        label=self.label(iter_setting,param),
                        run=run,
                        fit_data=fit_data
                        )
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
                if step == FaStep.comparison:
                    print(self.label(iter_setting, param)+", deviation: ",fluc.deviation(fa))
                fit_data = fa.fit_data
                if not plot_fit_line:
                    fit_data = None
                self.helper_seriesplot(
                    fa,
                    step,
                    label=self.label(iter_setting,param),
                    run=run,
                    fluct_data_hide_exact_data=True,
                    fit_data=fit_data
                    )
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

    def deviation(
            self,
            runs: list[Run],
            *,
            res_mode: ResolutionMode
    ) -> None:
        event_axis, dev_axis = fluc.deviation_axes(runs, res_mode=res_mode)
        plt.scatter([event * 1e-6 for event in event_axis], [dev * 100 for dev in dev_axis],marker="x",color="black")
        plt.xlabel("Events [10^6]")
        plt.ylabel("Deviation from model [%]")
        plt.savefig(settings.std_path / FluctuationAnalysis.fa_step_to_folder_name[FaStep.comparison] / "deviation.png", dpi=300)

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

    def get_smooth(self, energy, fluct_data, *, res_mode: ResolutionMode):
        np_energy = np.asarray(energy, dtype=float)
        np_data = np.asarray(fluct_data, dtype=float)
        sigma = func.sigma_res(energy,res_mode)
        dE = np_energy[1] - np_energy[0]
        delta_E = np_energy[:, None] - np_energy[None, :]
        for k in range(len(sigma)):
            sigma[k] = max(sigma[k], 0.003)
        G_fine = np.exp(-0.5 * (delta_E / sigma[None, :])**2)
        G_fine /= np.sqrt(2 * np.pi) * sigma[None, :]

        smeared_fine = G_fine @ np_data * dE

        sigma_rough = sigma * 3

        G_rough = np.exp(-0.5 * (delta_E / sigma_rough[None, :])**2)
        G_rough /= np.sqrt(2 * np.pi) * sigma_rough[None, :]

        smeared_rough = G_rough @ np_data * dE
        return smeared_fine, smeared_rough

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

    def get_avg_level_spacing(self, energy, stationary, Emin, Emax, res_mode: ResolutionMode):
        alpha = parameter[Setting.alpha_parameter]
        c_min_1 = self.autocorr_zero(energy, stationary, Emin, Emax) - 1
        sigma_arr = func.sigma_res(energy, res_mode)
        sigma = sigma_arr[bisect.bisect_left(energy, (Emin+Emax) / 2)]
        return c_min_1 * 2 * sigma * np.sqrt(np.pi) / alpha

    def get_level_density(self, energy, stationary, Emin, Emax, res_mode: ResolutionMode):
        return 1 / self.get_avg_level_spacing(energy, stationary, Emin, Emax, res_mode)

    def get_energy_width(self, energy):
        return np.mean(np.diff(energy))

    def get_fit(
            self,
            energy_range: list,
            nld: list,
            *,
            fit_range: tuple = (2.5, 4.5)
            ) -> tuple:
        model = lambda E, A, C: A*E + C
        p0 = (1.0,-1.0)
        energy_array = np.array(energy_range)
        log_nld = [np.log10(nld_val) for nld_val in nld]
        log_nld = np.array(log_nld)
        mask = np.isfinite(log_nld) & (energy_array >= fit_range[0]) & (energy_array <= fit_range[1])
        x = energy_array[mask]
        y = log_nld[mask]
        opt, cov = curve_fit(model, x, y, p0=p0)
        print(opt)
        return opt, cov

        

    @singledispatchmethod
    def fluctuation_analysis(
        self,
        energy: list,
        fluct_data: list,
        res_mode: ResolutionMode
        ) -> FluctuationAnalysisResult:
        fine, rough = self.get_smooth(energy, fluct_data, res_mode=res_mode)
        stationary = fine/rough

        E_start = parameter[Setting.analysis_E_start]
        E_end = parameter[Setting.analysis_E_end]
        sliding_window_shift = parameter[Setting.sliding_window_E_shift]
        E_step = parameter[Setting.analysis_E_step]
        energy_range = np.arange(E_start + E_step/2, E_end - E_step/2, sliding_window_shift)
        nld_list = []
        for E_val in energy_range:
            nld_list.append(
                self.get_level_density(energy, stationary, E_val - E_step/2, E_val + E_step/2,res_mode)
            )
        fit_data = self.get_fit(energy_range,nld_list)
        return FluctuationAnalysisResult(energy,fluct_data,fine,rough,stationary,energy_range,nld_list,fit_data,parameter)

    @fluctuation_analysis.register
    def _(
        self,
        run: Run,
        *,
        res_mode: ResolutionMode
        ):
        energy, smeared_data = extract.spectrum_smeared(run,plot=False,exp_res=parameter[Setting.exp_resolution],res_mode=res_mode)
        return self.fluctuation_analysis(energy, smeared_data,res_mode=res_mode)

    @singledispatchmethod
    def deviation(self, slope: float, error: float):
        return abs((1/(func.CTM_temp*np.log(10))) - slope)/ (1/(func.CTM_temp*np.log(10)))

    @deviation.register
    def _(self, fa: FluctuationAnalysisResult):
        return self.deviation(fa.fit_data[0][0], fa.fit_data[1][0][0])

    @deviation.register(list)
    def _(self, fa_list: list[FluctuationAnalysisResult]):
        return [self.deviation(fa) for fa in fa_list]

    def iterate(self, energy, fluct_data, nld_energy, nld, param, param_range):
        param0 = parameter[param]
        result = []
        for k in range(len(param_range)):
            print(Setting(param).name+": "+str(param_range[k]))
            parameter[param] = param_range[k]
            result.append(self.fluctuation_analysis(energy, fluct_data, nld_energy, nld))
        parameter[param] = param0
        return result

    def slope_deviations(
            self,
            runs: list[Run],
            *,
            res_mode: ResolutionMode
    ) -> list[float]:
        fas = [self.fluctuation_analysis(run,res_mode=res_mode) for run in runs]
        devs = [self.deviation(fa) for fa in fas]
        return devs

    def deviation_axes(
            self,
            runs: list[Run],
            *,
            res_mode: ResolutionMode
    ) -> tuple[list, list]:
        event_axis = [run.settings[Setting.g_nEvent] for run in runs]
        dev_axis = self.slope_deviations(runs,res_mode=res_mode)
        return (event_axis, dev_axis)


fluc = FluctuationAnalysis()
fluc.plot.plot_values_set()