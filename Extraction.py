import numpy as np
from enum import IntEnum, auto
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import uproot
from Settings import Setting, settings, parameter
from Simulation import Run, sim
from scipy.stats import norm
from enum import Enum
from Func import ResolutionMode

class Extraction:
    def __init__(self):
        pass

    def apply_resolution(self, energy: list[float], data: list[float], exp_res: float, res_mode: ResolutionMode) -> list:
        h = np.mean(np.diff(energy)) #bin width
        energy_left = [en - h/2 for en in energy]
        energy_edges = energy_left + [max(energy_left)+h]
        bin_center = lambda i: energy_edges[i] + h/2
        smeared = np.zeros(len(data)) #initializing smeared list
        match res_mode:
            case ResolutionMode.const:
                sigma = lambda E: exp_res
            case ResolutionMode.percentage:
                sigma = lambda E: 0.01*E
        for j in range(len(data)):
            src_energy = bin_center(j)
            weights = (
                norm.cdf(energy_edges[1:], loc=src_energy, scale=sigma(src_energy))
                -
                norm.cdf(energy_edges[:-1], loc=src_energy, scale=sigma(src_energy))
            )
            smeared += data[j] * weights
        return smeared

    def get_total_counts(
            self,
            run: Run
    ) -> tuple:
        q = settings.Q_76Ga
        energy_bins = 7000
        ExI = run.root_tree["ExI"]
        bin_energy = lambda k: q * k / (energy_bins - 1)
        find_bin_ind = lambda en: int(np.floor(en*(energy_bins-1)/q))
        bin_list = [0.0 for _ in range(energy_bins)]
        for ex in ExI:
            bin_list[find_bin_ind(ex)] += 1
        return bin_list

    def get_just_energies(
            self,
            run: Run
    ) -> dict:
        ExI = run.root_tree["ExI"]
        data_dict = {}
        for ex in ExI:
            data_dict[ex] = 0
        for ex in ExI:
            data_dict[ex] += 1
        return data_dict

    def pop_EJ(self, run: Run, *, spin_spacing = 1):
        energy_bins = parameter[Setting.fluct_bin]
        tree = run.root_tree
        JI_int = tree["JI_int"]
        ExI = tree["ExI"]
        nPar = tree["nPar"]
        JI_int = JI_int.copy()
        for k in range(len(JI_int)):
            if nPar[k] == 0:
                JI_int[k] = -JI_int[k]
        spin_min = JI_int.min()
        spin_max = JI_int.max()
        spin_axis = np.arange(spin_min, spin_max + 2*spin_spacing, spin_spacing) #2*spin_spacing works even though 1* should work no?
        energy_axis = np.array(sim.pop.ex_energy_list(settings.Q_76Ga, energy_bins))
        return np.histogram2d(JI_int, ExI, bins=[spin_axis, energy_axis])

    def get_fluct_data(self, run: Run, *, plot: bool = False) -> tuple[np.ndarray, np.ndarray]:
        counts, spin_edges, energy_edges = self.pop_EJ(run)
        if plot:
            plt.figure()
            im = plt.pcolormesh(spin_edges, energy_edges, counts.T, shading="auto", norm=LogNorm())
            plt.colorbar(im, label="Number of events")
            plt.title("EJ counts")
            plt.xlabel("J[hbar]")
            plt.ylabel("Ex[MeV]")
        energy = 1/2 * (energy_edges[1:] + energy_edges[:-1])
        fluct_data_dict = {}
        spins = spin_edges[:-1]
        for k in range(len(spins)):
            spin = spins[k]
            fluct_data_dict[int(spin)] = counts[k].tolist()
        return (energy.tolist(), fluct_data_dict[-1])

    def spectrum_smeared(self, run: Run, *, exp_res: float = 0.001, plot: bool = False, res_mode: ResolutionMode) -> tuple[list, list]:
        energy, data = self.get_fluct_data(run, plot=plot)
        if exp_res == 0:
            return energy, data
        return energy, self.apply_resolution(energy, data, exp_res,res_mode=res_mode)
        

extract = Extraction()