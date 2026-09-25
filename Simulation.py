from Settings import Setting, settings, parameter, NLD
import subprocess
from enum import IntEnum, auto
from pathlib import Path
import numpy as np
import time
import uproot
from Func import func
from dataclasses import dataclass
from typing import Any
from scipy.ndimage import gaussian_filter
from concurrent.futures import ProcessPoolExecutor
import shutil
import time
import matplotlib.pyplot as plt

@dataclass
class Run:
    settings: dict[Setting, Any]
    level_data: tuple[list[float],list[list[float]]]
    root_tree: dict[str, np.ndarray[float]]

class Simulation:
    def __init__(self):
        self.parallel = self.Parallel(self)
        self.pop = self.Population(self)
        self.run_path = settings.std_path / "runs"
        self.settings_file_name = "settings"
        self.createdPopFile_stdname = "\"createdPopFile.dat\""
        self.std_popFileName = "\"Ge76_popEQUAL.dat\""

    def get_level_data(self, run):
        cut = run.find("More levels exist at higher spins")
        #print(run[cut:])
        cut2 = run[cut:].find("E(MeV)")
        first = run[cut+cut2+10:]
        cut3 = first.find("Total Number of Levels")
        second = first[:cut3]
        return second

    def get_level_arrays(self, text):
        energy = []
        spin_array = []
        for k in range(len(text)):
            ch = text[k]
            if ch != '.':
                continue
            a = 1
            while text[k-(a+1)].isdigit():
                a = a+1
            num = float(text[k-a:k+4])
            energy.append(num)
            l = 20
            b = 8
            spin_array_one_energy = [[] for x in range(20)]
            while l > 0:
                if text[k+b] == " ":
                    b = b+1
                c = 1
                #print("b:",b)
                while text[k+b+c] != "|":
                    c = c+1
                #print("c:",c)
                spin_val = text[k+b:k+b+c]
                #print(spin_val)
                spin_val = spin_val.replace("\n", "")
                spin_array_one_energy[l-1] = int(spin_val)
                #print("spin"+str(20-l)+":"+spin_val)
                #print("Spin"+str(19-l)+":"+text[k+b:k+b+c+2])
                b = b+c+1
                l = l-1
            #print(spin_array_one_energy)
            spin_array.append(spin_array_one_energy)
        energy = np.array(energy)
        spin_array = np.array(spin_array)
        spin_array = spin_array.T
        spin_dict = {}
        for k in range(9):
            s = k - 9
            spin_dict[str(s)] = spin_array[k]
        spin_dict["-0"] = spin_array[9]
        spin_dict["+0"] = spin_array[10]
        for k in range(11, 20):
            s = k - 10
            spin_dict[str(s)] = spin_array[k]
        return (energy,spin_dict)   

    def simple_run(self, *, print_setting: bool = True, execution_path: Path = None, parameter_ref: dict[Setting, Any] = None, do_apply_settings: bool = True) -> str:
        if parameter_ref is None:
            parameter_ref = parameter
        if execution_path is None:
            execution_path = settings.root_file_folder
        if parameter_ref[Setting.fluct_bin] != parameter_ref[Setting.g_nConEBin]:
            print("Simple Run: Simulated and fluct binning arent identical.")
        execution_path.mkdir(parents=True, exist_ok=True)
        while True:
            try:
                if do_apply_settings:
                    settings.apply_settings(print_setting=print_setting,parameter_ref=parameter_ref,execution_path=execution_path)
                if not (execution_path / "settings.h").exists():
                    shutil.copy(settings.settings_file_path, execution_path / "settings.h")
                break
            except PermissionError:
                print("Tried to copy settings.h to execution path and failed.")
                time.sleep(2)
        return subprocess.run(["cmd", "/c", "root", str(settings.rainier_path / "modRAINIER.C")], capture_output=True, text=True, cwd=execution_path).stdout

    def run_simulation(self, *, save_path: Path = None, file_name: str = None, print_setting: bool = True, execution_path: Path = None) -> None:
        if file_name == None:
            file_name = "unnamed_run"
        if save_path == None:
            save_path = settings.std_path
        else:
            save_path.mkdir(exist_ok = True, parents=True)
        current_run_folder = save_path / file_name
        current_run_folder.mkdir(exist_ok=True)
        run_text = self.simple_run(print_setting=print_setting,execution_path=execution_path)
        with open(current_run_folder / (file_name+".txt"), "w") as file:
            file.write(run_text)
        with open(current_run_folder / (self.settings_file_name+".txt"), "w") as file:
            for key in parameter:
                file.write(Setting(key).name+" : "+str(parameter[key])+"\n")
        run_path = settings.root_file_folder / "Run0001.root"
        run_path.replace(current_run_folder / (file_name+".root"))

    def run_simulation_then_read(self, save_path: Path = None, file_name: str = None, *, print_setting: bool = True, execution_path: Path = None) -> Run:
        '''
        Executes run_simulation, then reads and returns the run.
        '''
        self.run_simulation(save_path=save_path,file_name=file_name,print_setting=print_setting,execution_path=execution_path)
        run_path = save_path / file_name
        return self.read_run(run_path)

    def run_simulation_events(self, number_of_events: int, *, save_path: Path = None, file_name: str = None, print_setting: bool = True, execution_path: Path = None) -> None:
        '''
        Same as run_simulation() but you can define the number of events as an argument.
        '''
        ev0 = parameter[Setting.g_nEvent]
        parameter[Setting.g_nEvent] = number_of_events
        self.run_simulation(save_path=save_path,file_name=file_name,print_setting=print_setting,execution_path=execution_path)
        parameter[Setting.g_nEvent] = ev0

    def run_simulation_events_then_read(self, number_of_events: int, *, save_path: Path = None, file_name: str = None, print_setting: bool = True, execution_path: Path = None) -> None:
        '''
        Same as run_simulation_then_read() but you can define the number of events as an argument.
        '''
        self.run_simulation_events(number_of_events, save_path=save_path, file_name=file_name, print_setting=print_setting,execution_path=execution_path)
        run_path = save_path / file_name
        return self.read_run(run_path)

    class Parallel:
        def __init__(self, sim):
            self.sim: Simulation = sim
            self.subfolder_name = "PARALLEL"

        def clear_parallel_folder(self):
            shutil.rmtree(settings.parallel_path)

        def run_parallels(self, *, events: int, max_workers: int = 10, save_path: Path = None) -> None:
            '''
            Runs a simulation with a given event number by redistributing the events across parallel simulation on different cpu cores.
            '''
            if save_path is None:
                save_path = settings.this_dir
            partial_events = events // max_workers
            save_folder = save_path / self.subfolder_name
            save_folder.mkdir(parents=True, exist_ok=True)
            worker_folder_name = lambda worker: "Worker "+str(worker+1)
            parameter[Setting.g_nEvent] = partial_events
            settings.apply_settings(print_setting=False)
            for worker in range(max_workers):
                worker_folder = save_folder / worker_folder_name(worker)
                worker_folder.mkdir()
                shutil.copy2(settings.settings_file_path, worker_folder / "settings.h")
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(
                    sim.simple_run,
                    print_setting=False,
                    execution_path=save_folder / worker_folder_name(worker),
                    parameter_ref=parameter,
                    do_apply_settings=False
                ) for worker in range(max_workers)]

                for future in futures:
                    future.result()
            parameter[Setting.g_nEvent] = events
            settings.apply_settings(print_setting=False)

        def collect_root_files(self, *, save_path: Path) -> None:
            parallel_folder = save_path / self.subfolder_name
            ind = 1
            for worker_folder in parallel_folder.iterdir():
                for file in worker_folder.iterdir():
                    if file.name[-5:] == ".root":
                        shutil.copy2(parallel_folder / worker_folder / file.name,
                                    parallel_folder / ("Run"+str(ind)+".root"))
                        ind += 1
                shutil.rmtree(parallel_folder / worker_folder)

        def combine_root_files(self, *, save_path: Path, name1: str, name2:str) -> None:
            subprocess.run([
                "hadd",
                "-f",
                "combined.root",
                name1,
                name2,
            ], cwd=save_path / self.subfolder_name, check=True)

        def add_root_files(self, *, save_path: Path) -> None:
            folder_path = save_path / self.subfolder_name
            file_names = []
            for root_file in folder_path.iterdir():
                file_names.append(root_file.name)
            shutil.copy2(folder_path / file_names[0], folder_path / "combined.root")
            for k in range(1,len(file_names)):
                (folder_path / "combined.root").rename(folder_path / "temp.root")
                self.combine_root_files(save_path=save_path,
                                        name1="temp.root",
                                        name2=file_names[k])
                (folder_path / "temp.root").unlink()

        def delete_remaining_files(self, *, save_path: Path):
            parallel_folder = save_path / self.subfolder_name
            for file in parallel_folder.iterdir():
                if file.name != "combined.root":
                    (parallel_folder / file.name).unlink()

        def save_as_readable(self, *, events: int, root_source: Path, destination: Path = None):
            if destination is None:
                destination = settings.run_folder
            this_run = destination / ("g_nEvent_"+str(events))
            this_run.mkdir(exist_ok=True, parents=True)
            run_text = ""
            with open(this_run / ("g_nEvent_"+str(events)+".txt"), "w") as file:
                file.write(run_text)
            with open(this_run / (sim.settings_file_name+".txt"), "w") as file:
                for key in parameter:
                    file.write(Setting(key).name+" : "+str(parameter[key])+"\n")
            shutil.copy2(root_source, this_run)
            root_source.unlink()

        def run(
                self,
                *,
                events: int,
                max_workers: int,
                save_path: Path
        ):
            parallel_path = settings.this_dir
            self.run_parallels(events=events,max_workers=max_workers,save_path=parallel_path)
            self.collect_root_files(save_path=parallel_path)
            self.add_root_files(save_path=parallel_path)
            self.delete_remaining_files(save_path=parallel_path)
            self.save_as_readable(
                events=events,
                root_source=parallel_path / self.subfolder_name / "combined.root",
                destination=save_path
                )

        def iterate(self, param_range: list, *, save_path: Path, max_workers: int) -> None:
            ev0 = parameter[Setting.g_nEvent]
            for k in range(len(param_range)):
                print(Setting(Setting.g_nEvent).name+": "+str(param_range[k]))
                parameter[Setting.g_nEvent] = param_range[k]
                start_time = time.perf_counter()
                self.run(events=param_range[k],max_workers=max_workers,save_path=save_path)
                time_passed = time.perf_counter() - start_time
                print("time taken: "+str(round(time_passed)))
            parameter[Setting.g_nEvent] = ev0

    def iterate(self, param: Setting, param_range: list, *, save_path: Path = None) -> None:
        if save_path == None:
            save_path = settings.std_path
        param0 = parameter[param]
        for k in range(len(param_range)):
            print(Setting(param).name+": "+str(param_range[k]))
            parameter[param] = param_range[k]
            start_time = time.perf_counter()
            self.run_simulation(save_path = save_path, file_name = str(Setting(param).name)+"_"+str(param_range[k]), print_setting=False)
            time_passed = time.perf_counter() - start_time
            print("time taken: "+str(round(time_passed)))
        parameter[param] = param0

    def get_nld_energy_delta(self, nld_energy): #CAN DO IT DIFFERENTLY THROUGH BIN AND ENERGY INTERVAL
        delta = []
        for k in range(len(nld_energy)-1):
            delta.append(nld_energy[k+1]-nld_energy[k])
        return sum(delta)/len(delta)

    def from_runtext_get_nld_energy_nld_dict(self, run_text):
        nld_energy, nld_dict = self.get_level_arrays(self.get_level_data(run_text))
        h = self.get_nld_energy_delta(nld_energy)
        hinv = 1/h
        for s in nld_dict.keys():
            nld_s = nld_dict[s]
            for k in range(len(nld_s)):
                nld_s[k] *= hinv
        return nld_energy, nld_dict

    def read_run(self, run_folder_path: Path) -> Run:
        settings = {}
        level_data = ([],[[]])
        root_tree = {}
        for file in run_folder_path.iterdir():
            if file.name[-3:] == "txt":
                with open(file) as f:
                    if file.name == self.settings_file_name+".txt":                
                        settings_text = f.readlines()
                        for entry in settings_text:
                            colon = entry.find(":")
                            param = entry[:colon-1]
                            param_value = entry[colon+2:-1]
                            if Setting[param] == Setting.popFile_name:
                                converted = str(param_value)
                            else:
                                converted = float(param_value)
                            settings[Setting[param]] = converted
                    else:
                        run_text = f.read()
                        if run_text != "":
                            level_data = self.from_runtext_get_nld_energy_nld_dict(run_text)
                        else:
                            level_data = ([],{})
            else:
                with uproot.open(file) as f:
                    tree = f["tree;1"]
                    root_tree["JI_int"] = tree["JI_int"].array(library="np")
                    root_tree["Egs"] = tree["Egs"].array(library="np")
                    root_tree["ExI"] = tree["ExI"].array(library="np")
                    root_tree["nPar"] = tree["nPar"].array(library="np")
        return Run(settings, level_data, root_tree)

    def read_folder(self, folder_path: Path) -> list[Run]:
        runs = []
        for file in folder_path.iterdir():
            runs.append(self.read_run(file))
        return runs

    def deep_plus_one(self, state, limits):
        n = len(state)
        for k in range(n):
            i = n - (k+1)
            n_i = limits[i]
            x_i = state[i]
            if x_i + 1 > n_i - 1:
                state[i] = 0
            else:
                state[i] += 1
                break
        return state

    def deep_add(self, value, state, limits):
        while value > 0:
            state = self.deep_plus_one(state, limits)
            value -= 1
        return state

    def iterate_grid(self, params, *, save_path = None):
        if save_path == None:
            save_path = settings.std_path / "runs"
        n = len(params)
        iter_lim = np.zeros(n, dtype=int)
        iter_state = np.zeros(n, dtype=int)
        for k in range(n):
            param, param_range = params[k] #elements in params are tuples of param and the range
            iter_lim[k] = len(param_range)
        combinations = np.prod(iter_lim)
        comb_0 = combinations
        while combinations > 0:
            print(str(comb_0 - combinations) + " / " + str(comb_0))
            for i in range(n):
                param, param_range = params[i]
                parameter[param] = param_range[iter_state[i]]
            self.run_simulation(file_name=str(comb_0 - combinations),save_path=save_path)
            iter_state = self.deep_plus_one(iter_state, iter_lim)
            combinations -= 1

    def helper_match_energy(self, energy, nld_energy, start_ind = 0): #could be made more effeicient with start_ind
        res = 0
        for k in range(start_ind, len(nld_energy)):
            res = k
            if nld_energy[k] >= energy:
                return max(res-1, 0)
        #print("needs bigger nld data to make pop file.")
        return res

    class Population:
        def __init__(self, sim):
            self.sim: Simulation = sim

        def bin_width(self, q: float, energy_bins: float) -> float:
            return q / (energy_bins - 1)

        def ex_energy_list(self, q: float, energy_bins: int) -> list:
            '''
            Bins an energy range from 0 up to q into equally spaced energy bins and returns the bin energies as a list.
            '''
            h = self.bin_width(q, energy_bins)
            ex = lambda bin: h * bin
            return [ex(bin) for bin in range(energy_bins)]

        def phase_space_factor(self, ex: float, q: float) -> float:
            '''
            Calculates beta decay phase space coefficient (Sargent's Law)
            '''
            return pow(q - ex, 5)

        def func(self, ex: float, q: float) -> float:
            '''
            Computes beta decay population probability (not an actual probability, not normalized).
            '''
            return self.phase_space_factor(ex, q) * func.rho(ex, 1)

        def normalized(self, dist: list) -> np.ndarray:
            '''
            Returns the a normalized list (probability distribution).
            ''' 
            total = sum(dist)
            return [val / total for val in dist]

        def dist(self, ex_energies: list, q: float) -> list:
            return [self.func(ex, q) for ex in ex_energies]

        def dist_normed(self, ex_energies: list, q: float):
            return self.normalized(self.dist(ex_energies, q))

        def make_file(self, *, q: float, energy_bins: int) -> None:
            '''
            Creates a beta decay like population file. Applies an experimental resolution by default, can be disabled.
            '''
            if not energy_bins > 1:
                raise ValueError("energy_bins have to be more than 1.")
            spins = range(10) #this is a fixed range. i know, thats a limitation for the program but it works like this right now.
            lines = []
            sp = " "
            signs = ["-", "+"]
            overhead = "bin Ex Popul. "
            for spin in spins:
                for sign in signs:
                    overhead += str(spin) + sign + sp
            overhead += "\n" + "\n"
            lines.append(overhead)

            ex_energies = self.ex_energy_list(q, energy_bins)
            pop_dist_normed = self.dist_normed(ex_energies, q)

            h = self.bin_width(q, energy_bins)
            parameter[Setting.g_nExPopI] = energy_bins
            parameter[Setting.g_dExRes] = h
            parameter[Setting.popFile_name] = f'"{str(settings.popfile_path).replace("\\", "\\\\")}"'
            parameter[Setting.g_dExIMax] = q
            settings.apply_settings(print_setting=False)

            
            for k in range(energy_bins):
                line = np.zeros(3 + 2*len(spins))
                line[0] = k #bin
                line[1] = ex_energies[k] #ex
                line[2] = 0 #popul.
                for s in spins:
                    for sign in signs:
                        if s == 1 and sign == "-":
                            if ex_energies[k] <= q:
                                line[5] = pop_dist_normed[k] * 100
                                line[2] = line[5]
                            else:
                                line[5] = 0
                        else:
                            if sign == "-1":
                                sep = 0
                            else:
                                sep = 1
                            line[3 + 2*s + sep] = 0
                lines.append(line)

            for i in range(1, len(lines)): #starts from 1 because of overhead
                line = lines[i]
                linestr = ""
                linestr += str(int(line[0])) + sp + str(line[1]) + sp + str(line[2]) + sp#str(round(line[1], 3))
                for s in spins:
                    for sign in signs:
                        if s == 1 and sign == "-":
                            linestr += str(line[5]) + sp
                        else:
                            if sign == "-1":
                                sep = 0
                            else:
                                sep = 1
                            linestr += str(line[3 + 2*s + sep]) + sp
                linestr += "\n"
                lines[i] = linestr

            settings.popfile_path.parent.mkdir(exist_ok=True, parents=True)
            with open(settings.popfile_path, "w") as f:
                f.writelines(lines)

sim = Simulation()

#sim.make_pop_file(7.5, 500)
#sim.make_pop_file()
# z = 32
# nmin = 40
# nmax = 60
# Z_range = [z]
# A_range = range(nmin+z,nmax+z+1)

# parameter[Setting.g_nEvent] = 1
# parameter[Setting.g_nConEBin] = 500

# sim.iterate_grid([(Setting.g_nZ, Z_range),
#                   (Setting.g_nAMass, A_range)])


#parameter[Setting.g_nEvent] = 0
#sim.run_simulation(file_name="ExpResRun1")
