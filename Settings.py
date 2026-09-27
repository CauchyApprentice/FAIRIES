from enum import IntEnum, auto
from pathlib import Path
from typing import Any
import shutil

class Setting(IntEnum):
    g_nReal = auto()
    g_nZ = auto()
    g_nAMass = auto()
    g_nConEBin = auto()
    g_nEvent = auto()
    g_nConSpbMax = auto()
    g_nDisLvlMax = auto()
    g_dExIMax = auto()
    g_dExRes = auto()
    g_nExPopI = auto()
    popFile_name = auto()
    sim_bin_width = auto()

    exp_resolution = auto()
    alpha_parameter = auto()
    sigma_fine = auto()
    
    fluct_bin = auto()

    analysis_E_step = auto()
    analysis_E_start = auto()
    analysis_E_end = auto()
    sliding_window_E_shift = auto()

    nld_model = auto()
    exci_mode = auto()

class NLD(IntEnum):
    CTM = auto()            #NLD models
    BSFG = auto()
    Table = auto()
    UsrDef = auto()

    CTM_g_dTemp = auto()    #parameters for the models
    CTM_dE0 = auto()
    BSFG_g_dE1 = auto()

class EXCI(IntEnum):
    single = auto()
    select = auto()
    spread = auto()
    full_rxn = auto()

class Settings:
    def __init__(self):
        self.Q_76Ga = 6.9163

        self.popfile_name = "createdPopFile.dat"

        self.rainier_sample_folder = Path(r"C:\RAINIER\sample_folder")
        self.rainier_path =Path(r"C:\RAINIER")

        #für zuhause
        self.this_dir = Path(r"C:\FAIRIES output")
        #für uni
        #self.this_dir = Path(r"C:\Users\Raphael\Desktop\FluctuationAnalysis\FAIRIES output")

        self.std_path = self.this_dir / "fluctuation_analysis"
        self.run_folder = self.this_dir / "RUNS"
        self.root_file_folder = self.this_dir / "ROOT_files"
        self.settings_file_path = self.root_file_folder / "settings.h"
        self.popfile_path = self.root_file_folder / self.popfile_name
        self.parallel_path = self.this_dir / "PARALLEL"

  

        self.folder_fluct_name = "fluct input"
        self.folder_smoothing_name = "smoothing"
        self.folder_stationary_name = "stationary"
        self.folder_autocorr_name = "autocorrelation"
        self.folder_comparison_name = "comparison"

        self.parameter: dict[Setting, Any] = {
            Setting.g_nReal : 1,
            Setting.g_nZ : 32,
            Setting.g_nAMass : 76,
            Setting.g_nConEBin : 2000,
            Setting.g_nEvent : 100,
            Setting.g_nConSpbMax : 21,
            Setting.g_nDisLvlMax : 1,
            Setting.g_dExIMax : self.Q_76Ga, #ONLY WORKS FOR THE bExFullRxn atm LOOK DEFINER
            Setting.g_dExRes : 0.003459879939969985, #same as above
            Setting.g_nExPopI : 2000, #same as above
            Setting.popFile_name : r'"C:\\FAIRIES output\\ROOT_files\\createdPopFile.dat"',

            Setting.sim_bin_width : 0.05,

            Setting.nld_model : NLD.CTM,

            #NLD.CTM_g_dTemp : 0.48473, this is problematic because it overwrites the upper entries
            #NLD.CTM_dE0 : -1.31817,
            #NLD.BSFG_g_dE1 : 0.968,

            Setting.exp_resolution : 0.003,

            Setting.exci_mode : EXCI.full_rxn,

            Setting.alpha_parameter : 0.273, #with PT 2.273
            Setting.analysis_E_step : 0.5,
            Setting.fluct_bin : 2000,

            Setting.analysis_E_start : 0,
            Setting.analysis_E_end : 7,
            Setting.sliding_window_E_shift : 0.1
        }

        self.setting_definer: dict[Setting, str] = {
            Setting.g_nReal : "const int g_nReal = ",
            Setting.g_nZ : "const int g_nZ = ",
            Setting.g_nAMass : "const int g_nAMass = ",
            Setting.g_nConEBin : "const int g_nConEBin = ",
            Setting.g_nEvent : "const int g_nEvent = ",
            Setting.g_nConSpbMax : "const int g_nConSpbMax = ",
            Setting.g_nDisLvlMax : "const int g_nDisLvlMax = ",
            Setting.g_dExIMax : "//MARKER\nconst double g_dExIMax = ",
            Setting.g_dExRes : "//MARKER\nconst double g_dExRes = ",
            Setting.g_nExPopI : "//MARKER\nconst int g_nExPopI = ",
            Setting.popFile_name : "//MARKER\nconst char popFile[] = "
        }

        self.value_setting: list[Setting] = [
                Setting.g_nReal,
                Setting.g_nZ,
                Setting.g_nAMass,
                Setting.g_nConEBin,
                Setting.g_nEvent,
                Setting.g_nConSpbMax,
                Setting.g_nDisLvlMax,
                Setting.g_dExIMax,
                Setting.g_dExRes,
                Setting.g_nExPopI,
                Setting.popFile_name
        ]

        self.def_str = "#define "

        self.def_setting: list[Setting] = [
            Setting.nld_model,
            Setting.exci_mode
        ]

        self.def_setting_definer: dict[Setting, str] = {
            Setting.nld_model : "///// Level Density, LD, model (Underlying LD) /////",
            Setting.exci_mode : "////////////////////// Excitation Settings /////////////////////////////////////"
        }

        self.def_value: dict[Setting, dict[NLD, str]] = {
            Setting.nld_model : {
                NLD.CTM : "bLD_CTM",
                NLD.BSFG : "bLD_BSFG",
                NLD.Table : "bLD_Table",
                NLD.UsrDef : "bLD_UsrDef",
            },
            Setting.exci_mode : {
                EXCI.single : "bExSingle",
                EXCI.select : "bExSelect",
                EXCI.spread : "bExSpread",
                EXCI.full_rxn : "bExFullRxn"
            }
        }

    def file_setup(self) -> None:
        self.this_dir.mkdir(exist_ok=True, parents=True)
        self.std_path.mkdir(exist_ok=True, parents=True)
        self.root_file_folder.mkdir(exist_ok=True, parents=True)
        self.parallel_path.mkdir(exist_ok=True, parents=True)


    def replace_def(self, text, param, new_def, *, print_setting = True):
        definer = self.def_setting_definer[param]
        pos = text.find(definer)
        pos = pos + len(definer)
        start_pos = pos + text[pos:].find(self.def_str)
        pos2 = start_pos + len(self.def_str)
        end_val = start_pos + text[pos2:].find(" ")
        return text[:pos2] + self.def_value[param][new_def] + text[end_val:]

    def replace_val(self, text, definer, new_val, *, print_setting = True):
        pos = text.find(definer)
        if pos == -1:
            print("Definer was not found in settings.h")
            print(definer)
            return
        start_val = pos + len(definer)
        end_val = start_val
        while text[end_val] != ";":
            end_val = end_val + 1
        val = text[start_val:end_val]
        if val != str(new_val):
            if print_setting:
                print("(changed) "+text[pos:start_val] + str(new_val))
            return text[:start_val] + str(new_val) + text[end_val:]
        else:
            if print_setting:
                print(text[pos:end_val])
            return text



    def apply_settings(self, *, parameter_ref = None, print_setting: bool = True, execution_path: Path = None) -> None:
        if execution_path is None:
            execution_path = self.settings_file_path.parent
        if parameter_ref is None:
            parameter_ref = self.parameter
        with open(execution_path / "settings.h", "r") as f:
            text = f.read()
            if text == "":
                raise ValueError("Setting.h is empty")
        with open(execution_path / "settings.h", "w") as f:
            for key in self.parameter.keys():
                if key in self.value_setting:
                    text = self.replace_val(text, self.setting_definer[key], self.parameter[key], print_setting=print_setting)
                elif key in self.def_setting:
                    text = self.replace_def(text, key, self.parameter[key])
            f.write(text)

settings = Settings()
parameter = settings.parameter
settings.file_setup()