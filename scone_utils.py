# Created Mar 2024 by R.Kessler
# generic parameters and utilities for scone codes (to avoid duplicatio)
#

import os, sys, yaml, glob, logging

SCONE_DIR = os.getenv('SCONE_DIR')
CWD       = os.getcwd()

# define single summary file name for each scone stage so that summary files
# are easy to locate with unix "find" command. PRPOGRAM_CLASS key inside each
# summary file indicates which scone stage.
SCONE_SUMMARY_FILE        = "SCONE_SUMMARY.LOG"   
PROGRAM_CLASS_HEATMAPS    = "scone_heatmaps"
PROGRAM_CLASS_TRAINING    = "scone_train"
PROGRAM_CLASS_PREDICT     = "scone_predict"
 
MODE_TRAIN   = "train"
MODE_PREDICT = "predict"
CONFIG_KEY_TRAINED_MODEL =  "trained_model"  # Dec 20 2025, RK

SIMTAG_Ia    = "Ia"      # from GENTYPE_TO_NAME dict in sim readme
SIMTAG_nonIa = "nonIa"   # and used for internal dictionaries
SIMTAG_LIST  = [ SIMTAG_Ia, SIMTAG_nonIa ]

KEY_README_DOCANA  = 'DOCUMENTATION'  # yaml block to read snana sim info

# define default heatmaps subdir, but allow command override with 
# --heatmaps_subdir <subdir>
HEATMAPS_SUBDIR_DEFAULT    = "heatmaps"  
HEATMAPS_SNID_SELECT_FILE  = "snid_select.hdf5"  # optional, produced by run_refactor.py 

# define a few standard base file names (full path determined by user or pippin)
SBATCH_DONE_FILE_BASE = "done.txt"
PREDICT_CSV_FILE_BASE = "predictions.csv"

FILTER_WAVE_FILE = "filter_mean_wavelength.dat"  # to store <wave> vs. band
KEY_BAND_TO_WAVE = 'band_to_wave'

# =============================
def setup_logging():
    #logging.basicConfig(level=logging.DEBUG,
    logging.basicConfig(level=logging.INFO,
        format="[%(levelname)6s |%(filename)15s]   %(message)s")
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logging.getLogger("seaborn").setLevel(logging.ERROR)
    return
    # end setup_logging

def print_job_command():
    base_code = os.path.basename(sys.argv[0])
    logging.info(f" ========== BEGIN {base_code} =============== ")
    logging.info(f" full command: {' '.join(sys.argv)} \n")
    logging.info(f" SCONE_DIR:    {SCONE_DIR} \n")
    return

def print_simtag_info(comment, info_list):
    info_SNIa = info_list[0]
    info_nonIa = info_list[1]
    logging.info(f"{comment} = {info_SNIa}({SIMTAG_Ia})  {info_nonIa}({SIMTAG_nonIa})")
    return

def is_data_real(data_dir):
    # Created Apr 2024
    # returns True if this data dir is real data;
    # sim is identified by particular keys in readme.

    version     = os.path.basename(data_dir)
    readme_file   = f"{data_dir}/{version}.README"

    with open(readme_file, "r") as r:
        contents = yaml.load(r, Loader=yaml.Loader)
        
    if 'STAT_SUMMARY' in contents[KEY_README_DOCANA]:
        is_data = False
    else:
        is_data = True

    return is_data

def load_config_expandvars(config_file, key_expandvar_list):

    # read yaml input from config file, then expandvars for
    # each element in key_expandvar_list. Be careful that each
    # element is either a list or string.
    #
    # Preserve each un-expanded list with [key]_orig;
    # e.g, if config['mydir'] = $MYDIR/whatever then 
    #      config['mydir'] will be expanded, 
    #  and a new variable is created:
    #      config['mydir_orig'] = '$MYDIR/whatever'

    if not os.path.exists(config_file):
        sys.exit(f"\n ERROR: input config_file = \n\t{config_file}\n does not exist.")

    with open(config_file, "r") as c:
        config = yaml.load(c, Loader=yaml.Loader)

    # expand env vars
    for key_path in key_expandvar_list:
        if key_path in config:
            path_tmp      = config[key_path]

            # preserve path before expandvars
            config[key_path+'_orig'] = path_tmp  

            if isinstance(path_tmp,list):
                config[key_path] = [os.path.expandvars(path) for path in path_tmp]
            else:
                # make sure path_tmp is a string and not None or False
                if isinstance(path_tmp,str):
                    config[key_path] = os.path.expandvars(path_tmp)

    # TO DO: check that all paths exist ?
    return config
    # end  load_config_expandvars


def compress_files(flag, dir_name, wildcard, name_backup, wildcard_keep ):

    # Extracted from $SNANA_DIR/util/submit_batch/submit_util.py
    #  (sys.path.insert doesn't work because scone env cannot import submit_batch options)
    # Name of created tar file is BACKUP_{name_backup}.tar 
    #
    # Inputs
    #  flag > 0 -> compress
    #  flag < 0 -> uncompress
    #  dir_name -> cd to this directory
    #  wildcard -> include these files in tar file
    #  name_backup -> tar file name is BACKUP_{name_backup}.tar
    #        if name_backup has .tar extension, the use this name with BACKUP prefix.
    #  wildcard_keep -> do NOT remove these files
    #

    if '.tar' in name_backup:
        tar_file   = name_backup
    else:
        tar_file   = f"BACKUP_{name_backup}.tar"

    targz_file = f"{tar_file}.gz"
    cddir      = f"cd {dir_name}"

    # be careful if wildcard string is too short; don't want rm *
    if len(wildcard) < 3 :
        msgerr = []
        msgerr = f"wildcard = '{wildcard}' is dangerously short string"
        msgerr = f"that could result in removing too much."
        msgerr = f"Provide longer wildcard string."
        log_assert(False,msgerr)

    if flag > 0 :
        cmd_tar  = f"tar -cf {tar_file} {wildcard} "
        cmd_gzip = f"gzip {tar_file}"

        if len(wildcard_keep) == 0 :
            cmd_rm   = f"rm {wildcard}"
        else:
            # remove all wildcard files EXCEPT for wildcard_keep 
            cmd_rm = f"find {wildcard} ! -name '{wildcard_keep}' " + \
                     "-type f -exec rm {} +"

        cmd_all  = f"{cddir} ; {cmd_tar} ; {cmd_gzip} ; {cmd_rm} "
    else:
        cmd_unpack = f"tar -xzf {targz_file}"
        cmd_rm     = f"rm {targz_file}"
        cmd_all    = f"{cddir} ; {cmd_unpack} ; {cmd_rm} "

    os.system(cmd_all)
    return

    # end compress_files


def get_sim_readme_yaml(simdir):
    # Created Apr 2024 by R.Kessler
    # for input sim dir, return contents of readme-yaml file
    version     = os.path.basename(simdir)
    readme_file = os.path.expandvars(f"{simdir}/{version}.README")
    contents    = load_config_expandvars(readme_file, [] )
    return contents

def load_SIM_README_DOCANA(config):
    # read each sim readme and store DOCUMENTATION block;   
    # return list of DOCUMENTATION blocks so that other utils 
    # can read from memory instead of re-reading from disk 

    readme_contents_list = []

    key_path   = 'input_data_paths'   # config file key
    
    if key_path not in config:
        return readme_contents_list

    for simdir in config[key_path]:
        contents  = get_sim_readme_yaml(simdir)[KEY_README_DOCANA]
        readme_contents_list.append(contents)

    config['sim_readme_contents_list'] = readme_contents_list
    return


def load_SIM_GENTYPE_TO_NAME(config):

    # Created Apr 2024 by R.Kessler 
    # If sim training data readme has GENTYPE_TO_NAME dictionary, read it and 
    # add it to config as if it were read from the scone-input file.    

    readme_contents_list = config.setdefault('sim_readme_contents_list', "")
    if len(readme_contents_list) == 0 : return

    key_README_GENTYPE = 'GENTYPE_TO_NAME'
    SIM_GENTYPE_TO_NAMES = {}  # e.g., 10:  Ia  SALT3
    SIM_GENTYPE_TO_CLASS = {}  # e.g., 10:  Ia

    for readme_contents in readme_contents_list:        
        gentype_dict = readme_contents.setdefault(key_README_GENTYPE,None)
        if gentype_dict is not None:
            # store unique gentypes to avoid duplication
            for id_type, str_type in gentype_dict.items():
                if id_type not in SIM_GENTYPE_TO_NAMES:
                    SIM_GENTYPE_TO_NAMES[id_type] = str_type
                    SIM_GENTYPE_TO_CLASS[id_type] = str_type.split()[0]

    config['SIM_GENTYPE_TO_NAMES']  = SIM_GENTYPE_TO_NAMES
    config['SIM_GENTYPE_TO_CLASS']  = SIM_GENTYPE_TO_CLASS

    logging.info(f"Read SIM_GENTYPE_TO_NAMES from sim-readme files: " \
                 f" {SIM_GENTYPE_TO_NAMES}")

    logging.info(f"Read SIM_GENTYPE_TO_CLASS from sim-readme files: " \
                 f" {SIM_GENTYPE_TO_CLASS}")

    return


def load_SIM_GENFILTER_WAVE(config):

    # Created Apr 2024 by R.Kessler 
    # Load mean wavelength vs. band and store them in config
    # as if they were read from scone input file

    readme_contents_list = config['sim_readme_contents_list'] 
    if len(readme_contents_list) == 0 : return

    key_README_FILTERS = 'FILTERS'

    contents     = readme_contents_list[0] # use first sim dir
    band_to_wave = contents.setdefault(key_README_FILTERS,None)  

    config[KEY_BAND_TO_WAVE] = band_to_wave
    logging.info(f"Read {KEY_BAND_TO_WAVE} from sim-readme file: " \
                 f" {band_to_wave}")

    return

def load_TRAIN_GENFILTER_WAVE(config):
    # created Jun 2024 by R.Kessler
    # for predict mode, read filters from train_model dir to ensure 
    # consistency with filter wavelengths used in the training.

    config[KEY_BAND_TO_WAVE] = None
    trained_model_path    = config['trained_model']
    filter_mean_wave_file = f"{trained_model_path}/{FILTER_WAVE_FILE}"

    if not os.path.exists(filter_mean_wave_file): return

    with open(filter_mean_wave_file, "rt") as f:
        band_to_wave = yaml.load(f, Loader=yaml.Loader)

    config[KEY_BAND_TO_WAVE] = band_to_wave
    logging.info(f"Read {KEY_BAND_TO_WAVE} from trained_model: " \
                 f" {band_to_wave}")
  
    return

def load_SIM_STAT_SUMMARY(config):

    # Created Apr 2024 by R.Kessler 
    # Read and store Ia and NONIa stats from sim ... may be needed
    # to internally compute prescale to get desired number of events.

    readme_contents_list = config['sim_readme_contents_list'] 
    if len(readme_contents_list) == 0 : return

    key_STAT_SUMMARY = 'STAT_SUMMARY'
    SIM_STAT_SUMMARY = {}
    SIM_STAT_SUMMARY[SIMTAG_Ia]    = 0
    SIM_STAT_SUMMARY[SIMTAG_nonIa] = 0

    for readme_contents in readme_contents_list:        
        stat_summary_list = readme_contents[key_STAT_SUMMARY]
        for stat_row in stat_summary_list:
            model     = stat_row.split()[0]
            nlc_gen   = int(stat_row.split()[1])  # not used
            nlc_write = int(stat_row.split()[2])
            if 'SNIa' in model:
                simtag = SIMTAG_Ia
            elif 'NONIa' in model:
                simtag = SIMTAG_nonIa
            else:
                continue

            SIM_STAT_SUMMARY[simtag] += nlc_write

    config['SIM_STAT_SUMMARY'] = SIM_STAT_SUMMARY

    logging.info(f"Read SIM_STAT_SUMMARY from sim-readme files: " \
                 f"{SIM_STAT_SUMMARY}")

    return
