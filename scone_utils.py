# Created Mar 2024 by R.Kessler
# generic utilities for scone codes (to avoid duplicatio)
#

import os, sys, yaml, glob, logging

SCONE_DIR = os.getenv('SCONE_DIR')

# define single summary file name for each scone stage so that summary files
# are easy to locate with unix "find" command. PRPOGRAM_CLASS key inside each
# summary file indicates which scone stage.
SCONE_SUMMARY_FILE        = "SCONE_SUMMARY.LOG"   
PROGRAM_CLASS_HEATMAPS    = "scone_heatmaps"
PROGRAM_CLASS_TRAINING    = "scone_train"
PROGRAM_CLASS_PREDICT     = "scone_predict"
 

# define default heatmaps subdir, but allow command override with 
# --heatmaps_subdir <subdir>
HEATMAPS_SUBDIR_DEFAULT = "heatmaps"  

SBATCH_DONE_FILE = "done.txt"

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

