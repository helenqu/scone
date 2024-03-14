#!/usr/bin/env python
#
# Begin run.py refactor,  March 2024 (R,Kessler, H.Qu)
#
# Main reason is to make modifications compatible with those in create_heatmaps_job.py
# that fixes occasionaly heatmap crashes. Also do some re-organization for future
# maintainability.
#
#  
import os, sys, yaml, shutil
import argparse, subprocess
from astropy.table import Table
import numpy as np
import h5py

from   scone_utils import *
import scone_utils as util


# code locations
SCONE_DIR           = os.path.dirname(os.path.abspath(__file__))
SCONE_HEATMAPS_DIR  = os.path.join(SCONE_DIR, "create_heatmaps") 

# default code names under $SCONE_DIR
JOBNAME_HEATMAP             = "create_heatmaps_job.py"
JOBNAME_TRAIN               = "model_utils.py"

# sbatch info
SBATCH_HEATMAPS_WALLTIME    = '4:00:00'
SBATCH_HEATMAPS_PREFIX      = "create_heatmaps"

SBATCH_TRAIN_FILE           = "job.slurm"    # match original pippin
SBATCH_TRAIN_LOG            = "output.log"
SBATCH_TRAIN_WALLTIME       = '20:00:00'

SBATCH_MEM = 32   # GB, applies to all stagee. Can overrwrite with" sbatch_mem:" key in config

# ---------------------------------------
# HELPER FUNCTIONS


def load_gentype_dict(config_path):

    # read integer<->string map of types from hard-wired file in under $SCONE_DIR.
    # This needs to be replaced with something that reads simulated README so that
    # scone stays in sync with user GENTYPES.
    gentype_path = os.path.join(SCONE_HEATMAPS_DIR, "default_gentype_to_typename.yml")
    gentype_config = util.load_config_expandvars(gentype_path,[])["gentype_to_typename"]
    return gentype_config


def count_FITS_files(input_data_path_list):
    # open [version].LIST fill and sum number of entries
    count = 0
    for data_path in input_data_path_list:
        version = os.path.basename(data_path)
        list_file = f"{data_path}/{version}.LIST"
        with open(list_file,"rt") as l:
            count += len(l.readlines())
    return count


# autogenerate some parts of config
def append_scone_config_file(config_path,config):

    with open(config_path, "a+") as c:
        c.write(f"\n# Appended by run.py\n")
        c.write(f"types: \n")
        for str_type in [ 'II', 'Ia' ]:
            c.write(f"- {str_type}\n")
    return 


def get_args():
    parser = argparse.ArgumentParser(description='create heatmaps from lightcurve data')

    msg = "path to config file"
    parser.add_argument('--config_path', type=str, default=None, help = msg)

    msg = f"alternate heatmaps subdir (default is '{HEATMAPS_SUBDIR_DEFAULT}')"
    parser.add_argument('--heatmaps_subdir', type=str, 
                        default=HEATMAPS_SUBDIR_DEFAULT, help = msg)

    msg = "nosubmit: create sbatch inputs, but do not submit"
    parser.add_argument("-n", "--nosubmit", help=msg, action="store_true")

    msg = "define config file keys"
    parser.add_argument("-H", "--HELP", help=msg, action="store_true")

    ARGS = parser.parse_args()

    return ARGS



def prepare_sbatch_info(SCONE_CONFIG):

    key_default = 'sbatch_template_default'
    key         = 'sbatch_template_train'
    if key not in SCONE_CONFIG:
        SCONE_CONFIG[key] = SCONE_CONFIG[key_default]

    # check option to override memory
    key = 'sbatch_mem'
    if key in SCONE_CONFIG:
        global SBATCH_MEM
        SBATCH_MEM = SCONE_CONFIG[key]

    return


def get_jobname(config, jobname_base):

    # default is public code under $SCONE_DIR
    jobname = jobname_base

    # check for scone_dir override based on code path from run.py
    key_scone_dir_list = [ 'scone_dir', 'SCONE_DIR' ] # allow either case to be safe
    for key_scone_dir in key_scone_dir_list :
        if key_scone_dir in config:
            scone_dir = config[key_scone_dir]
            jobname = os.path.join(scone_dir, jobname_base)

    return jobname


def write_sbatch_for_train(ARGS, config):

    # Created Mar 6 2024 by R.Kessler
    # write sbatch file for model training
    # (formely done by pippin, but moved here since heatmap sbatch is created here)

    output_path       = config['output_path']
    init_env          = config['init_env_train']

    sbatch_template  = os.path.expandvars(config['sbatch_template_train'])
    sbatch_file      = output_path + '/' + SBATCH_TRAIN_FILE
    sbatch_log_file  = output_path + '/' + SBATCH_TRAIN_LOG
    
    jobname = get_jobname(config, JOBNAME_TRAIN)

    job_string = f"cd {output_path} \n\n" \
                 f"{init_env} \n\n" \
                 f"{jobname} " \
                 f"--config_path  {ARGS.config_path} "

    if ARGS.heatmaps_subdir != HEATMAPS_SUBDIR_DEFAULT:
        job_string += f"--heatmaps_subdir {ARGS.heatmaps_subdir} "

    job_string += '\n\n'

    # tack on logic to write SUCCESS or FAILURE to done file
    done_logic = 'if [ $? -eq 0 ]; then \n' \
                 f"   echo classify SUCCESS >> {output_path}/{SBATCH_DONE_FILE} \n" \
                 f"else \n" \
                 f"   echo classify FAILURE >> {output_path}/{SBATCH_DONE_FILE} \n" \
                 f"fi \n"

    job_string += done_logic

    REPLACE_KEY_DICT = { 
        'REPLACE_NAME'          : "model_train",
        'REPLACE_MEM'           : str(SBATCH_MEM) + 'GB',
        'REPLACE_LOGFILE'       : sbatch_log_file,
        'REPLACE_JOB'           : job_string,
        'REPLACE_WALLTIME'      : SBATCH_TRAIN_WALLTIME,
        'REPLACE_CPUS_PER_TASK' : '1'
    }

    # write new sbatch_file
    sbatch_key_replace(sbatch_template, sbatch_file, REPLACE_KEY_DICT)

    logging.info(f"Created sbatch training file: {sbatch_file}")

    return sbatch_file
    # end write_sbatch_for_train


def write_sbatch_for_heatmaps(ARGS, config):

    outdir_heatmap   = config['outdir_heatmap']
    ncore            = config['sbatch_ncore_heatmap']
    sbatch_template  = os.path.expandvars(config['sbatch_template_default'])

    ntask_tot        = count_FITS_files(config['input_data_paths'])
    ntask_per_cpu    = int(ntask_tot / ncore) + 1

    # start mem/core at 1GB and keep doubling until we exceed total limit
    mem_per_core = 1 
    while ntask_per_cpu*mem_per_core  <= SBATCH_MEM/2 :
        mem_per_core *= 2
    str_mem_per_core = f"{mem_per_core}GB"  # convert to string for slurm


    init_env = config['init_env_heatmaps']

    # - - - - 
    sbatch_heatmap_file_list = []

    for i in range(0,ncore):
        prefix           = f"{SBATCH_HEATMAPS_PREFIX}__{i:03d}"
        sbatch_file      = f"{outdir_heatmap}/{prefix}.sh"
        sbatch_log_file  = f"{outdir_heatmap}/{prefix}.log"
        jobname = get_jobname(config, JOBNAME_HEATMAP)

        sbatch_heatmap_file_list.append(sbatch_file)

        job_string = f"cd {outdir_heatmap} \n\n" \
                     f"{init_env} \n\n" \
                     f"{jobname} " \
                     f"--config_path  {ARGS.config_path} " \
                     f"--slurm_id {i}  --nslurm_tot {ncore} "

        if ARGS.heatmaps_subdir != HEATMAPS_SUBDIR_DEFAULT:
            job_string += f"--heatmaps_subdir {ARGS.heatmaps_subdir} "

        REPLACE_KEY_DICT = { 
            'REPLACE_NAME'          : prefix,
            'REPLACE_MEM'           : str_mem_per_core,
            'REPLACE_LOGFILE'       : sbatch_log_file,
            'REPLACE_JOB'           : job_string,
            'REPLACE_WALLTIME'      : SBATCH_HEATMAPS_WALLTIME,
            'REPLACE_CPUS_PER_TASK'  : str(ntask_per_cpu)
        }

        # write new sbatch_file
        sbatch_key_replace(sbatch_template, sbatch_file, REPLACE_KEY_DICT)

    n_file = len(sbatch_heatmap_file_list)
    logging.info(f"Created {n_file} sbatch files for heatmap creation.")

    return sbatch_heatmap_file_list
    # end write_sbatch_for_heatmaps


def mkdir_heatmaps(ARGS, config):
    
    outdir         = SCONE_CONFIG['output_path']    
    outdir_heatmap = os.path.join(outdir, ARGS.heatmaps_subdir)
    SCONE_CONFIG['outdir_heatmap'] = outdir_heatmap

    # create new heatmaps directory (clobber old one if it exists)
    if os.path.exists(outdir_heatmap):
        shutil.rmtree(outdir_heatmap)

    logging.info(f"create output dir for heatmaps: \n\t {outdir_heatmap} ")

    os.makedirs(outdir_heatmap)

    return

def sbatch_key_replace(sbatch_template_file, sbatch_out_file, REPLACE_KEY_DICT):

    # Created Mar 6 2024 by R.Kessler
    # read sbatch_template_files and write lines to new sbatch_out_file,
    # make substitutions as indicated by REPLACE_DICT.

    with open(sbatch_template_file,"rt") as t:
        line_template_list = t.readlines()

    with open(sbatch_out_file,"wt") as s:

        for line in line_template_list:
            line = line.rstrip()  # remove trailing spaces and CR                                 
            line_out = line
            # replace keys                                                                        
            for key_replace, val_replace in REPLACE_KEY_DICT.items():
                if key_replace in line:
                    line_out = line.replace(key_replace, val_replace)

            s.write(f"{line_out}\n")
    return
    # end sbatch_key_replace

def print_config_help():

    help_config = f"""
    **** config help menu *** 

batch_size:     32     # ??

# training params
categorical:    false  # ??
class_balanced: true   # ??
num_epochs:     400    # ??

# heatmap params
num_mjd_bins:        180  # number of MJD bins for heatmap
num_wavelength_bins: 32   # numberr of wave bins for heatmap
prescale_heatmap:    10   # prescale sim events for heatmaps (used for training)

input_data_paths:
  - <sim_output_folder1>  # e.g., Ia sims
  - <sim_output_folder2>  # e.g., contamination sims 
  etc ... 

init_env_train:     source activate scone_cpu_tf2.6  # use gpu env to go faster
init_env_heatmaps:  source activate scone_cpu_tf2.6

output_path: <dirname>  # write all output here (clobbers existing folder)
    
trained_model: null  # ???

# intger GENTYPE <-> string-type declarations 
# (needs refactor to read directly from sim-output README)

sn_type_id_to_name:
  10:  Ia
  12:  II
  21:  II
  26:  II
  32:  II
  110: Ia
  112: II
  121: II
  126: II
  132: II

types: 
  - II
  - Ia

    """

    print(f"\n{help_config}")

    return

# ==================================
# START MAIN FUNCTION
# ==================================

if __name__ == "__main__":

    util.setup_logging()

    util.print_job_command()

    ARGS = get_args()

    if ARGS.HELP:
        print_config_help()
        sys.exit(0)

    key_expandvar_list = [ 'output_path', 'input_data_paths', 
                           'sbatch_template_default', 'sbatch_template_train' ]
    SCONE_CONFIG  = util.load_config_expandvars(ARGS.config_path, key_expandvar_list)
    GENTYPE_DICT  = load_gentype_dict(ARGS.config_path)

    # define output for heatmaps
    mkdir_heatmaps(ARGS, SCONE_CONFIG)

    # define scone code location based on location of run.py
    scone_dir = os.path.dirname(sys.argv[0])
    if len(scone_dir) > 0:
        SCONE_CONFIG['scone_dir'] = scone_dir

    # - - - - 
    # append type list to config file for now ; later it should be auto-computed
    append_scone_config_file(ARGS.config_path, SCONE_CONFIG)

    # write sbatch files for heatmaps and for training:
    prepare_sbatch_info(SCONE_CONFIG)
    sbatch_heatmap_file_list = write_sbatch_for_heatmaps(ARGS, SCONE_CONFIG)
    sbatch_train_file        = write_sbatch_for_train(ARGS, SCONE_CONFIG)

    # launch each heatmap job into slurm
    if ARGS.nosubmit:
        sys.exit(f"\n\t !!! Skip job launch. Bye bye !!!\n ")

    # - - -  launch jobs - - - - - -
    logging.info(f"Launch heatmap-generation jobs into slurm.")
    jid_list = []
    for sbatch_file in sbatch_heatmap_file_list:
        out = subprocess.run(["sbatch", "--parsable", sbatch_file], capture_output=True)
        jid_list.append(out.stdout.decode('utf-8').strip())

    # prepare model-training command to run after all create_heatmaps jobs complete
    model_train_sbatch_cmd = ['sbatch' ]
    model_train_sbatch_cmd.append(f"--dependency=afterok:{':'.join(jid_list)}")
    model_train_sbatch_cmd.append(sbatch_train_file)

    logging.info(f"Launch training job with command: \n{model_train_sbatch_cmd}\n")
    #sys.exit("\n xxx bye xxx \n")
    subprocess.run(model_train_sbatch_cmd)

    logging.info(f"\nFinished launching heatmap & training jobs; wait for jobs to finish.\n")

    # ==== END ===
