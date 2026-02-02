#!/usr/bin/env python
#
# Begin run.py refactor,  March 2024 (R,Kessler, H.Qu)
#
# Main reason is to make modifications compatible with those in create_heatmaps_job.py
# that fixes occasionaly heatmap crashes. Also do some re-organization for future
# maintainability.
# "scone" task refers to either train or predict mode.
# "heatmaps" task is the same for train or predict mode.
#
# Aug 26 2025 RK -
#    if there are multiple snid_select_files (i.e., Ia and CC),
#    check that VERSION_PHOTOMETRY key in each select_file is a valid sim-version;
#    skip invalid snid_select_files to avoid false duplicates.
#    See new method use_select_file(...)
# Aug 29 2025 RK - fix rubble from Aug 26; always use a SIMGEN-DUMP file
#
# Dec 20 2025 RK - abort immediately if no "trained_model" is given for predict mode.
#
# Jan 26 2026 RK - when reading SIMGEN_DUMP[ALL] file (snid_select_file arg), 
#        select rows with FLAG_ACCEPT=1 (if this column exists).
#
# Jan 27 2026 RK - new check_duplicates() method to separately check SIM-Ia and SIM-nonIa,
#                  and to abort on appropriate error.
#

import os, sys, yaml, shutil, gzip
import argparse, subprocess
from astropy.table import Table
import numpy as np
import pandas as pd
import h5py

from   scone_utils import *
import scone_utils as util


# code locations
SCONE_DIR           = os.path.dirname(os.path.abspath(__file__))
SCONE_HEATMAPS_DIR  = os.path.join(SCONE_DIR, "create_heatmaps") 

# default code names under $SCONE_DIR
JOBNAME_HEATMAP             = "create_heatmaps_job.py"
JOBNAME_SCONE               = "model_utils.py"  # train or predict

# hard-wire max wall time, but should be a config input parameter;
# eventually pippin should select wall time based on biascor or sim data.
# xxx mark delete SBATCH_HEATMAPS_WALLTIME    = '4:00:00'
SBATCH_HEATMAPS_WALLTIME    = '10:00:00'   # Aug 29 2025: more wall time for biascor

SBATCH_HEATMAPS_PREFIX      = "create_heatmaps"

SBATCH_SCONE_FILE           = "job.slurm"    # train or predict
SBATCH_SCONE_LOG            = "output.log"   # idem

SBATCH_WALLTIME_DICT = { MODE_TRAIN:   '20:00:00',  
                         MODE_PREDICT: '4:00:00'}

SBATCH_JOB_NAME_DICT = { MODE_TRAIN:   f"scone_{MODE_TRAIN}", 
                         MODE_PREDICT: f"scone_{MODE_PREDICT}" }

# define default memory (GB); applies to all stages. 
# Can overrwrite with" sbatch_mem:" key in config
SBATCH_MEM = 32   

# ---------------------------------------
# HELPER FUNCTIONS


def count_FITS_files(input_data_path_list):
    # open [version].LIST fill and sum number of entries
    count = 0
    for data_path in input_data_path_list:
        version = os.path.basename(data_path)
        list_file = f"{data_path}/{version}.LIST"
        with open(list_file,"rt") as l:
            count += len(l.readlines())
    return count

def create_snid_select_file(config):

    # Created Apr 2024 by R.Kessler
    # By default, the sim select file is the SIMGEN-DUMP file for each sim data folder.
    # If snid_select_files is given in the config input, these are FITRES
    # files (presumably from LCFIT) to use instead of the SIMGEN-DUMP files.
    #
    # Be careful to use is_data/is_sim flags in logic.

    IS_MODEL_TRAIN   = (config['mode'] == MODE_TRAIN)
    IS_MODEL_PREDICT = (config['mode'] == MODE_PREDICT)

    input_data_paths  = config['input_data_paths']
    snid_select_files = config.setdefault('snid_select_files',None)

    # allowed GENTYPE keys in simgen-dump or LCFIT-output FITRES file
    KEYLIST_GENTYPE = [ 'GENTYPE', 'SIM_GENTYPE', 'SIM_TYPE_INDEX' ]

    is_data = util.is_data_real(input_data_paths[0])
    is_sim  = not is_data

    logging.info(f"is_data_real={is_data}   is_data_sim={is_sim}")

    # bail immediately on real data if there is no user-define FITRES file 
    if snid_select_files is None and is_data :
        return 0

    # if there are no user-defined snid_select_files, then use simgen-dump file by default
    if snid_select_files is None and is_sim :
        # use default simgen-dump files for sim; if no dump file, it is real data so bail
        snid_select_files = []
        for simdir in input_data_paths:
            version      = os.path.basename(simdir)
            dump_file    = f"{simdir}/{version}.DUMP"
            dump_file_gz = f"{dump_file}.gz"
            if os.path.exists(dump_file):
                snid_select_files.append(dump_file)
            elif os.path.exists(dump_file_gz):
                snid_select_files.append(dump_file_gz)
            else: 
                sys.exit(f"\n cannot select CIDs for sim because there is no DUMP file\n\t {dump_file}")

    sim_version_list = []
    if is_sim:
        util.load_SIM_README_DOCANA(config)
        util.load_SIM_GENTYPE_TO_NAME(config)  # read map of gentype <--> Ia,nonIa  
        SIM_GENTYPE_TO_CLASS = config['SIM_GENTYPE_TO_CLASS']
        for simdir in input_data_paths:
            sim_version_list.append(os.path.basename(simdir))  # Aug 26 2025

    logging.info("Begin reading [SIMGEN_DUMP] snid_select_files: ")
    snid_all_list    = []
    gentype_all_list = [] 
    for select_file in snid_select_files:
        select_file = os.path.expandvars(select_file)  # Aug 26 2025

        if not use_select_file(is_sim, select_file, sim_version_list):
            logging.info(f"Skip snid_select_file that doesn't match sim versions: \n" \
                         f"{select_file}")
            continue
        
        if '.gz' in select_file:
            df = pd.read_csv(select_file, compression='gzip', 
                             comment="#", delim_whitespace=True)
        else:
            df = pd.read_csv(select_file,
                             comment="#", delim_whitespace=True)

        n_df_tot = len(df)

        KEY_FLAG_ACCEPT = 'FLAG_ACCEPT'
        if KEY_FLAG_ACCEPT in list(df.columns):
            df_select = df.loc[df[KEY_FLAG_ACCEPT] == 1 ]  # Jan 27 2026: needed for SIMGEN_DUMPALL
        else:
            df_select = df  # read entire SIMGEN_DUMP file for legacy sims (before FLAG_ACCEPT column)

        n_df = len(df_select)
        logging.info(f"Read {n_df} (of {n_df_tot}) rows from snid_select_file {select_file}")
        
        snid_all_list += df_select['CID'].tolist()    
        found_gentype = False        
        for key_gentype in KEYLIST_GENTYPE:
            if key_gentype in df.columns:
                gentype_all_list += df_select[key_gentype].tolist()
                found_gentype = True
                
        if IS_MODEL_TRAIN and found_gentype is False:
            msgerr = f'\nERROR: could not find GENTYPE key in \n{select_file}\n' \
                     f'after checking for {KEYLIST_GENTYPE}'
            assert False,  msgerr 
    
    # check for duplicates:
    check_duplicates(is_data, snid_all_list, gentype_all_list, config)

    # - - - - - - -
    # check nevt_select and/or prescale options
    snid_list    = snid_all_list     # default with no prescale
    gentype_list = gentype_all_list

    if is_sim:
        nsim_tot_list = n_per_class(gentype_all_list,SIM_GENTYPE_TO_CLASS)
        print_simtag_info("NSIM", nsim_tot_list)

    if IS_MODEL_TRAIN :
        key_nevt         = 'nevt_select_heatmaps'
        nevt_select      = config.setdefault(key_nevt,None)
        key_ps           = 'prescale_heatmaps'
        ps_select        = config.setdefault(key_ps,None)
    else:
        nevt_select = None
        ps_select   = None


    # - - - - - -  -
    ps_list    = []
    if nevt_select:
        nevt_select_list = nevt_select.split(',')  # e.g., "40000,25000".split(',')
        for nsimtot, nevt_select in zip(nsim_tot_list,nevt_select_list):
            ps = int( float(nsimtot) / float(nevt_select) + 0.5)
            if ps == 0 : ps = 1
            ps_list.append(ps)
        print_simtag_info("user-defined nevt_select", nevt_select_list)
        print_simtag_info("Computed prescale", ps_list)
    elif ps_select :
        ps_list     = str(ps_select).split(',')
        if len(ps_list) == 1 :
            ps_list.append(ps_list[0])
        print_simtag_info("user-defined prescale", ps_list)

    else:
        ps_list = [ 1, 1 ]

    # - - - - - - - -
    # apply pre-scale if defined. 
    # start with brute-force (slow) method, and hopefully later find
    # a more efficiency way.
    if is_data:
        snid_list = snid_all_list  # never apply ps to real data
    else:
        snid_list_dict = { SIMTAG_Ia: [],  SIMTAG_nonIa: [] }
        snid_list    = []
        gentype_list = []
        ps_list = [ int(ps_list[0]), int(ps_list[1]) ]
        for snid, gentype in zip(snid_all_list, gentype_all_list):
            for ps, simtag in zip(ps_list, SIMTAG_LIST):
                match_type = (SIM_GENTYPE_TO_CLASS[gentype] == simtag) # math Ia or nonIa
                match_ps   = (int(snid) % ps == 0)
                if match_type and match_ps:
                    snid_list.append(snid)
                    gentype_list.append(gentype)
                    snid_list_dict[simtag].append(snid)

        n_final_per_type = n_per_class(gentype_list,SIM_GENTYPE_TO_CLASS)
        print_simtag_info("NSIM(after prescale)", n_final_per_type)

    # - - - - - - -
    # write snid_list to hdf5 file.
    # Write separate list for Ia and nonIa to avoid random SNID overlaps
    # (e.g., Ia SNID matching a nonIa SNID, and vice versa)

    n_select         = len(snid_list)
    outdir_heatmaps  = config['outdir_heatmaps']
    snid_hdf5_file = f"{outdir_heatmaps}/{HEATMAPS_SNID_SELECT_FILE}"
    config['snid_hdf5_file'] = snid_hdf5_file

    logging.info(f"Prepare to write {n_select} selected SNIDs to {snid_hdf5_file}")
    f = h5py.File(snid_hdf5_file, "w")

    # write prescale list [Ia,nonIa] first
    f.create_dataset("prescales", data = ps_list)

    if is_data:
        f.create_dataset("ids", data=snid_list, dtype=np.int32)
    else:
        # sim
        for simtag in SIMTAG_LIST :
            snid_list = snid_list_dict[simtag]
            logging.info(f"\t write {len(snid_list)} SNIDs for {simtag}")
            f.create_dataset("ids_" + simtag, data=snid_list, dtype=np.int32)

    f.close()

    return n_select


def check_duplicates(is_data, snid_all_list, gentype_all_list, config):

    # Created Jan 2026 by R.Kessler
    # For real data, abort on any duplicate.
    # For sim, abort on SNIa duplicates, or NONIA duplicates, but allow
    # duplicates between SNIa and NonIa because Ia and nonIa sims are
    # often generated separately.

    n_all_list     = []
    n_dup_list     = []
    label_dup_list = []
    valid = True

    if is_data:
        n_all, n_dup = count_duplicates(snid_all_list)
        n_all_list = [ n_all ]
        n_dup_list = [ n_dup ]
        label_dup_list = [ "Data"]
        valid = (n_dup == 0)
    else:
        SIM_GENTYPE_TO_CLASS = config['SIM_GENTYPE_TO_CLASS']
        snid_list1 = [ snid for snid, t in zip(snid_all_list, gentype_all_list) \
                       if SIM_GENTYPE_TO_CLASS[t] == SIMTAG_Ia]
        snid_list2 = [ snid for snid, t in zip(snid_all_list, gentype_all_list) \
                       if SIM_GENTYPE_TO_CLASS[t] == SIMTAG_nonIa]
        n1_all, n1_dup = count_duplicates(snid_list1)
        n2_all, n2_dup = count_duplicates(snid_list2)
        n_all_list = [ n1_all, n2_all ]
        n_dup_list = [ n1_dup, n2_dup ]
        label_dup_list = [ 'SIM-'+SIMTAG_Ia,  'SIM-'+SIMTAG_nonIa ]
        valid = (n1_dup == 0) and (n2_dup == 0)

    # - - - - 
    for n_all, n_dup, label in zip(n_all_list, n_dup_list, label_dup_list):
        str_err=''
        if n_dup > 0: str_err = "==> ERROR" 
        logging.info(f"Found {n_dup} duplicates from {n_all}  {label}  events  {str_err}")

    assert valid,  f"ABORT on invalid {label_dup_list} duplicates; " \
        f"\n\t see duplicate message(s) above."
        
    return
    

def count_duplicates(snid_list):
    n_all    = len(snid_list)
    n_unique = len(set(snid_list))
    n_dup    = n_all - n_unique
    return n_all, n_dup

def use_select_file(is_sim, select_file, sim_version_list):

    # For sims, read VERSION_PHOTOMETRY from comment at top of select_file
    # and make sure that this version is one of the sims.
    # Assumes SNANA-formatted FITRES file is used for snid_select_file.
    
    # always return true for real data
    if not is_sim: return True

    # if simgen-dump file, always return True
    suffix = select_file.split('.')[1]
    if 'DUMP' in suffix: return True

    # open select_file for reading
    if '.gz' in select_file:
        f = gzip.open(select_file,"rt")
    else:
        f = open(select_file,"rt")

    NLINE_ABORT = 10
    KEY = "VERSION_PHOTOMETRY:"
    for i, line in enumerate(f):
        if i >= NLINE_ABORT:
            msgerr = f'\n\nERROR: could not find {KEY} key in first {i} lines\n' \
                     f'in select file {select_file}'
            assert False,  msgerr 
            
        if KEY in line:
            wdlist = line.split()
            j      = wdlist.index(KEY)
            select_version = wdlist[j+1]
            select_version_list = select_version.split(',') # could be comma sep list
            break
        
    # - - - - - - - - - - - -
    use = False
    for select_version in select_version_list:
        if select_version in sim_version_list:
            use = True
            
    f.close()
        
    return use


def n_per_class(gentype_list,SIM_GENTYPE_TO_CLASS):
    n_all = len(gentype_list)
    n_Ia  = sum(1 for i in gentype_list if SIM_GENTYPE_TO_CLASS[i] == SIMTAG_Ia)
    n_nonIa    = n_all - n_Ia
    return [ n_Ia, n_nonIa ]

def get_args():
    parser = argparse.ArgumentParser(description='Run scone on lightcurve data')

    msg = "path to config file"
    parser.add_argument('--config_path', type=str, default=None, help = msg)

    msg = f"alternate heatmaps subdir (default is '{HEATMAPS_SUBDIR_DEFAULT}')"
    parser.add_argument('--heatmaps_subdir', type=str, 
                        default=HEATMAPS_SUBDIR_DEFAULT, help = msg)

    msg = f"optional sbatch job-name " \
          f"(for train or predict mode, not for heatmaps)"
    parser.add_argument('--sbatch_job_name', type=str, 
                        default=None, help = msg)

    msg = "nosubmit: create sbatch inputs, but do not submit"
    parser.add_argument("-n", "--nosubmit", help=msg, action="store_true")

    msg = "print help for config file keys"
    parser.add_argument("-H", "--HELP", help=msg, action="store_true")

    ARGS = parser.parse_args()
    
    if ARGS.HELP: 
        print_config_help()
        sys.exit(0)

    # if config_path does not include full path, prepend current directory to path;
    # needed so that slurm jobs under heatmaps point to config file with full path.
    if '/' not in ARGS.config_path:
        ARGS.config_path = CWD + '/' + ARGS.config_path

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


def write_sbatch_for_scone(ARGS, config):

    # Created Mar 6 2024 by R.Kessler
    # Write scone-sbatch file for model training or predict mode.
    # 
    # (formely done by pippin, but moved here since heatmap sbatch is created here)

    mode = config['mode']
    IS_MODEL_TRAIN   = (mode == MODE_TRAIN)
    IS_MODEL_PREDICT = (mode == MODE_PREDICT)

    # Dec 20 2025 RK - abort immediately if trained_model isn't provided for predict mode
    if IS_MODEL_PREDICT and CONFIG_KEY_TRAINED_MODEL not in config:
        msgerr = \
                 f' ERROR: missing required "{CONFIG_KEY_TRAINED_MODEL}:" config key ' \
                 f'for {mode} mode in \n' \
                 f'\t {ARGS.config_path}'
        assert False,  msgerr 

    output_path       = config['output_path']

    # few 'train' items are the same for train and predict
    init_env         = config.setdefault('init_env_train',"")  
    sbatch_template  = os.path.expandvars(config['sbatch_template_train'])

    sbatch_file      = output_path + '/' + SBATCH_SCONE_FILE
    sbatch_log_file  = output_path + '/' + SBATCH_SCONE_LOG
    
    jobname = get_jobname(config, JOBNAME_SCONE)

    job_string = f"cd {output_path} \n\n" \
                 f"{init_env} \n\n" \
                 f"{jobname} " \
                 f"--config_path  {ARGS.config_path} "


    job_string += '\n\n'

    # tack on logic to write SUCCESS or FAILURE to done file
    done_file  = f"{output_path}/{SBATCH_DONE_FILE_BASE}"
    done_logic = 'if [ $? -eq 0 ]; then \n' \
                 f"   echo classify SUCCESS >> {done_file} \n" \
                 f"else \n" \
                 f"   echo classify FAILURE >> {done_file} \n" \
                 f"fi \n"

    job_string += done_logic


    sbatch_job_name = config.setdefault('sbatch_job_name',
                                        SBATCH_JOB_NAME_DICT[mode] )
    sbatch_walltime = SBATCH_WALLTIME_DICT[mode]

    # - - - - 
    # check overrides from user or pippin
    if ARGS.heatmaps_subdir != HEATMAPS_SUBDIR_DEFAULT:
        job_string += f"--heatmaps_subdir {ARGS.heatmaps_subdir} "

    if ARGS.sbatch_job_name:
        sbatch_job_name = ARGS.sbatch_job_name  

    # - - - - -
    REPLACE_KEY_DICT = { 
        'REPLACE_NAME'          : sbatch_job_name,
        'REPLACE_MEM'           : str(SBATCH_MEM) + 'GB',
        'REPLACE_LOGFILE'       : sbatch_log_file,
        'REPLACE_JOB'           : job_string,
        'REPLACE_WALLTIME'      : sbatch_walltime,
        'REPLACE_CPUS_PER_TASK' : '1'
    }

    # write new sbatch_file
    sbatch_key_replace(sbatch_template, sbatch_file, REPLACE_KEY_DICT)

    logging.info(f"Created sbatch {mode} file: {sbatch_file}")

    return sbatch_file
    # end write_sbatch_for_scone


def write_sbatch_for_heatmaps(ARGS, config):

    # write sbatch (slurm) files used to create heatmaps

    mode             = config['mode']
    outdir_heatmaps  = config['outdir_heatmaps']
    ncore            = config['sbatch_ncore_heatmaps']
    sbatch_template  = os.path.expandvars(config['sbatch_template_default'])

    ntask_tot        = count_FITS_files(config['input_data_paths'])
    if ncore > ntask_tot: ncore = ntask_tot  # avoid using CPUs with no task
    ntask_per_cpu    = int(ntask_tot / ncore) + 1

    # start mem/core at 1GB and keep doubling until we exceed total limit
    mem_per_core = 1 
    while ntask_per_cpu*mem_per_core  <= SBATCH_MEM/2 :
        mem_per_core *= 2
    str_mem_per_core = f"{mem_per_core}GB"  # convert to string for slurm

    init_env = config.setdefault('init_env_heatmaps',"")

    arg_heatmaps_snid_select = ""    
    nsnid_select = create_snid_select_file(config)
    if nsnid_select > 0 :
        arg_heatmaps_snid_select = f"--hdf5_select_file {HEATMAPS_SNID_SELECT_FILE}"

    # - - - - 
    sbatch_heatmap_file_list = []

    for i in range(0,ncore):
        prefix           = f"{SBATCH_HEATMAPS_PREFIX}_{mode}_{i:03d}"
        sbatch_file      = f"{outdir_heatmaps}/{prefix}.sh"
        sbatch_log_file  = f"{outdir_heatmaps}/{prefix}.log"
        jobname = get_jobname(config, JOBNAME_HEATMAP)

        sbatch_heatmap_file_list.append(sbatch_file)

        job_string = f"cd {outdir_heatmaps} \n\n" \
                     f"{init_env} \n\n" \
                     f"{jobname} " \
                     f"--config_path  {ARGS.config_path} " \
                     f"--slurm_id {i}  --nslurm_tot {ncore} " \
                     f"{arg_heatmaps_snid_select} "

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
    outdir_heatmaps = os.path.join(outdir, ARGS.heatmaps_subdir)
    SCONE_CONFIG['outdir_heatmaps'] = outdir_heatmaps

    # create new heatmaps directory (clobber old one if it exists)
    if os.path.exists(outdir_heatmaps):
        shutil.rmtree(outdir_heatmaps)

    logging.info(f"create output dir for heatmaps: \n\t {outdir_heatmaps} ")

    os.makedirs(outdir_heatmaps)

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

# slurm inputs
sbatch_template_default:      $SBATCH_TEMPLATES/SBATCH_scone_cpu.TEMPLATE
sbatch_template_train:        $SBATCH_TEMPLATES/SBATCH_scone_cpu.TEMPLATE 
sbatch_ncore_heatmaps:        10
sbatch_mem:                   20  # GB : includes all ncore cpus/gpus


batch_size:     32     # ??

# training params
categorical:    false   # ??
class_balanced: false   # create heatmaps with same NEVT per class (disabled in refactor)
num_epochs:     400     # ??

# heatmap params
num_mjd_bins:        180  # number of MJD bins for heatmap
num_wavelength_bins: 32   # numberr of wave bins for heatmap

nevt_select_heatmaps: 20000,25000  # compute prescales to get Ia,nonIa stats

# pick snid subset from already existing FITRES files, such as from LCFIT
# Pippin fills this based on OPTS key "OPTIONAL_MASK_FIT:  <MASK>"
snid_select_files:
  - fitres file 1 
  - fitres file 2 
  - etc ...

prescale_heatmaps:    10    # prescale = 10 for sim-training events for heatmaps
#      or
prescale_heatmaps:  20,10   # prescale = 20 for Ia, and 10 for nonIa


# specify input data (pippin automatically fills this)
input_data_paths:
  - <sim_output_folder1>  # e.g., Ia sims
  - <sim_output_folder2>  # e.g., contamination sims 
  etc ... 

# define conda envs
init_env_train:     source activate scone_cpu_tf2.6  # use gpu env to go faster
init_env_heatmaps:  source activate scone_cpu_tf2.6

# specify outputs

output_path: <dirname>   # write all output here (clobbers existing folder)
    
# for predict mode,
prob_column_name:  PROB_SCONE  # name of PROB colummn in output predictions.csv 

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

    key_expandvar_list = [ 'output_path', 'input_data_paths', 'snid_select_files',
                           'sbatch_template_default', 'sbatch_template_train' ]
    SCONE_CONFIG  = util.load_config_expandvars(ARGS.config_path, key_expandvar_list)

    # define output for heatmaps
    mkdir_heatmaps(ARGS, SCONE_CONFIG)

    # define scone code location based on location of run.py
    scone_dir = os.path.dirname(sys.argv[0])
    if len(scone_dir) > 0:
        SCONE_CONFIG['scone_dir'] = scone_dir

    # - - - - 
    # write sbatch files for heatmaps and for training:
    prepare_sbatch_info(SCONE_CONFIG)
    sbatch_heatmap_file_list = write_sbatch_for_heatmaps(ARGS, SCONE_CONFIG)
    sbatch_scone_file        = write_sbatch_for_scone(ARGS, SCONE_CONFIG)

    # launch each heatmap job into slurm
    if ARGS.nosubmit:
        sys.exit(f"\n\t !!! Skip job launch. Bye bye !!!\n ")

    # - - -  launch jobs - - - - - -
    logging.info(f"Launch heatmap-generation jobs into slurm.")
    jid_list = []
    for sbatch_file in sbatch_heatmap_file_list:
        out = subprocess.run(["sbatch", "--parsable", sbatch_file], 
                             capture_output=True)
        jid_list.append(out.stdout.decode('utf-8').strip())

    # prepare model-training/predict command to run after all create_heatmaps jobs complete
    scone_sbatch_cmd = ['sbatch' ]
    scone_sbatch_cmd.append(f"--dependency=afterok:{':'.join(jid_list)}")
    scone_sbatch_cmd.append(sbatch_scone_file)

    mode = SCONE_CONFIG['mode']
    logging.info(f"Launch {mode} job with command: \n{scone_sbatch_cmd}\n")
    subprocess.run(scone_sbatch_cmd)

    logging.info(f"")
    logging.info(f"Finished launching heatmap + {mode} jobs; wait for jobs to finish.\n")

    # ==== END ===
