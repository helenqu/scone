#!/usr/bin/env python
#
#
# Feb 29 2024 RK 
#   + replace print with logging.info
#   + add 'heamaps_logfile'  to list of paths to expandvars (in load_config)
#   + add 'heamaps_donefile' to list of paths to expandvars (in load_config)
#   + define jobid[_list] that should match the heatmap_[jobid] files.
#   + new methods  write_log_fail_message and write_donefile
#   + replace .format with more reliable f-strings
#   + new config input keys prescale_heatmap  ;
#      keep legacy key sim_fraction 
#   + new/optional --slurm_id and --nslurm_tot input (more efficient)

import os, sys, yaml, logging, glob
import argparse
import subprocess
import multiprocessing as mp
from create_heatmaps.manager import CreateHeatmapsManager

from   scone_utils import *
import scone_utils as util


# declare hard-wired globals
MSG_DONEFILE_SUCCESS = "CREATE HEATMAPS SUCCESS"
MSG_DONEFILE_FAILURE = "CREATE HEATMAPS FAILURE"

heatmap_summary_wildcard  =  "heatmap*summary"
heatmap_file_wildcard     =  "heatmap*tfrecord"


# ============== begin ============

def get_args():
    parser = argparse.ArgumentParser(description='create heatmaps from lightcurve data')

    parser.add_argument('--config_path', type=str, 
                        help='absolute or relative path to your yml config file, i.e. "/user/files/create_heatmaps_config.yml"')

    # input explicit start and end jobid (original feature)
    parser.add_argument('--start', type=int, default=None,
                        help='metadata/lcdata files index to start processing at')
    parser.add_argument('--end', type=int, default=None,
                        help='metadata/lcdata files index to stop processing at')

    # ... OR ...

    # input slurm_id and nslurm_tot and internally figure out jobids (RK - mar 2024)
    # This input option requires no user knowledge of how many FITS/PHOT files.
    parser.add_argument('--slurm_id', type=int, default=None,
                        help='jobid from 0 to njobtot-1')
    parser.add_argument('--nslurm_tot', type=int, default=None,
                        help='total number or slurm jobs')

    # optional redirect of heatmaps subdir
    parser.add_argument('--heatmaps_subdir', type=str, default=HEATMAPS_SUBDIR_DEFAULT,
                        help='alternate heatmaps subdir output')

    args = parser.parse_args()

    # define and append string to identify job in logging (not used for computation)
    if args.start is not None:
        args.jobid_string = f"jobid_{args.start}-{args.end}"  
    else:
        args.jobid_string = f"slurm_id{args.slurm_id:02d}"

    return args


def load_config(args):

    config_path = args.config_path

    key_expandvar_list = [ 'input_data_paths', 'lcdata_paths', 'metadata_paths', 
                           'heatmaps_path', 'output_path', 
                           'heatmaps_logfile', 'heatmaps_donefile'  ]

    config = util.load_config_expandvars(config_path, key_expandvar_list)
    
    # if sim data folders are provided, read list file(s) and set internal array
    # for each PHOT.FITS and *HEAD.FITS file
    load_lcdata_metadata(config)

    key = 'trained_model'
    config[key] = config.setdefault(key,None)
    if key in config:
        config[key] = os.path.expandvars(str(config[key]))  # str protects null
        # Q: should we abort if this path does not exist ? YES

    # if no prescale keys are given, set them to 1.
    key = 'prescale_heatmap' 
    config[key] = config.setdefault(key,1)

    key = 'sim_fraction'  # legacy key for old run.py
    config[key] = config.setdefault(key,1)
    
    key = 'heatmaps_path'
    config[key] = config['output_path'] + '/' + args.heatmaps_subdir

    return config

def load_lcdata_metadata(config):

    # Created Mar 2024 by R.Kessler
    # If individual HEAD.FITS and PHOT.FITS are not already read into config,
    # read VERSION.LIST file for each 'input_data_paths' and store all of the
    # HEAD and PHOT paths. This allows input config file to contain either
    #  + list of sim data folders ('input_data_paths' key)
    #          or 
    #  + list of every PHOT and HEAD file ('lcdata_paths' and 'metadata_paths' keys)
    #

    key_path   = 'input_data_paths'  # expected input
    key_meta   = 'metadata_paths'    # expected output (unless it already exists)
    key_lcdata = 'lcdata_paths'      # idem

    if key_meta in config and key_lcdata in config: 
        return

    config[key_meta]   = []
    config[key_lcdata] = []
    n_load = 0

    if key_path in config:
        for data_path in config[key_path]:
            data_path = os.path.expandvars(data_path)
            version   = os.path.basename(data_path)
            list_file = f"{data_path}/{version}.LIST" 
            with open(list_file,"rt") as l:
                meta_list = l.readlines()
                for meta in meta_list:
                    meta   = data_path + '/' + meta.rstrip() + '.gz'
                    lcdata = meta.replace("HEAD.FITS", "PHOT.FITS")
                    config[key_meta].append(meta)
                    config[key_lcdata].append(lcdata)
                    n_load += 1

        logging.info("Stored path for {n_load} PHOT.FITS files (lcdata_paths).")
        logging.info("Stored path for {n_load} HEAD.FITS files (meta_paths).")

    return

def write_log_fail_message(args, config, failed_jobid_list):

    # xxxxxx mark delete Mar 3 2024 xxxxxxxxx
    # write error message to log file.
    #heatmaps_path = config["heatmaps_path"]
    #default_logfile = os.path.join(heatmaps_path, 
    #                               f"create_heatmaps__{os.path.basename(args.config_path).split('.')[0]}.log")
    #logfile_path = config.get("heatmaps_logfile", default_logfile)
    # xxxxxxxxxx end mark xxxxxxxxxxx


    msg  = f"\n{MSG_DONEFILE_FAILURE}\n"
    msg += f"\t Indices of failed create heatmaps jobs: {failed_jobid_list}\n"
    msg += f"\t For failed indices, check LC data files or metadata in config yml at \n" \
           f"\t\t {args.config_path}\n"
    msg += f"\t see above for logs\n"
    logging.info(f"{msg}")
        
    return


def get_heatmap_file_list(config):
    heatmaps_path       = config['heatmaps_path']
    summary_list        = glob.glob1(heatmaps_path, heatmap_summary_wildcard)
    heatmap_list        = glob.glob1(heatmaps_path, heatmap_file_wildcard)
    return  heatmap_list, summary_list

def all_heatmaps_done(args, config):
    heatmap_list, summary_list  = get_heatmap_file_list(config)
    n_summ_found  = len(summary_list)
    n_summ_expect = len(config['lcdata_paths'])    
    all_done      = ( n_summ_found == n_summ_expect )
    
    logging.info(f"{args.jobid_string}: Found {n_summ_found} of {n_summ_expect} summary files.")

    return all_done

def write_final_summary_file(args, config):

    # Created Mar 1 2024 by R.Kessler
    # Read each  heatmap*summary file, sum CPU, sum NLC per type;
    # then write grand summary to a single file that can be read by
    # other pipeline components.

    # scoop up list of summary files .xyz
    heatmap_list, summary_list  = get_heatmap_file_list(config)
    n_heatmap_found = len(heatmap_list)
    n_summ_found    = len(summary_list)
    n_summ_expect   = len(config['lcdata_paths'])    
    
    if n_summ_found == 0 : return

    # - - - - -
    heatmaps_path       = config['heatmaps_path']
    final_summary_file  = heatmaps_path + '/' + SCONE_SUMMARY_FILE
    logging.info(f"{args.jobid_string} " \
                 f"Create {SCONE_SUMMARY_FILE} from {n_summ_found} heatmap*.summary files.")
    
    # init things that are summed over summary files
    cpu_sum_minutes      = 0.0
    nlc_sum_dict         = {}
    lcdata_dir_unique    = []

    for summ_file in summary_list:
        summ_file_path = f"{heatmaps_path}/{summ_file}"
        with open(summ_file_path, "rt") as s:
            summary_info = yaml.load(s, Loader=yaml.Loader) 
            
            # append list of unique data directories
            dir_name = os.path.dirname(summary_info['LCDATA_PATH'])
            if dir_name not in lcdata_dir_unique:
                lcdata_dir_unique.append(dir_name)

            # sum nlc for each true type; avoid crash if NLC_dict is empty.
            NLC_dict = summary_info['NLC']
            if NLC_dict is not None:
                for lctype, nlc in NLC_dict.items():
                    if lctype not in nlc_sum_dict: nlc_sum_dict[lctype] = 0
                    nlc_sum_dict[lctype] += int(nlc)

            # sum cpu
            cpu_sum_minutes += float(summary_info['CPU'])
    

    # - - - - - - 
    # create final summary file

    with open(final_summary_file,"wt") as s:

        s.write(f"PROGRAM_CLASS:      {PROGRAM_CLASS_HEATMAPS}\n")
        s.write(f"N_HEATMAP_FILE:     {n_heatmap_found}    # expect {n_summ_expect}\n")
        s.write(f"N_HEATMAP_SUMMARY:  {n_summ_found}    # expect {n_summ_expect} \n")

        cpu_sum_hr = cpu_sum_minutes/60.0                            
        s.write(f"CPU_SUM:        {cpu_sum_hr:.2f}   # hr \n")

        s.write(f"SIM_FRACTION:      {config['sim_fraction']}  \n")
        s.write(f"PRESCALE_HEATMAP:  {config['prescale_heatmap']} \n")

        s.write("N_LC: \n")
        for lctype, nlc in nlc_sum_dict.items():
            s.write(f"  {lctype}:  {nlc}    # true type: n_lc in heatmaps (for training)\n")

        # write all directories used (not FITS files; just folders)
        s.write(f"\n")
        s.write(f"INPUT_DATA_DIRS: \n")
        for dir_name in lcdata_dir_unique:
            s.write(f"  - {dir_name}\n")

    return

def write_done_file(config, donefile_info):

    # Write done.txt file to communicate "all done" for higher level pipelines.


    default_donefile = os.path.join(config["heatmaps_path"], SBATCH_DONE_FILE)
    donefile_path    = config.get("heatmaps_donefile", default_donefile)

    with open(donefile_path, "w+") as donefile:
        donefile.write(donefile_info)

    return


def create_heatmaps(config, index):
    CreateHeatmapsManager().run(config, index)

# ===================================================
if __name__ == "__main__":

    util.setup_logging()

    util.print_job_command()

    args   = get_args()

    config = load_config(args)

    proc_list  = []
    jobid_list = []


    if args.start is not None:
        # orignal/legacy method; beware unbalanced load per cpu
        for jobid in range(args.start, args.end):
            proc = mp.Process(target=create_heatmaps, args=(config, jobid))
            proc.start()
            logging.info(f"\t started heatmap jobid {jobid}")
            proc_list.append(proc)
            jobid_list.append(jobid)
    else:
        # optional refactor; requires no user-knowledge of file count,
        # and balances CPU load
        n_lcdata = len(config['lcdata_paths'])
        for jobid in range(0,n_lcdata):
            if jobid % args.nslurm_tot == args.slurm_id:
                proc = mp.Process(target=create_heatmaps, args=(config, jobid))
                proc.start()
                logging.info(f"\t started heatmap jobid {jobid}")
                proc_list.append(proc)
                jobid_list.append(jobid)

    # - - - - - -
    njob_submit = len(jobid_list)
    logging.info(f"{args.jobid_string}: all {njob_submit} heatmap jobs submitted.")
    
    for proc, jobid in zip(proc_list, jobid_list) :
        proc.join()              # wait until procs are done
        logging.info(f"\t heatmap jobid {jobid} is done.")

    
    # - - - - 
    # check for failures
    failed_jobid_list = []
    for proc, jobid in zip(proc_list,jobid_list):
        if proc.exitcode != 0:
            failed_jobid_list.append(jobid)

    if len(failed_jobid_list) == 0:
        donefile_info = MSG_DONEFILE_SUCCESS
    else:
        donefile_info = MSG_DONEFILE_FAILURE
        write_log_fail_message(args, config, failed_jobid_list)  

    # - - - - -  - -
    # always update final summary file 
    write_final_summary_file(args, config)

    # - - - - - - -
    # declare final DONE when all heatmaps are there
    if all_heatmaps_done(args, config):

        # write done stamp for higher level pipeline
        write_done_file(config, donefile_info)  

        # tar up summary files; then remove them to cleanup
        util.compress_files(+1, config['heatmaps_path'], heatmap_summary_wildcard,
                       "heatmap_summaries", "" )

        util.compress_files(+1, config['heatmaps_path'], "create_heatmaps*",
                       "create_heatmaps", "" )

    # === END MAIN ===
