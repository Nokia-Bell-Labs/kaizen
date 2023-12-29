import argparse
import subprocess
import glob
import os
import re
import multiprocessing as mp
import signal
import time

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, required=True)
parser.add_argument("--script", type=str, required=True)
parser.add_argument("--start_task", type=int, default=0)
parser.add_argument("--num_tasks", type=int, default=5)
parser.add_argument("--gpus", type=str, default="0,1", help="Comma-separated list of gpu indices")
parser.add_argument("--tag", type=str, default="")


def consumer(consumer_id, job_queue, results_queue, setup_args=None):
    job = None
    try:
        print(f'{consumer_id} {os.getpid()} Starting consumer')
        results_queue.put({
            'type': 'init',
            'data': os.getpid()
        })
        # consumer_vars = consumer_setup(consumer_id, setup_args)
        while(True):
            job = job_queue.get(timeout=60)
            print(f"{consumer_id} {os.getpid()} get job") # {job}
            if job is None:
                break
            else:
                return_value = process_job(setup_args, job)
                results_queue.put({
                    'type': 'job_finished',
                    'data': return_value
                })
        print(f"{consumer_id} {os.getpid()} exitting loop")
    except Exception as e:  
        print("DEBUG: There is some issue with the below arg settings. \n Copy the args and recreate the error by running contrastive_training.py for further debugging!")
        print(e)
        print(job)
    finally:
        if job is not None:
            job_queue.put(job)
        results_queue.put(None)
        print(f'Stopping consumer {consumer_id} {os.getpid()}')

def process_job(consumer_vars, job):
    new_job = consumer_vars + " " + job
    run_command(new_job)

def run_command(command):
    p = subprocess.Popen(command, shell=True)       
    p.wait()

def main(args):
    gpus = args.gpus.split(",")
    
    task_folders = glob.glob(os.path.join(args.folder, "*"))
    print(task_folders)
    task_folder_lookup = {}
    for folder in task_folders:
        re_match = re.search("task(?P<task_idx>\d*)-.*", os.path.basename(folder))
        if re_match is not None:
            re_match_dict = re_match.groupdict()
            if "task_idx" in re_match_dict:
                task_idx = int(re_match_dict["task_idx"])
                models = sorted(glob.glob(os.path.join(folder, '*.ckpt')))
                if len(models) > 0:
                    task_folder_lookup[task_idx] = models[-1]
    
    all_jobs = []
    for task_idx in range(args.start_task, args.num_tasks):
        if task_idx in task_folder_lookup:
            job = f'TAG="T{task_idx}-{args.tag}" TASK_IDX={task_idx} NUM_TASKS={args.num_tasks} \
                PRETRAINED_PATH="{task_folder_lookup[task_idx]}" \
                bash {args.script}'
            all_jobs.append(job)
            # run_command(job)
        else:
            print("==============")
            print(f"[ERR] Cannot find model for task {task_idx}")
            print("==============")
    
    job_queue = mp.Queue()
    results_queue = mp.Queue()
    processes = [mp.Process(target=consumer, args=(i, job_queue, results_queue, f"CUDA_VISIBLE_DEVICES={gpus[i]}")) for i in range(len(gpus))]
    process_pids = []
    active_consumer_counter = len(processes)
    finished_job_counter = 0
    try:
        print("Putting jobs...")
        for job in all_jobs:
            job_queue.put(job)

        print(f"{os.getpid()} Server - starting consumers")
        for p in processes:
            p.start()
        for _ in range(len(processes)):
            job_queue.put(None)
        print(f"{os.getpid()} Server - finished putting jobs")

        while(True):
            job_results = results_queue.get()
            print("Job results", job_results)
            if job_results is None:
                active_consumer_counter -= 1
                if active_consumer_counter == 0:
                    break
            elif job_results['type'] == 'init':
                process_pids.append(job_results['data'])
            elif job_results['type'] == 'job_finished':
                finished_job_counter += 1
        
        print('Closing workers')
        for p in processes:
            p.join(60)
    except KeyboardInterrupt:
        print("Interrupted from Keyboard")
    finally:
        print("Terminating Processes", processes)
        for p in processes:
            try:
                p.terminate()
            except Exception as e: 
                print(f"Unable to terminate process {p}, processes might still exist.", e)
        print("Killing Processes", process_pids)
        for pid in process_pids:
            try:
                # os.kill(pid, signal.SIGTERM)
                os.kill(pid, signal.SIGKILL)
            except Exception as e:
                print(f"Unable to kill process {pid}, processes might still exist.", e)
        try:
            job_queue.close()
            results_queue.close()
        except Exception as e: 
            print("Unable to close job queues, processes might still be open", e)
        
    print(f'Finished, processed {finished_job_counter} jobs')

if __name__ == "__main__":
    args, unknown_args = parser.parse_known_args()
    main(args)

