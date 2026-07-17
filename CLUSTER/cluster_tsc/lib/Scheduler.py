#!/usr/bin/env python3

import os
import getpass
import uuid
import time
import subprocess
import re
import requests
import numpy as np



__LIB_MAYOR_VERSION__ = 5
__LIB_MINOR_VERSION__ = 0


class Credential:
    lifespan = 1800
    credential: str = None
    user: str = None
    creation_time: int = 0

    def __init__(self, user, lifespan=1800):
        self.user = user
        self.lifespan = lifespan
        self.create_credential()

    def create_credential(self)->str:

        self.creation_time = time.time()
        token=None
        try:
            result = subprocess.run(
            ["/usr/bin/scontrol", "token", f"username={self.user}", f"lifespan={self.lifespan}"],
                capture_output=True,
                text=True,
                check=True
            )
        except subprocess.CalledProcessError as e:
            print(f"Error al ejecutar scontrol: {e}")
            self.creation_time = 0
            return None
        m = re.search(r"SLURM_JWT=(.*)", result.stdout)
        if m:
            token = m.group(1)
        self.credential = token
        return self.credential
      
    def get_credential(self)->str:
        current_time = time.time()
        if self.credential is None  or (self.lifespan-(current_time-self.creation_time)<120):
            self.create_credential()
        return self.credential

class SlurmRestClient:
    base_script_cpus = """#!/bin/bash
#SBATCH --output={logdir}/{basename_log}_%j.out
#SBATCH --error={logdir}/{basename_log}_%j.err     
srun {cmd_line}
"""

    base_script_gpus = """#!/bin/bash
#SBATCH --output={logdir}/{basename_log}_%j.out
#SBATCH --error={logdir}/{basename_log}_%j.err
#SBATCH --gres=gpu:{gpus} 
srun {cmd_line}
"""
    def __init__(self, base_url, 
                 api_version="v0.0.41", 
                 user=None, 
                 token=None, 
                 timeout=30, 
                 lifespan=1800,
                 debug=False):
        self.base_url = base_url.rstrip("/")
        self.api_version = api_version
        self.user = user or os.getenv("SLURM_USER") or getpass.getuser()
        self.timeout = timeout
        self.debug = debug
        self.credential_manager = Credential(self.user, lifespan=lifespan)
        self.token = self.credential_manager.get_credential() if token is None else token

    def _headers(self):
        return {
            "Content-Type": "application/json",
            "X-SLURM-USER-NAME": self.user,
            "X-SLURM-USER-TOKEN": self.credential_manager.get_credential(),
        }

    def _url(self, path):
        return f"{self.base_url}/slurm/{self.api_version}{path}"

    def _request(self, method, path, **kwargs):
        response = requests.request(
            method,
            self._url(path),
            headers=self._headers(),
            timeout=self.timeout,
            **kwargs,
        )

        try:
            data = response.json()
        except ValueError:
            data = response.text

        if not response.ok:
            raise RuntimeError(
                f"Error REST Slurm\n"
                f"URL: {response.url}\n"
                f"HTTP: {response.status_code}\n"
                f"Respuesta: {data}"
            )

        return data

    def get_partitions(self):
        return self._request("GET", "/partitions")

    def submit_script(
        self,
        cmd_line: str,
        account: str = "general",
        name: str = "rest_job",
        workdir: str="/tmp",
        logdir: str = None,
        environment=None,
        attributes: dict = None,
        tasks: int =1,
        time_limit: int = 3600,
        partition: str ="main",
        cpus: int=1,
        gpus: int=0,
        memory_per_cpu: int = 2048,
        basename_log: str = "slurm_job"
    ):
        if environment is None:
            environment = [
                "PATH=/bin:/usr/bin:/sbin:/usr/sbin",
                "SLURM_EXPORT_ENV=ALL",
            ]
        elif isinstance(environment, dict):
            environment = [f"{key}={value}" for key, value in environment.items()]
        if logdir is None:
            logdir = workdir
        memory_string = f"{memory_per_cpu}M" if isinstance(memory_per_cpu, int) else memory_per_cpu
        if self.debug:
            print(f"Submitting job with {cpus} CPUs and {memory_string} memory per CPU")
            print(f"Script:\n{self.base_script_cpus.format(cmd_line=cmd_line,logdir=logdir,basename_log=basename_log)}")
        if basename_log is None or basename_log == "None" or basename_log == "":
            basename_log = "slurm_job"
        
        data = self.get_partitions()
        qos="normal"
        for s_partition in data.get("partitions", []):
            if self.debug:
                print(f"Partition: {s_partition.get("name")}:\t{s_partition.get("qos")}")
                print(f"Requested partition: {partition}")
            if s_partition.get("name") == partition:
                qos = s_partition.get("qos")["assigned"]
                if self.debug:
                    print(f"QOS: {qos}")

        if gpus > 0:
            script = self.base_script_gpus.format(cmd_line=cmd_line,logdir=logdir,basename_log=basename_log,gpus=gpus)
        else:
            script = self.base_script_cpus.format(cmd_line=cmd_line,logdir=logdir,basename_log=basename_log)

        payload = {
            "script": script,
            "job": {
                "name": name,
                "account": account,
                "current_working_directory": workdir,
                "environment": environment,
                "cpus_per_task": cpus,
                "memory_per_cpu": memory_string,                
                "tasks": int(tasks),
                "time_limit": int(time_limit),
                "partition": partition,
                "qos": qos,
                "standard_output": f"{logdir}/{basename_log}_%j.out",
                "standard_error": f"{logdir}/{basename_log}_%j.err"
            },
        }
        if gpus > 0:
            if attributes is not None:
                if "gpu_model" in attributes.keys():
                    gpu_model = attributes["gpu_model"]
                    payload["job"]["tres_per_task"] = f"gres/gpu:{gpu_model}:{gpus}"
                else:
                  payload["job"]["tres_per_task"] = f"gres/gpu:{gpus}"
            else:
                payload["job"]["tres_per_task"] = f"gres/gpu:{gpus}"
        data = self._request("POST", "/job/submit", json=payload)

        job_id = data.get("job_id") or data.get("jobId")
        return job_id, data

    def get_job(self, job_id):
        return self._request("GET", f"/job/{job_id}")

    def cancel_job(self, job_id):
        return self._request("DELETE", f"/job/{job_id}")
    
    
    def get_job_state(self, job_id):
        data = self.get_job(job_id)

        jobs = data.get("jobs", [])

        if not jobs:
            raise RuntimeError(f"No se encontró el job {job_id}")

        job = jobs[0]

        return {
            "job_id": job.get("job_id"),
            "job_state": job.get("job_state"),
            "exit_code": job.get("exit_code"),
            "state_reason": job.get("state_reason"),
        }

    def wait_job(self, job_id, poll_interval=5):
        final_states = {
            "COMPLETED",
            "FAILED",
            "CANCELLED",
            "TIMEOUT",
            "OUT_OF_MEMORY",
            "NODE_FAIL",
            "PREEMPTED",
        }

        while True:
            state = self.get_job_state(job_id)

            # print(state)

            job_state = state.get("job_state", "")

            if job_state in final_states:
                return state

            time.sleep(poll_interval)

class TaskInfo:
    TASK_NEVERRUN = 0
    TASK_RUNNING  = 1
    TASK_FINISHED = 2
    TASK_STOPED   = 3
    TASK_FAILED   = 4
    TASK_SUBMITED = 5
    TASK_PENDING  = 6

    def __init__(self,
                 cmd: str,
                 cpus: int = 1,
                 gpus: int = 0,
                 mem: int = 2048,
                 tid: str = "",
                 tname: str = ""):
        self.cmd = cmd
        self.hostname=""
        self.cpus=cpus
        self.gpus=gpus
        self.mem=mem
        self.tid=tid
        self.tname=tname
        self.uuid=""
        self.status=0
        self.num_runs=0
        self.name=tname
        self.start_time=None
        self.stop_time=None
        self.job_id=None
        self.response=""

    def change_status(self,new_status):
        self.status = new_status
        if (new_status == 1):
            self.num_runs += 1

    def change_hostname(self,new_host):
        self.hostname=new_host

    def change_uuid(self,new_uuid):
        self.uuid=new_uuid

    def change_name(self,new_name):
        self.tname = new_name

    def get_num_runs(self):
        return self.num_runs

    def getCmd(self):
        return self.cmd

    def getName(self):
        return self.name

    def setStartTime(self,start_time):
        self.start_time=start_time

    def setStopTime(self,stop_time):
        self.stop_time=stop_time

    def getStartTime(self):
        return self.start_time

    def getStopTime(self):
        return self.stop_time

    def getHostname(self):
        return self.hostname

    def set_job_id(self, job_id: int):
        self.job_id = job_id

    def get_job_id(self):
        return self.job_id
    
    def set_response(self, response):
        self.response = response

    def get_response(self):
        return self.response
    
class SchedulerMaster:

    SCHEDULER_STOPPED = 0
    SCHEDULER_RUNNING = 1
    SCHEDULER_PAUSED = 2

    CPU_LEVEL={
        'core2'         : 1,
        'nehalem'       : 2,
        'westmere'      : 3,
        'sandybridge'   : 4,
        'ivybridge'     : 5,
        'broadwell'     : 6,
        'skylake-avx512': 7,
        'cascadelake'   : 8,
        'icelake-server': 9
    }

    GPU_LEVEL={
        'gtx1080ti'     : 4,
        'rtx2080ti'     : 5,
        'rtx3080ti'     : 6,
        'rtx3090'       : 7,
        'rtxa4000'      : 8,
        'rtx4090'       : 9
    }

    FINAL_STATES = {
            "COMPLETED",
            "FAILED",
            "CANCELLED",
            "TIMEOUT",
            "OUT_OF_MEMORY",
            "NODE_FAIL",
            "PREEMPTED"
    }

    def __init__(self,
                 masteruri: str,
                 user: str,
                 cmd,
                 CPUs: int,
                 MEMORY: int,
                 ENVIRONMENT_VARS,
                 account: str = None,
                 URIS: list = None,
                 basename: str ="task",
                 logfile =None,
                 errfile = None,
                 GPUs: int = 0 ,
                 attributes: dict = None,
                 queue: str = None,
                 workdir: str = None,
                 logdir: str = None,
                 basename_log: str = None,
                 lifetime_credentials: int = 1800,
                 debug: bool = False
                 ):
        
        self.CPUs            = CPUs
        self.MEMORY          = MEMORY
        self.ENVIRONMENT     = ENVIRONMENT_VARS
        self.URIS            = URIS
        self.GPUs            = GPUs
        self.T_totales       = len(cmd)
        self.taskData        = {}
        self.tasksLaunched   = 0
        self.tasksFinished   = 0
        self.tasksFailed     = 0
        self.attributes      = {} if attributes is None else attributes
        self.taskrunning     = 0
        self.tasksubmited    = 0
        self.taskpending     = 0
        self.maxretries      = 0
        self.maxfailed       = 0
        self.maxlost         = -1
        self.driver          = None
        self.framework       = None
        self.multiroles      = False
        self.waiting_tasks   =[]
        self.running_tasks   ={}
        self.submited_tasks  ={}
        self.finnished_tasks =[]
        self.failed_tasks    =[]
        self.blacklisted_workers         =[]
        self.failcount_workers           ={}
        self.numtask_assigned_workers    ={}
        self.max_simultaneous_tasks = -1
        self.min_simultaneous_tasks = -1
        self.max_num_task_per_hosts = -1
        self.num_task_per_hosts = {}
        self.status          = self.SCHEDULER_STOPPED
        self.running_thread  = None
        self.uuid            = str(uuid.uuid4())
        self.active_cache    = False
        self.close_driver    = True
        self.LOGFILE_NAME    = logfile
        self.ERRFILE_NAME    = errfile
        self.start_time      = None
        self.stop_time       = None
        self.credentials     = None
        self.ack_offer       = True
        self.driver_status   = 0
        self.frameworkId     = ""
        self.gpus_minmem     = 0
        self.gpus_cap        = 0
        self.queue            = queue
        self.workdir          = workdir
        self.basename_log     = basename_log
        self.logdir           = logdir
        self.lifetime_credentials = lifetime_credentials
        self.debug            = debug
        self.account          = "general" if account is None else account
        

        if self.GPUs > 0:
            self.needs_gpu = True
            self.account = "gpu" if account is None else account
        else:
            self.needs_gpu = False
        if self.LOGFILE_NAME is not None:
            self.LOGFILE     = open(self.LOGFILE_NAME,"wt+",encoding="utf-8")
        else:
            self.LOGFILE     = None

        if self.ERRFILE_NAME is not None:
            self.ERRFILE     = open(self.ERRFILE_NAME,"wt+",encoding="utf-8")
        else:
            self.ERRFILE     = None

        self.update               = 0
        numtasks = len(cmd)
        self.writelog(f"Numero de comandos: {len(cmd)}")
        for tid,ind_cmd in enumerate(cmd):
            tname = f"{basename}-{tid:05d}/{numtasks}"
            tuid  = f"{self.uuid}-{tid:05d}"
            self.taskData[tuid] = TaskInfo(ind_cmd,
                                           cpus = CPUs,
                                           gpus = GPUs,
                                           mem = MEMORY,
                                           tid = tuid,
                                           tname = tname)
            self.waiting_tasks.insert(0,tuid)
        self.writelog(f"Numero de tareas a ejecutar: {len(self.waiting_tasks)}")        

        self.slurm_client = SlurmRestClient(masteruri, user=user, lifespan=self.lifetime_credentials)

    def writelog(self,msg):
        if self.LOGFILE is None:
            print(msg)
        else:
            print(msg,file=self.LOGFILE)

    def writerrror(self,msg):
        if self.ERRFILE is None:
            print(msg)
        else:
            print(msg,file=self.ERRFILE)

    def setLogFile(self,filelog):
        if self.LOGFILE is not None:
            self.LOGFILE.close()
        self.LOGFILE_NAME = filelog
        self.LOGFILE      = open(self.LOGFILE_NAME,"wt+",encoding="utf-8")

    def setErrorFile(self,filelog):
        if self.ERRFILE is not None:
            self.ERRFILE.close()
        self.ERRFILE_NAME = filelog
        self.ERRFILE      = open(self.ERRFILE_NAME,"wt+",encoding="utf-8")

    def closeLogFile(self):
        if self.LOGFILE is not None:
            self.LOGFILE.close()

    def closeErrorFile(self):
        if self.ERRFILE is not None:
            self.ERRFILE.close()

    def setMaxNumTaskRunning(self,num):
        self.max_simultaneous_tasks = num

    def setMinNumTaskRunning(self,num):
        self.min_simultaneous_tasks = num

    def setMaxTaskRetries(self,num):
        self.maxretries = num

    def setStartTime(self,st):
        self.start_time = st

    def setStopTime(self,st):
        self.stop_time = st

    def setMaxFailedTasks(self,num):
        self.maxfailed = num

    def setMaxNumTaskPerHost(self, num):
        self.max_num_task_per_hosts = num

    def setAttributes(self, attributes):
        self.attributes = attributes

    def addAttribute(self,attribute,value):
        self.attributes[attribute] = value

    def getAttribute(self,attribute):
        try:
            return self.attributes[attribute]
        except KeyError as e:
            self.writelog(f"Attribute {attribute} not found. Error: {e}")
            return None

    def getStartTime(self):
        return self.start_time

    def getStopTime(self):
        return self.stop_time

    def getTasksInfo(self):
        return self.taskData

    def getSchedulerStatus(self):
        return self.status

    def getMaxNumRetries(self):
        return self.maxretries

    def getMaxFailedTasks(self):
        return self.maxfailed

    def getMinNumTaskRunning(self):
        return self.min_simultaneous_tasks

    def getNumRunningTasks(self):
        return len(self.running_tasks)
    
    def launch_tasks(self):
        running_status = len(self.waiting_tasks) > 0
        running_status = running_status and ( self.max_simultaneous_tasks == -1 or self.taskrunning < self.max_simultaneous_tasks )
        running_status = running_status and self.status == self.SCHEDULER_RUNNING
        while running_status:
            task_id = self.waiting_tasks.pop()
            task_info = self.taskData[task_id]
            if self.debug:
                self.writelog(f"Submitting task {task_info.getName()} with command: {task_info.getCmd()}")
            if (task_info.gpus > 0) and not self.needs_gpu:
                self.writelog(f"Task {task_info.getName()} requires {task_info.gpus} GPUs but the scheduler is not configured to use GPUs. Marking task as failed.")
                task_info.change_status(TaskInfo.TASK_FAILED)
                self.tasksFailed += 1
                self.failed_tasks.append(task_id)
                if self.maxfailed != -1 and self.tasksFailed >= self.maxfailed:
                    self.status = self.SCHEDULER_STOPPED
                    self.stop_time = time.time()
                    self.writelog(f"Maximum number of failed tasks reached ({self.tasksFailed}). Framework stopped.")
                    self.stop_framework()
                continue
            try:
                job_id, response = self.slurm_client.submit_script(
                    account = self.account,
                    cmd_line=task_info.getCmd(),
                    name=task_info.getName(),
                    workdir=self.workdir,
                    environment=self.ENVIRONMENT,
                    attributes=self.attributes,
                    tasks=1,
                    partition=self.queue,  # Ajusta la partición si es necesario
                    cpus=task_info.cpus,
                    gpus=task_info.gpus,
                    memory_per_cpu=int(np.ceil(task_info.mem/task_info.cpus)),
                    logdir=self.logdir,
                    basename_log=self.basename_log
                )
                task_info.set_job_id(job_id)
                task_info.change_status(TaskInfo.TASK_SUBMITED)
                task_info.setStartTime(time.time())
                task_info.set_response(response)
                self.submited_tasks[job_id] = task_id
                self.writelog(f"Task {task_info.getName()} submitted with Job ID: {job_id}")
                if self.debug:
                    self.writelog(f"Info task {job_id}: {response}")
                self.tasksubmited += 1
            except Exception as e:
                # Hay que decidir si se vuelve a intentar o no, por ahora se marca como fallida directamente
                # self.waiting_tasks.append(task_id) 
                self.writelog(f"Error submitting task {task_info.getName()}: {e}")
                task_info.change_status(TaskInfo.TASK_FAILED)
                self.tasksFailed += 1
                self.failed_tasks.append(task_id)
                if self.maxfailed != -1 and self.tasksFailed >= self.maxfailed:
                    self.status = self.SCHEDULER_STOPPED
                    self.stop_time = time.time()
                    self.writelog(f"Maximum number of failed tasks reached ({self.tasksFailed}). Framework stopped.")
                    self.stop_framework()
            running_status = len(self.waiting_tasks) > 0
            running_status = running_status and ( self.max_simultaneous_tasks == -1 or self.taskrunning < self.max_simultaneous_tasks )
            running_status = running_status and self.status == self.SCHEDULER_RUNNING

    def start_framework(self):
        self.status = self.SCHEDULER_RUNNING
        self.writelog("Framework started")
        self.start_time = time.time()
        self.launch_tasks()

    def check_tasks(self):
        for job_id in list(self.submited_tasks.keys()):
            task_info = self.taskData[self.submited_tasks[job_id]]
            task_id = self.submited_tasks[job_id]
            try:
                if self.debug:
                    print(f"Checking status of task {task_info.getName()} with Job ID: {job_id}")
                state = self.slurm_client.get_job_state(job_id)
                if self.debug:
                    print(f"Status of task {task_info.getName()}: {state}")
                job_state = "UNKNOWN"
                job_state_A = state.get("job_state", "")
                if isinstance(job_state_A, list):
                    job_state = job_state_A[0] if len(job_state_A) > 0 else "UNKNOWN"
                else:
                    job_state = job_state_A
                if job_state == "RUNNING" :
                    task_info.change_status(TaskInfo.TASK_RUNNING)
                    task_info.setStartTime(time.time())
                    self.running_tasks[job_id] = task_id
                    del self.submited_tasks[job_id]
                    self.taskrunning += 1
                    self.taskpending -= 1
                    self.writelog(f"Task {task_info.getName()} is now running with Job ID: {job_id}")
                elif job_state in {"PENDING", "CONFIGURING"}:
                    task_info.change_status(TaskInfo.TASK_PENDING)
                    if self.debug:
                        self.writelog(f"Task {task_info.getName()} is pending or configuring.")
                elif job_state in self.FINAL_STATES:
                    task_info.setStopTime(time.time())
                    task_info.set_response(state)
                    if job_state == "COMPLETED":
                        task_info.change_status(TaskInfo.TASK_FINISHED)
                        self.finnished_tasks.append(task_id)
                        self.writelog(f"Task {task_info.getName()} finished successfully.")
                        self.tasksFinished += 1
                        self.tasksubmited -= 1
                    elif job_state == "CANCELLED":
                        task_info.change_status(TaskInfo.TASK_STOPED)
                        self.writelog(f"Task {task_info.getName()} was cancelled.")
                        self.tasksubmited -= 1
                        self.status = self.SCHEDULER_STOPPED
                    else:
                        task_info.change_status(TaskInfo.TASK_FAILED)
                        self.failed_tasks.append(task_id)
                        self.writelog(f"Task {task_info.getName()} failed with state: {job_state}.")
                        self.tasksubmited -= 1
                        self.tasksFailed += 1
                        if self.maxfailed != -1 and self.tasksFailed >= self.maxfailed:
                            self.status = self.SCHEDULER_STOPPED
                            self.stop_time = time.time()
                            self.writelog(f"Maximum number of failed tasks reached ({self.tasksFailed}). Framework stopped.")
                            self.stop_framework()
                    del self.submited_tasks[job_id]
                    if self.tasksFinished == self.T_totales:
                        self.status = self.SCHEDULER_STOPPED
                        self.stop_time = time.time()
                        self.writelog("All tasks finished. Framework stopped.")
                        break
            except Exception as e:
                self.writelog(f"Error checking status of task {task_info.getName()}: {e}")                
        for job_id in list(self.running_tasks.keys()):  # Iteramos sobre una copia de la lista
            task_info = self.taskData[self.running_tasks[job_id]]
            task_id = self.running_tasks[job_id]
            try:
                if self.debug:
                    print(f"Checking status of task {task_info.getName()} with Job ID: {job_id}")
                state = self.slurm_client.get_job_state(job_id)
                if self.debug:
                    print(f"Status of task {task_info.getName()}: {state}")
                job_state = "UNKNOWN"
                job_state_A = state.get("job_state", "")
                if isinstance(job_state_A, list):
                    job_state = job_state_A[0] if len(job_state_A) > 0 else "UNKNOWN"
                else:
                    job_state = job_state_A
                if job_state in self.FINAL_STATES:
                    task_info.setStopTime(time.time())
                    task_info.set_response(state)
                    if job_state == "COMPLETED":
                        task_info.change_status(TaskInfo.TASK_FINISHED)
                        self.finnished_tasks.append(task_id)
                        self.writelog(f"Task {task_info.getName()} finished successfully.")
                        self.tasksFinished += 1
                        self.taskrunning -= 1
                        del self.running_tasks[job_id]
                    elif job_state == "CANCELLED":
                        task_info.change_status(TaskInfo.TASK_STOPED)
                        self.writelog(f"Task {task_info.getName()} was cancelled.")
                        self.taskrunning -= 1
                        del self.running_tasks[job_id]
                        self.status = self.SCHEDULER_STOPPED
                    else:
                        task_info.change_status(TaskInfo.TASK_FAILED)
                        self.failed_tasks.append(task_id)
                        self.writelog(f"Task {task_info.getName()} failed with state: {job_state}.")
                        self.taskrunning -= 1
                        self.tasksFailed += 1
                        del self.running_tasks[job_id]
                        if self.maxfailed != -1 and self.tasksFailed >= self.maxfailed:
                            self.status = self.SCHEDULER_STOPPED
                            self.stop_time = time.time()
                            self.writelog(f"Maximum number of failed tasks reached ({self.tasksFailed}). Framework stopped.")
                            self.stop_framework()

                    if self.tasksFinished == self.T_totales:
                        self.status = self.SCHEDULER_STOPPED
                        self.stop_time = time.time()
                        self.writelog("All tasks finished. Framework stopped.")
                        break
                elif job_state == "RUNNING":
                    if task_info.status != TaskInfo.TASK_RUNNING:
                        task_info.setStartTime(time.time())
                        task_info.change_status(TaskInfo.TASK_RUNNING)
                        self.writelog(f"Task {task_info.getName()} is now running.")
                elif job_state in {"PENDING", "CONFIGURING", "COMPLETING"}:
                    if self.debug:
                        print(f"Task {task_info.getName()} is in state {job_state}.")
                    if task_info.status != TaskInfo.TASK_RUNNING:
                        task_info.change_status(TaskInfo.TASK_RUNNING)
                        self.writelog(f"Task {task_info.getName()} is pending or configuring.")
            except Exception as e:
                self.writelog(f"Error checking status of task {task_info.getName()}: {e}")

    def pool_tasks(self):
        self.check_tasks()
        if self.status == self.SCHEDULER_RUNNING:
            if self.max_simultaneous_tasks == -1 or self.taskrunning < self.max_simultaneous_tasks:
                if len(self.waiting_tasks) > 0:
                    self.launch_tasks ()

        return self.status

    def getStatistics(self):
        stats = {
            "total_tasks": self.T_totales,
            "tasks_finished": self.tasksFinished,
            "tasks_running": len(self.running_tasks),
            "tasks_submitted": len(self.submited_tasks),
            "tasks_pending": self.taskpending,
            "tasks_waiting": len(self.waiting_tasks),
            "tasks_failed": len(self.failed_tasks),
            "start_time": self.start_time,
            "stop_time": self.stop_time,
        }
        return stats
    
    def stop_framework(self):
        self.status = self.SCHEDULER_STOPPED
        if len(self.submited_tasks) > 0:
            for job_id in list(self.submited_tasks.keys()):
                task_id = self.submited_tasks[job_id]
                task_info = self.taskData[task_id]
                try:
                    self.slurm_client.cancel_job(job_id)
                    if self.debug:
                        self.writelog(f"Cancelled job {job_id} for task {task_info.getName()}")
                except Exception as e:
                    self.writelog(f"Error cancelling job {job_id} for task {task_info.getName()}: {e}")
        if len(self.running_tasks) > 0:
            for job_id in list(self.running_tasks.keys()):
                task_id = self.running_tasks[job_id]
                task_info = self.taskData[task_id]
                try:
                    self.slurm_client.cancel_job(job_id)
                    if self.debug:
                        self.writelog(f"Cancelled job {job_id} for task {task_info.getName()}")
                except Exception as e:
                    self.writelog(f"Error cancelling job {job_id} for task {task_info.getName()}: {e}")
        self.stop_time = time.time()
        self.writelog("Framework stopped")