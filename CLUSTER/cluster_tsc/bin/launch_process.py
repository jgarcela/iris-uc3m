#!/usr/bin/env python3

import os
from pathlib import Path
import sys
import time
import json
import numpy as np
import pwd
import smtplib
import ssl
import signal
import re
import subprocess
import argparse
from urllib.parse import urlparse

from http.client import HTTPSConnection
from base64 import b64encode


from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from warnings import warn as warning

SCRIPT_DIR=Path(__file__).resolve().parent
sys.path.insert(0,str(SCRIPT_DIR))
CWD=os.getcwd()
sys.path.insert(0,CWD)

import Scheduler
from Scheduler import SchedulerMaster

__VERSION__ = "5.0.0"
__MAYOR_VERSION__ = 5
__MINOR_VERSION__ = 0
__ISSUE_VERSION__ = 0

__MINIMUM_SCHEDULER_MAYOR_VERSION__ = 5
__MINIMUM_SCHEDULER_MINOR_VERSION__ = 0

HProceso = None

class TCConfig:
    config_file = None
    config = None

    def __init__(self, configfile):
        self.config_file = configfile
        with open(self.config_file,encoding='utf-8') as HConfig:
            self.config = json.load(HConfig)

    def parse(self,section,item):
        try:
            return self.config[section][item]
        except:
            return None

    def parseInt(self,section,item,default=0):
        try:
            return int(self.parse(section,item))
        except Exception as e:
            return default

    def parseFloat(self,section,item,default=0.0):
        try:
            return float(self.parse(section,item))
        except Exception as e:
            return default

    def parseFunction(self,section,item):
        return eval(self.parse(section,item))

    def parseArray(self,section,item):
        return self.parse(section,item)

    def parseSection(self,section,default=None):
        try:
            return self.config[section]
        except Exception as e:
            warning(f"Error parsing section '{section}': {e}")
            return default

    


class TCResources:
    CPU=0
    GPUS=0
    MEMORY=0
    DISK=None
    config = None

    def __init__(self,config):
        self.config = config
        self.parseResources(self.config)

    def parseResources(self,config):
        self.CPU=config.parseInt('resources','CPUS',default=1)
        self.GPUS=config.parseInt('resources','GPUS',default=0)
        self.MEMORY=config.parseInt('resources','MEMORY',default=20480)
        self.DISK=config.parse('resources','DISK')

    def getNumCPUS(self):
        return self.CPU
    
    def getMemory(self):
        return self.MEMORY

    def getGPUS(self):
        return self.GPUS
    
    def getDisk(self):
        return(self.DISK)

class TCAttributes:
    attributes = {}
    config = None
    def __init__(self,config):
        self.config = config
        self.parseAttributes(self.config)

    def parseAttributes(self,config):
        if config.parseSection('attributes') is None:
            self.attributes = {}
        else:
            self.attributes = self.config.parseSection('attributes',default={})

    def getAttributes(self) -> dict:
        return self.attributes

class TCTasks:
    cmds            = []
    basecmd         = None
    var_parameters  = {}
    parameters      = []
    task_basename   = ""

    def __init__(self,
                 config,
                 debug=False
                 ):
        self.basecmd        = config.parse('tasks','cmdbase')
        self.parameters     = config.parse('tasks','parameters')
        self.var_parameters = config.parse("tasks","var_parameters")
        self.task_basename  = config.parse("tasks","task_basename")
        self.debug          = debug
        self.setup_cmds()

    def setup_cmds(self):
        parameters_set=[dict(self.parameters)]
        for k in self.var_parameters.keys():
            val_parameter = self.var_parameters[k]
            if (isinstance(val_parameter,list)):
                iter_parameter = val_parameter
            elif (isinstance(val_parameter,range)):
                iter_parameter = list(val_parameter)
            elif (isinstance(val_parameter,np.ndarray)):
                iter_parameter = list(val_parameter)
            elif (isinstance(val_parameter,dict)):
                iter_parameter = list(val_parameter.values())
            elif (isinstance(val_parameter,str)):
                data_parameter = eval(val_parameter)
                if (isinstance(data_parameter,list)):
                    iter_parameter = data_parameter
                elif (isinstance(data_parameter,range)):
                    iter_parameter = list(data_parameter)
                elif (isinstance(data_parameter,np.ndarray)):
                    iter_parameter = list(data_parameter)
                elif (isinstance(data_parameter,dict)):
                    iter_parameter = list(data_parameter.values())
                else:
                    warning(f"Tipo de parametro variable no soportado para '{k}': {type(data_parameter)}, se tratara como string")
                    iter_parameter = [eval(str(val_parameter))]                
            else:
                warning(f"Tipo de parametro variable no soportado para '{k}': {type(val_parameter)}, se tratara como string")
                iter_parameter = [str(val_parameter)]
            new_parameters_set=[]
            for p1 in parameters_set:
                for new_parameter in iter_parameter:
                    new_p1 = dict(p1)
                    new_p1[k] = new_parameter
                    new_parameters_set.append(new_p1)
            parameters_set = new_parameters_set
        cmds = []
        print("Generando comandos a partir de la plantilla: {0}".format(self.basecmd))
        print("Parametros fijos: {0}".format(self.parameters))
        print("Parametros variables: {0}".format(self.var_parameters))
        print("Numero de combinaciones de parametros: {0}".format(len(parameters_set)))
        for params in parameters_set:
            if self.debug:
                print ("Parametros: {0}".format(params))
            new_cmd = self.basecmd.format(**params)
            cmds.append(new_cmd)
        self.cmds = list(set(cmds))
        if self.debug:
            print(f"Comandos ({len(cmds)}): {cmds[0]}")
        else:
            print(f"Numero de tareas a ejecutar: {len(cmds)}")

    def get_taskname (self):
        return self.task_basename

    def get_cmds(self):
        return self.cmds

class TCFramework:
    name = ""
    max_num_procs_simultaneos = -1
    max_failed_tasks = 0
    max_num_task_per_hosts = 0
    enviroment_vars  = {}
    uris             = []
    stdout           = None
    stderr           = None
    notification_at_end = 0
    max_retries_tasks = 0

    def __init__(self,config):
        self.name                       = config.parse("framework","name")
        self.max_num_procs_simultaneos  = config.parseInt("framework","max_num_procs_simultaneos",-1)
        self.max_failed_tasks           = config.parseInt("framework","max_failed_tasks",0)
        self.max_retries_tasks          = config.parseInt("framework","max_retries_tasks",0)
        self.enviroment_vars            = config.parse("framework","environment_vars")
        self.uris                       = config.parse("framework","uris")
        self.stdout                     = config.parse("framework","stdout")
        self.stderr                     = config.parse("framework","stderr")
        self.notification               = config.parseInt("framework","notification_at_end",0)
        self.max_num_task_per_hosts     = config.parseInt("framework","max_num_task_per_hosts",0)

    def getName(self):
        return self.name
    
    def getMaxNumProcsSimultaneos(self):
        return self.max_num_procs_simultaneos

    def getMaxFailedTasks(self) :
        return self.max_failed_tasks
    
    def getMaxRetriesTasks(self) :
        return self.max_retries_tasks

    def getEnviromentVars(self):
        return self.enviroment_vars

    def getURIS(self):
        return self.uris

    def getMaxNumTaskPerHost(self):
        return self.max_num_task_per_hosts

class TCExecutor:
    scheduler       = None
    configured      = False
    startup_time    = None
    framework       = None
    driver          = None
    tasks           = None
    resources       = None
    attributes      = None
    status          = 0
    exit_val        = 0
    config          = None
    tasksdef        = None
    resourcesdef    = None
    frameworkdef    = None
    stdout          = ""
    stderr          = ""
    masteruri       = ""
    notify          = False
    email           = ""
    roles           = []
    userId          = list()

    def __init__(self,config,
                notify=None, 
                email=None, 
                stdout=None, 
                stderr=None, 
                master=None,
                account=None,
                queue=None,
                workdir=None,
                logdir=None,
                basename_log=None,
                debug=None
                ):
        
        self.config = config
        self.resourcesdef   = TCResources(self.config)
        self.tasksdef       = TCTasks(self.config)
        self.frameworkdef   = TCFramework(self.config)
        self.attributesdef  = TCAttributes(self.config)
        self.masteruri      = config.parse("cluster","master") if master is None else master
        self.email          = config.parse("framework","email") if email is None else email
        self.notify         = config.parseInt("framework","notification",0)==1 if notify is None else notify
        self.stdout         = config.parse("framework","stdout") if stdout is None else stdout
        self.stderr         = config.parse("framework","stderr") if stderr is None else stderr
        self.userId         = pwd.getpwuid( os.getuid() ).pw_name
        self.workdir        = config.parse("framework","workdir") if workdir is None else workdir
        self.queue          = config.parse("cluster","queue") if queue is None else queue
        self.logdir         = config.parse("framework","logdir") if logdir is None else logdir
        self.basename_log   = config.parse("framework","basename_log") if basename_log is None else basename_log
        self.debug          = config.parseInt("framework","debug",0) if debug is None else debug
        self.account        = config.parse("cluster","account") if account is None else account
        self.workdir        = config.parse("framework","workdir") if workdir is None else workdir

        if self.workdir is None or self.workdir == "":
            self.workdir = os.getcwd()
            print(f"Directorio de trabajo (workdir) no especificado en el fichero de configuracion, se usara '{self.workdir}' por defecto")

        if self.masteruri is None:
            warning("URI del master no especificada en el fichero de configuracion, se usara 'https://subserver2.tsc.uc3m.es:6820' por defecto")
            self.masteruri = "https://subserver2.tsc.uc3m.es:6820"
        else:
            parsed_uri = urlparse(self.masteruri)
            if not parsed_uri.scheme or not parsed_uri.netloc:
                warning(f"URI del master '{self.masteruri}' no es valida, se usara 'https://subserver2.tsc.uc3m.es:6820' por defecto")
                self.masteruri = "https://subserver2.tsc.uc3m.es:6820"
            else:
                if parsed_uri.scheme == "zk":
                    warning(f"URI del master '{self.masteruri}' con esquema 'zk' es antigua, se usara 'https://subserver2.tsc.uc3m.es:6820' por defecto")
                    self.masteruri = "https://subserver2.tsc.uc3m.es:6820"
                    
        if self.queue is None:
            if self.resourcesdef.getGPUS() > 0:
                self.queue = "gpus"
            else:
                self.queue = "main"
            print(f"Cola de trabajo (queue) no especificada en el fichero de configuracion, se usara '{self.queue}' por defecto")
        
        if self.stdout is not None and self.stdout != "":
            sys.stdout=open(self.stdout,'w',encoding='utf-8')

        if self.stderr is not None and self.stderr != "":
            sys.stderr=open(self.stderr,'w',encoding='utf-8')
        return

    def setup(self) ->  None :

        self.scheduler = SchedulerMaster(
                                        self.masteruri,
                                        self.userId,
                                        self.tasksdef.get_cmds(),
                                        self.resourcesdef.getNumCPUS(),
                                        self.resourcesdef.getMemory(),
                                        self.frameworkdef.getEnviromentVars(),
                                        account=self.account,
                                        URIS=self.frameworkdef.getURIS(),
                                        basename=self.tasksdef.get_taskname(),
                                        GPUs=self.resourcesdef.getGPUS(),
                                        attributes=self.attributesdef.getAttributes(),
                                        queue=self.queue,
                                        workdir=self.workdir,
                                        logdir=self.logdir,
                                        basename_log=self.basename_log,
                                        debug=self.debug
                )
        if self.debug:
            print(f"comandos: {self.tasksdef.get_cmds()}\n")
        else:             
            print(f"Numero de tareas a ejecutar: {len(self.tasksdef.get_cmds())}\n")
        self.scheduler.setAttributes(self.attributesdef.getAttributes())
        self.scheduler.setMaxNumTaskRunning(self.frameworkdef.getMaxNumProcsSimultaneos())
        self.scheduler.setMaxNumTaskPerHost(self.frameworkdef.getMaxNumTaskPerHost())
        self.scheduler.setMaxFailedTasks(self.frameworkdef.getMaxFailedTasks())
        self.scheduler.setMaxTaskRetries(self.frameworkdef.getMaxRetriesTasks())
        self.configured=True
        
    def start(self) -> None:
        if not self.configured:
            self.setup()
        self.startup_time = time.time()
        try:
            self.scheduler.start_framework()
            print(f"Iniciando ejecucion\nMaster status: {self.scheduler.getSchedulerStatus()}")
        except Exception as error:
            e = sys.exc_info()[0]
            print( f"Error: {e}")
            print( f"Exception message: {repr(error)}")
            

    def send_mail(self,target=None,msg="",subject=""):
        userId = self.userId
        sender_email= f"{userId}@tsc.uc3m.es"
        smtp_server = "hermod.tsc.uc3m.es"
        port = 587  # For starttls
        msg = f"Tarea Finalizada: {self.frameworkdef.getName()}\n\n{msg}"

        # Try to log in to server and send email
        try:
            server = smtplib.SMTP(smtp_server,port)
            server.ehlo() # Can be omitted
            server.starttls() # Secure the connection
            server.ehlo() # Can be omitted
            message = MIMEMultipart("alternative")
            message["Subject"] = subject
            message["From"] = "cluster@tsc.uc3m.es"
            message["Reply-to"] = "noreply@tsc.uc3m.es"
            message["To"] = target
            part1 = MIMEText(msg, "plain")
            message.attach(part1)
            server.sendmail(sender_email, target, message.as_string())
        except Exception as e:
            # Print any error messages to stdout
            print(e)
        finally:
            server.quit()

    def CloseDriver(self):
        self.scheduler.CloseDriver()

    def StopDriver(self):
        self.scheduler.stop_framework()

    def main(self):
        self.setup()
   
        self.start()
        
        counter = 0
        while (self.scheduler.getSchedulerStatus() == SchedulerMaster.SCHEDULER_RUNNING):
            counter += 1
            if (counter % 60 )==0:
                print(self.scheduler.getStatistics())
            self.scheduler.pool_tasks()
            time.sleep(1)
        if self.notify:
            subject=f"Ejecucion finalizada en Granja de HPC: {self.frameworkdef.getName()}"
            self.send_mail(target=self.email,msg=self.scheduler.getStatistics(),subject=subject)


def Hsignal_INT(sig, frame):
    global HProceso
    print("Captura de SIGNAL")
    if not HProceso == None:
        print("Parando proceso de cluster con SIG: {0}".format(sig))
        if sig == 2:
            HProceso.StopDriver()
        else:
            HProceso.CloseDriver()


def help():
    print(f"\
        launch_process: (ver {__VERSION__}\n\
        -c config.json\n\
            Fichero de configuracion de Framework\n\
            Parametro obligatorio\n\
        -h : ensena este mensaje\n""")

def main():
    global HProceso

    parser = argparse.ArgumentParser(description='Lanza un proceso en la granja de HPC de TSC a partir de una definicion en un fichero json')
    parser.add_argument("-c", "--config", help="Fichero de configuracion de Framework", required=True)
    parser.add_argument("--notify", help="Envia un email al finalizar la ejecucion", action="store_true")
    parser.add_argument("--email", help="Email al que enviar la notificacion al finalizar la ejecucion (requerido si se especifica --notify)", type=str)
    parser.add_argument("--stdout", help="Fichero donde redirigir la salida estandar del proceso", type=str)
    parser.add_argument("--stderr", help="Fichero donde redirigir la salida de errores del proceso", type=str)
    parser.add_argument("--master", help="URI del master del cluster (ejemplo: https://subserver2.tsc.uc3m.es:6820)", type=str)
    parser.add_argument("--account", help="Cuenta a usar en el cluster (ejemplo: general, gpu)", type=str)
    parser.add_argument("--queue", help="Cola de ejecucion", type=str)
    parser.add_argument("--workdir", help="Directorio de trabajo del proceso en el cluster", type=str)
    parser.add_argument("--logdir", help="Directorio donde se almacenaran los logs del proceso en el cluster", type=str)
    parser.add_argument("--basename_log", help="Nombre base de los ficheros de log del proceso en el cluster", type=str)
    parser.add_argument("--version", action="version", version=f"%(prog)s {__VERSION__}")
    parser.add_argument("--debug", help="Activa el modo debug", action="store_true")
    parser.add_argument("--scheduler-version", action="version", version=f"Scheduler library version: {Scheduler.__LIB_MAYOR_VERSION__}.{Scheduler.__LIB_MINOR_VERSION__}")    

    args = parser.parse_args()
    if Scheduler.__LIB_MAYOR_VERSION__ < __MINIMUM_SCHEDULER_MAYOR_VERSION__ :
        print(f"Minima version de la libreria Scheduler: {__MINIMUM_SCHEDULER_MAYOR_VERSION__}")
        sys.exit(-1)

    configfile = args.config

    if args.notify:
        if not args.email:
            print("Error: --email es requerido si se especifica --notify")
            sys.exit(-1)
    if args.email:
        print(f"Notificacion al finalizar la ejecucion se enviara a: {args.email}")

    if not configfile is None:
        print(f"Iniciando ejecucion de proceso con definicion: {configfile}")
        print("Registrando captura de INT")
        signal.signal(signal.SIGINT,Hsignal_INT)
        signal.signal(signal.SIGTERM,Hsignal_INT)
        signal.signal(signal.SIGQUIT,Hsignal_INT)
        HProceso = TCExecutor(TCConfig(configfile),
                             notify=args.notify,
                             email=args.email,
                             stdout=args.stdout,
                             stderr=args.stderr,
                             master=args.master,
                             queue=args.queue,
                             account=args.account,
                             workdir=args.workdir,
                             logdir=args.logdir,
                             basename_log=args.basename_log,
                             debug=args.debug
                             )
        print("Lanzando ejecucion")
        HProceso.main()
    else:
        help()




if __name__ == "__main__":
    main()