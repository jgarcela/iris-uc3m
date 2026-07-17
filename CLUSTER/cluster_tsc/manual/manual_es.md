::: hero
::: wrap
# Manual de Uso del Cluster del Departamento de Teoría de la Señal y Comunicaciones

Guía práctica para lanzar trabajos, seleccionar particiones, usar recursos CPU/GPU y consultar el estado de la granja gestionada con SLURM.

::: meta
[Cluster: Scylla_DTSC_Cluster]{.chip} [Controladores: subserver1, subserver2]{.chip} [Gestor: SLURM]{.chip}
:::
:::
:::

::: layout
::: sidebar-inner
::: sidebar-title
Manual SLURM
:::

::: sidebar-subtitle
Índice
:::

[Introducción](#intro) [Acceso al sistema](#acceso) [Entornos de trabajo](#entornos) [Recursos disponibles](#recursos) [Tabla de decisión](#decision) [Barridos paramétricos JSON](#json-barridos) [Ejemplos de uso](#ejemplos) [Comandos básicos](#comandos) [Resolución de problemas](#problemas) [Interfaces web](#web) [Buenas prácticas](#buenas) [Plantilla recomendada](#plantilla)
:::

::: {.content role="main"}
::: {#intro .section}
## 1. Introducción

SLURM es el gestor de recursos y planificador de trabajos usado en el cluster del DTSC. Su función principal es repartir de forma ordenada los recursos compartidos de computación ---CPU, memoria, nodos y GPUs--- entre los usuarios y grupos de investigación.

::: {.grid .grid-4}
::: card
### Gestión centralizada

Evita conflictos entre usuarios y permite definir políticas de uso por particiones, cuentas y QoS.
:::

::: card
### Optimización

Ajusta la asignación de CPUs, memoria, tiempo y GPUs a las necesidades reales de cada trabajo.
:::

::: card
### Colas inteligentes

Prioriza trabajos cortos, depuración, trabajos normales, largos o GPU según la política del cluster.
:::

::: card
### Trazabilidad

Permite consultar quién ejecutó un trabajo, cuándo, con qué recursos y con qué resultado.
:::
:::
:::

::: {#acceso .section}
## 2. Acceso al sistema

Para lanzar procesos al cluster de SLURM, los usuarios deben conectarse a las máquinas autorizadas para su grupo de trabajo. Desde esas máquinas se preparan los scripts y se envían los trabajos con `sbatch`, `srun`, `salloc` o mediante la librería de lanzamiento basada en `launch_process.py` y `Scheduler.py`.

::: {.grid .grid-4}
::: card
### Grupo de Comunicaciones

`ceres`
:::

::: card
### Grupo ML4DS

`hator00` a `hator07`, `izanami`
:::

::: card
### Grupo de Procesado Multimedia

`amaterasu` o `kusanagi`
:::

::: card
### Grupo GTS

`meiga01` a `meiga06`, `arvak` o `skoll`
:::

::: card
### Otros

`medusa00`, `glenfiddish` o `macallan`
:::
:::

::: {.callout .warnbox}
**Importante:** no se debe entrar por SSH directamente a los nodos de cómputo para ejecutar procesos pesados. Los trabajos deben lanzarse mediante SLURM.
:::
:::

::: {#entornos .section}
## 3. Preparación de entornos de trabajo

Los usuarios pueden usar las librerías preinstaladas y optimizadas para cada máquina, o preparar entornos aislados de Python con `venv` o Conda.

Para entornos, datos y resultados se recomienda utilizar volúmenes compartidos como `/export/usuarios01/USER` o `/export/clusterdata/USER`, de forma que los ficheros estén disponibles desde los nodos donde se ejecuten las tareas.

### Entorno Python con venv

    mkdir -p /export/clusterdata/$USER/venvs
    python3 -m venv /export/clusterdata/$USER/venvs/mi_entorno
    source /export/clusterdata/$USER/venvs/mi_entorno/bin/activate
    python -m pip install --upgrade pip
    pip install numpy scipy pandas scikit-learn torch

### Entorno con Conda

    conda create -p /export/clusterdata/$USER/conda/envs/mi_entorno python=3.11
    conda activate /export/clusterdata/$USER/conda/envs/mi_entorno
    conda install numpy scipy pandas scikit-learn

::: callout
**Recomendación:** ubica los entornos en volúmenes compartidos como `/export/usuarios01/USER` o `/export/clusterdata/USER`, para que estén disponibles desde los nodos que ejecuten el trabajo.
:::
:::

::: {#recursos .section}
## 4. Recursos disponibles

::: {.grid .grid-4}
::: card
::: kpi
8
:::

**particiones configuradas**
:::

::: card
::: kpi
76
:::

**nodos CPU en particiones generales**
:::

::: card
::: kpi
10
:::

**nodos GPU bastet\[01-10\]**
:::

::: card
::: kpi
32
:::

**GPUs disponibles en partición gpus**
:::
:::

### Particiones

::: table-wrap
  Partición                       QoS            Cuenta     Tiempo máx.   Nodos   CPUs   TRES
  ------------------------------- -------------- ---------- ------------- ------- ------ ------------------------------------------------------
  **main**                        normal         general    7-00:00:00    76      1264   cpu=1264,mem=6056057M,node=76,billing=1264
  **gpus**                        normal_gpu     gpu        7-00:00:00    10      216    cpu=216,mem=1673511M,node=10,billing=216,gres/gpu=32
  **priority**                    priority       priority   02:00:00      76      1264   cpu=1264,mem=6056057M,node=76,billing=1264
  **debug**                       debug          debug      1-00:00:00    76      1264   cpu=1264,mem=6056057M,node=76,billing=1264
  **gpus_priority**               priority_gpu   priority   02:00:00      10      216    cpu=216,mem=1673511M,node=10,billing=216,gres/gpu=32
  **long** [default]{.pill .ok}   long           general    30-00:00:00   76      1264   cpu=1264,mem=6056057M,node=76,billing=1264
  **gpus_long**                   long_gpu       gpu        30-00:00:00   10      216    cpu=216,mem=1673511M,node=10,billing=216,gres/gpu=32
  **gpus_debug**                  debug_gpu      gpu        30-00:00:00   10      216    cpu=216,mem=1673511M,node=10,billing=216,gres/gpu=32
:::

### Niveles de Calidad de Servicio --- QoS

::: table-wrap
  QoS                Descripción          Prioridad   Tiempo máximo   Límite GPU
  ------------------ -------------------- ----------- --------------- -------------
  **normal**         Normal QOS default   100         7-00:00:00      gres/gpu=0
  **debug**          debug                1000        06:00:00        gres/gpu=0
  **long**           long                 10          30-00:00:00     gres/gpu=0
  **priority**       priority             2000        1-00:00:00      gres/gpu=0
  **normal_gpu**     normal_gpu           100         7-00:00:00      gres/gpu=32
  **debug_gpu**      debug_gpu            1000        06:00:00        gres/gpu=1
  **long_gpu**       long_gpu             10          30-00:00:00     gres/gpu=16
  **priority_gpu**   priority_gpu         2000        1-00:00:00      gres/gpu=16
:::

::: callout
La partición `long` aparece como partición por defecto en la configuración actual. Para trabajos CPU normales se recomienda especificar explícitamente `--partition=main` cuando se desee usar la cola general de hasta 7 días.
:::
:::

::: {#decision .section}
## 5. Tabla de decisión

Como regla general, usa una partición CPU si no necesitas GPUs y una partición GPU si el programa requiere CUDA, PyTorch/TensorFlow con aceleración, o procesamiento explícito en GPU.

::: table-wrap
  Necesidad                            Partición recomendada          QoS                           Cuenta       Ejemplo relacionado
  ------------------------------------ ------------------------------ ----------------------------- ------------ -------------------------------------
  Trabajo CPU estándar, hasta 7 días   `main`                         `normal`                      `general`    [Script CPU](#cpu)
  Prueba corta o depuración CPU        `debug`                        `debug`                       `debug`      [Script CPU con menos tiempo](#cpu)
  Trabajo CPU largo, hasta 30 días     `long`                         `long`                        `general`    [Plantilla recomendada](#plantilla)
  Trabajo GPU estándar                 `gpus`                         `normal_gpu`                  `gpu`        [Una GPU](#gpu1)
  Prueba corta en GPU                  `gpus_debug`                   `debug_gpu`                   `gpu`        [Una GPU](#gpu1)
  Entrenamiento largo GPU              `gpus_long`                    `long_gpu`                    `gpu`        [Múltiples GPUs](#gpu-multi)
  Trabajo prioritario corto            `priority` o `gpus_priority`   `priority` / `priority_gpu`   `priority`   Solo si tienes autorización
:::
:::

::: {#json-barridos .section}
## 6. Definición de tareas de barrido paramétrico usando fichero JSON

Uno de los problemas habituales al lanzar experimentos basados en barridos paramétricos con un gran número de combinaciones de parámetros es automatizar su lanzamiento, seguimiento y reintento en caso de fallo.

Basados en el gestor antiguo de MESOS, se ha migrado la librería Python `launch_process.py` + `Scheduler.py` para que funcione con el nuevo gestor SLURM. Esta nueva versión asegura compatibilidad con la definición de experimentos antiguos, adaptándolos para poder lanzarlos directamente sin cambios. Existe, no obstante, un cambio de comportamiento en cuanto a la ubicación de ficheros temporales y consultas al sistema.

### Lanzamiento de experimentos

Una vez definido el fichero de configuración JSON, el experimento se lanza mediante la herramienta `launch_process.py`. Esta herramienta genera todas las combinaciones de parámetros, envía las tareas a SLURM y supervisa su ejecución hasta la finalización del framework.

    launch_process.py -c CONFIGURACION.json

El fichero `CONFIGURACION.json` contiene la definición completa del experimento: conexión al cluster, recursos, parámetros variables, parámetros estáticos y comportamiento del framework.

También se pueden pasar parámetros extra desde la línea de comandos para redefinir valores definidos dentro del JSON. Para consultar todas las opciones disponibles:

    launch_process.py --help

::: {.grid .grid-2}
::: card
### Versión estable

Última versión de la librería y ejemplos de ficheros JSON:

<https://www.tsc.uc3m.es/cluster/scripts_latest.zip>
:::

::: card
### Versión de desarrollo

Versiones inestables desde el repositorio de desarrollo:

<https://hermes.tsc.uc3m.es/git/hmolina/cluster_tsc/archive/master.tar.gz>
:::
:::

::: {.callout .warnbox}
**Compatibilidad CPU/GPU:** si no se solicitan GPUs y se configura una partición `queue` con GPUs, los procesos se rechazarán. Si se solicitan GPUs y se configura una `queue` sin GPUs, los procesos quedarán en `PENDING` esperando recursos que esa partición no puede proporcionar.
:::

### Sección `cluster`

Define la configuración de conexión al cluster y las particiones y QoS que se utilizarán para lanzar las tareas.

::: table-wrap
  Variable   Descripción                                                                                                                  Posibles valores
  ---------- ---------------------------------------------------------------------------------------------------------------------------- --------------------------------------------------------------------------------------------------------------------------------------------------------------------
  `master`   Endpoint del servicio `slurmrestd` al que se conectará la librería para enviar trabajos y consultar el estado del cluster.   `https://subserver2.tsc.uc3m.es:6820` o `https://subserver1.tsc.uc3m.es:6820`. Es importante especificar el puerto `6820`, correspondiente al servicio SLURM REST.
  `queue`    Partición SLURM a utilizar. Esta clave sustituye el concepto antiguo de cola del sistema basado en MESOS.                    Cualquiera de las particiones definidas en la sección 4: `main`, `debug`, `long`, `priority`, `gpus`, `gpus_debug`, `gpus_long`, `gpus_priority`.
  `qos`      Nivel de servicio esperado, asociado a la partición seleccionada.                                                            `normal`, `debug`, `long`, `priority`, `normal_gpu`, `debug_gpu`, `long_gpu`, `priority_gpu`.
:::

    "cluster": {
      "master": "https://subserver2.tsc.uc3m.es:6820",
      "queue": "main",
      "qos": "normal"
    }

### Sección `resources`

Define los recursos necesarios en cada nodo para la ejecución de la tarea.

::: table-wrap
  Variable   Descripción                                 Valor recomendado
  ---------- ------------------------------------------- -------------------------------------------------------------------------------------------------------------------------------------------
  `CPUS`     Número de cores por proceso.                `4`
  `MEMORY`   Memoria requerida por proceso, en MBytes.   `4096` MBytes por CPU requerida. Por ejemplo, para 4 CPUs se recomienda empezar con al menos `16384` MBytes si la aplicación lo necesita.
  `GPUS`     Número de GPUs requeridas por proceso.      `0` para trabajos CPU; `1` o más para trabajos GPU.
:::

    "resources": {
      "CPUS": 4,
      "GPUS": 0,
      "MEMORY": 8192
    }

### Sección `tasks`

La sección `tasks` es un diccionario que define el nombre base de las tareas, el comando a ejecutar, los parámetros variables del barrido y los parámetros estáticos comunes a todas las ejecuciones.

::: table-wrap
  Clave              Descripción
  ------------------ ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  `task_basename`    Nombre con el que se identificará cada tarea lanzada a los nodos remotos. El nombre se construirá a partir de `task_basename` más el identificador de la tarea y el número total de tareas.
  `cmdbase`          Sintaxis del comando que se debe ejecutar. Es un string basado en Python cuyos valores entre llaves se sustituyen por valores individuales de `var_parameters` o por variables estáticas definidas en `parameters`.
  `var_parameters`   Parámetros variables para cada comando a ejecutar. Se realizará una combinación entre todas las variables aquí definidas para generar todas las tareas. Los valores pueden ser listas de valores numéricos, cadenas de texto o expresiones Python que generen listas de datos de forma automática, por ejemplo `range(inicio, final, paso)`, `np.arange(...)`, `os.getcwd()` u otras expresiones que permitan construir secuencias de valores de forma programática.
  `parameters`       Parámetros estáticos de la tarea. Estos valores son comunes a todos los comandos generados.
:::

    "tasks": {
      "task_basename": "DeepAutoencoder",
      "cmdbase": "{baseprogram} --lr {learning_rate} --epochs {numepochs} --batch-size {batchsize}",
      "var_parameters": {
        "learning_rate": [
          0.00000000100,
          0.00000000101,
          0.00000000102,
          0.00000000103,
          0.00000000104,
          0.00000000105
        ]
      },
      "parameters": {
        "baseprogram": "/usr/bin/python3 mnist_pytorch.py",
        "database": "MNIST",
        "numepochs": 10,
        "batchsize": 250,
        "localdir": "/export/clusterdata/USER/data_test",
        "shared_dir": "/export/clusterdata/USER/autoencoder/results"
      }
    }

Con estos parámetros, se generarán automáticamente los comandos a ejecutar sustituyendo las variables definidas entre llaves (`{}`) por los valores establecidos en los campos `var_parameters` y `parameters`. Por ejemplo, los dos primeros comandos generados serían:

    /usr/bin/python3 mnist_pytorch.py --lr 0.00000000100 --epochs 10 --batch-size 250
    /usr/bin/python3 mnist_pytorch.py --lr 0.00000000101 --epochs 10 --batch-size 250

Y así sucesivamente hasta completar todos los valores definidos en la lista `learning_rate`. Cuando existan varias variables dentro de `var_parameters`, el sistema generará automáticamente todas las combinaciones posibles entre ellas, creando una tarea independiente para cada combinación de parámetros.

::: callout
**Cálculo del número de tareas:** el número total de tareas generadas será el producto del número de valores definidos para cada una de las variables incluidas en `var_parameters`.
:::

    learning_rate : 6 valores
    batchsize     : 4 valores
    optimizer     : 3 valores

    Total tareas = 6 × 4 × 3 = 72

### Clave `framework`

La clave `framework` es un diccionario que define los comandos y el entorno de ejecución de cada uno de los procesos remotos lanzados en los nodos esclavos.

::: table-wrap
  Clave                         Descripción
  ----------------------------- ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
  `name`                        Nombre del framework.
  `max_num_procs_simultaneos`   Máximo número de procesos simultáneos. Use `-1` para no limitar el número de procesos.
  `max_failed_tasks`            Máximo número de tareas que pueden fallar antes de declarar el framework fallido. `0`: ante el primer fallo se para la ejecución completa. `n>0`: pueden fallar hasta `n` tareas antes de parar el framework. `-1`: las tareas fallidas se reasignan y el framework no se para por fallos individuales.
  `environment_vars`            Diccionario con variables de entorno que se definirán en cada proceso remoto. Para procesos Matlab es necesario exportar `HOME`.
  `uris`                        Vector con el conjunto de ficheros que se distribuirán a cada proceso remoto.
  `stdout`                      Fichero donde se almacenará la salida de consola del proceso maestro `launch_process.py`.
  `stderr`                      Fichero donde se almacenará la salida de error del proceso maestro `launch_process.py`.
  `notification`                `1` si se desea enviar un correo al finalizar las tareas.
  `email`                       Dirección de correo de destino cuando `notification` vale `1`. El correo se enviará desde `cluster@tsc.uc3m.es`.
  `max_num_task_per_host`       Máximo número de tareas simultáneas por equipo. Es útil cuando cada proceso necesita un puerto específico y no puede haber más de cierto número de procesos por máquina.
  `workdir`                     Directorio donde se ejecutarán los procesos. Allí encontrará scripts y librerías de ejecución y, salvo que se defina otra variable, salvará los ficheros de log de cada proceso. Si no se define, se asumirá como `workdir` el directorio actual desde donde se lanza `launch_process.py`.
  `logdir`                      Directorio donde se almacenarán los logs de cada tarea. Si no se especifica, será el mismo `workdir`.
:::

    "framework": {
      "name": "Ejemplo Matlab",
      "max_num_procs_simultaneos": -1,
      "max_failed_tasks": 0,
      "environment_vars": {
        "MATLABPATH": "/export/usuarios01/USUARIO/matlab",
        "HOME": "/export/usuarios01/USUARIO"
      },
      "stdout": "stdout.log",
      "stderr": "stderr.log",
      "notification": 1,
      "email": "USUARIO@tsc.uc3m.es",
      "max_num_task_per_host": 3,
      "workdir": "/export/clusterdata/USUARIO/experimentos/matlab",
      "logdir": "/export/clusterdata/USUARIO/experimentos/matlab/logs"
    }

Este ejemplo configura un framework de nombre **Ejemplo Matlab**. El máximo número de procesos simultáneos se define con `max_num_procs_simultaneos=-1`, por lo que no se limita cuántos procesos se lanzan en la granja.

El máximo número de tareas fallidas se fija con `max_failed_tasks=0`, indicando que ante la primera tarea fallida el framework debe detenerse. El diccionario `environment_vars` exporta las variables `MATLABPATH` y `HOME`; en experimentos con Matlab es necesario definir `HOME`.

Las variables `stdout` y `stderr`, si están definidas, indican los ficheros a los que se redireccionará la salida estándar y la salida de error del programa `launch_process.py`. Si `notification` es `1`, se enviará un correo a la dirección indicada en `email` desde la cuenta `cluster@tsc.uc3m.es`. La variable `max_num_task_per_host` limita cuántas tareas simultáneas pueden ejecutarse en un mismo nodo; en el ejemplo, como máximo 3 tareas por equipo.

### Ejemplos completos de ficheros JSON

Los siguientes ejemplos muestran configuraciones completas para el lanzador. El primer ejemplo usa una configuración ya adaptada a SLURM con `queue=main`. El segundo ejemplo corresponde a una definición antigua basada en MESOS, conservada como ejemplo de compatibilidad y como referencia para migración a la nueva estructura SLURM.

#### Ejemplo Matlab

    {
        "cluster": {
            "master"            : "https://subserver2.tsc.uc3m.es:6820",
        "queue"           : "main"
        },
        "resources": {
            "CPUS": 4,
            "GPUS": 0,
            "MEMORY": 16384
        },
        "tasks": {
            "task_basename"     : "Ejemplo_Matlab",
            "cmdbase"           : "{working_dir}/{baseprogram} {funcion_matlab} {num_params} {var_param1} {param2} {var_param3} ",
            "var_parameters"    : {
                "var_param1"    : [3,6,8],
                "var_param3"    : "np.arange(10)",
            "working_dir"   : "os.getcwd()"
            },
            "parameters"        : {
                "baseprogram"       : "MatlabFunctions",
                "funcion_matlab"    : "cpu_eater",
                "num_params"        : 3,
                "param2"            : 100
            }
        },

        "framework" : {
            "name"                      : "Ejemplo Matlab",
            "max_num_procs_simultaneos" : -1,
            "max_failed_tasks"          : 0,
            "environment_vars"      : {
                "MATLABPATH"        : "/export/usuarios01/USUARIO/matlab",
                "HOME"              : "/export/usuarios01/USUARIO"
            },
            "stdouta"                : "stdout.log",
            "stderra"                : "stderr.log",
            "notification"          : 1,       
        "email"                 : "h.molina@tsc.uc3m.es"
        }




        
    }

#### Ejemplo PyTorch MNIST / DeepAutoencoder

    {
        "cluster": {
            "master"            : "zk://10.0.12.18:2181,10.0.12.77:2181,10.0.12.60:2181,10.0.12.51:2181,10.0.12.75:2181,10.0.12.76:2181,10.0.12.78:2181/mesos_gpu"
        },
        "resources": {
            "CPUS": 4,
            "MEMORY": 8192
        },
        "tasks": {
            "task_basename"     : "DeepAutoencoder",
            "cmdbase"           : "{working_dir}/{baseprogram} --lr {learning_rate} --epochs {numepochs} --batch-size {batchsize}",
            "var_parameters"    : {
                "working_dir"        : "os.getcwd()",
                "learning_rate"        : [
                    0.00000000100,
                    0.00000000101,
                    0.00000000102,
                    0.00000000103,
                    0.00000000104,
                    0.00000000105,
                    0.00000000106,
                    0.00000000107,
                    0.00000000108,
                    0.00000000109,
                    0.00000000101,
                    0.00000000102,
                    0.00000000103,
                    0.00000000104,
                    0.00000000105,
                    0.00000000106,
                    0.00000000107,
                    0.00000000108,
                    0.00000000109,
                    0.00000000110,
                    0.00000000111,
                    0.00000000112,
                    0.00000000113,
                    0.00000000114,
                    0.00000000115,
                    0.00000000117,
                    0.00000000118,
                    0.00000000119,
                    0.00000000120,
                    0.00000000121,
                    0.00000000122,
                    0.00000000123,
                    0.00000000124,
                    0.00000000125,
                    0.00000000126,
                    0.00000000127,
                    0.00000000128,
                    0.00000000129,
                    0.00000000130,
                    0.00000000131,
                    0.00000000132,
                    0.00000000133,
                    0.00000000134,
                    0.00000000135,
                    0.00000000136,
                    0.00000000137,
                    0.00000000138,
                    0.00000000139,
                    0.00000000140,
                    0.00000000141,
                    0.00000000142,
                    0.00000000143,
                    0.00000000144,
                    0.00000000145,
                    0.00000000146,
                    0.00000000147,
            0.00000000148,
            0.00000001001,
                    0.00000001002,
                    0.00000001003,
                    0.00000001004,
                    0.00000001005,
                    0.00000001006,
                    0.00000001007,
                    0.00000001008,
                    0.00000001009,
                    0.00000001001,
                    0.00000001002,
                    0.00000001003,
                    0.00000001004,
                    0.00000001005,
                    0.00000001006,
                    0.00000001007,
                    0.00000001008,
                    0.00000001009,
                    0.00000001010,
                    0.00000001011,
                    0.00000001012,
                    0.00000001013,
                    0.00000001014,
                    0.00000001015,
                    0.00000001017,
                    0.00000001018,
                    0.00000001019,
                    0.00000001020,
                    0.00000001021,
                    0.00000001022,
                    0.00000001023,
                    0.00000001024,
                    0.00000001025,
                    0.00000001026,
                    0.00000001027,
                    0.00000001028,
                    0.00000001029,
                    0.00000001030,
                    0.00000001031,
                    0.00000001032,
                    0.00000001033,
                    0.00000001034,
                    0.00000001035,
                    0.00000001036,
                    0.00000001037,
                    0.00000001038,
                    0.00000001039,
                    0.00000001040,
                    0.00000001041,
                    0.00000001042,
                    0.00000001043,
                    0.00000001044,
                    0.00000001045,
                    0.00000001046,
                    0.00000001047,
                    0.00000001048,
            0.00000001101,
                    0.00000001102,
                    0.00000001103,
                    0.00000001104,
                    0.00000001105,
                    0.00000001106,
                    0.00000001107,
                    0.00000001108,
                    0.00000001109,
                    0.00000001101,
                    0.00000001102,
                    0.00000001103,
                    0.00000001104,
                    0.00000001105,
                    0.00000001106,
                    0.00000001107,
                    0.00000001108,
                    0.00000001109,
                    0.00000001110,
                    0.00000001111,
                    0.00000001112,
                    0.00000001113,
                    0.00000001114,
                    0.00000001115,
                    0.00000001117,
                    0.00000001118,
                    0.00000001119,
                    0.00000001120,
                    0.00000001121,
                    0.00000001122,
                    0.00000001123,
                    0.00000001124,
                    0.00000001125,
                    0.00000001126,
                    0.00000001127,
                    0.00000001128,
                    0.00000001129,
                    0.00000001130,
                    0.00000001131,
                    0.00000001132,
                    0.00000001133,
                    0.00000001134,
                    0.00000001135,
                    0.00000001136,
                    0.00000001137,
                    0.00000001138,
                    0.00000001139,
                    0.00000001140,
                    0.00000001141,
                    0.00000001142,
                    0.00000001143,
                    0.00000001144,
                    0.00000001145,
                    0.00000001146,
                    0.00000001147,
                    0.00000001148
                ]
            },
            "parameters"        : {
                "baseprogram"       : "mnist_pytorch.py",
                "database"          : "MNIST",
                "numepochs"         : 10,
                "batchsize"         : 250
            }
        },

        "framework" : {
            "name"                      : "Prueba Nuevo Lanzador",
            "max_num_procs_simultaneos" : -1,
            "max_failed_tasks"          : 0,
            "environment_vars"      : {
                "PATH"              : "/usr/local/bin:/usr/bin:/bin"
            },
            "stdout"                : "stdout.log",
            "stderr"                : "stderr.log",
            "notification_at_end"   : 1          
        }




        
    }

::: {.callout .warnbox}
**Nota de migración:** en ficheros antiguos puede aparecer un `master` de tipo `zk://...` usado por MESOS. Aunque se recomienda actualizar esta variable al nuevo formato basado en SLURM REST, por ejemplo `https://subserver2.tsc.uc3m.es:6820`, la herramienta detecta automáticamente si se trata de una configuración antigua y redirige la ejecución al servidor principal de SLURM definido por omisión. De esta forma, los experimentos antiguos pueden seguir lanzándose sin modificar el fichero JSON, aunque se recomienda migrarlos progresivamente al nuevo formato.
:::
:::

::: {#ejemplos .section}
## 7. Ejemplos de uso

### Lanzar un trabajo en CPU {#cpu}

    #!/bin/bash
    #SBATCH --job-name=cpu_ejemplo
    #SBATCH --partition=main
    #SBATCH --account=general
    #SBATCH --qos=normal
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=4
    #SBATCH --mem=8G
    #SBATCH --time=02:00:00
    #SBATCH --output=/export/clusterdata/%u/logs/%x-%j.out
    #SBATCH --error=/export/clusterdata/%u/logs/%x-%j.err

    set -euo pipefail

    cd /export/clusterdata/$USER/proyectos/mi_proyecto
    source /export/clusterdata/$USER/venvs/mi_entorno/bin/activate

    python mi_programa.py

### Lanzar un trabajo en GPU con una sola GPU {#gpu1}

    #!/bin/bash
    #SBATCH --job-name=gpu_1
    #SBATCH --partition=gpus
    #SBATCH --account=gpu
    #SBATCH --qos=normal_gpu
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=8
    #SBATCH --mem=32G
    #SBATCH --gres=gpu:1
    #SBATCH --time=08:00:00
    #SBATCH --output=/export/clusterdata/%u/logs/%x-%j.out
    #SBATCH --error=/export/clusterdata/%u/logs/%x-%j.err

    set -euo pipefail

    cd /export/clusterdata/$USER/proyectos/entrenamiento
    source /export/clusterdata/$USER/venvs/torch/bin/activate

    nvidia-smi
    python train.py --epochs 50 --batch-size 128

### Lanzar un trabajo en GPU con múltiples GPUs, por ejemplo PyTorch {#gpu-multi}

    #!/bin/bash
    #SBATCH --job-name=pytorch_multi_gpu
    #SBATCH --partition=gpus
    #SBATCH --account=gpu
    #SBATCH --qos=normal_gpu
    #SBATCH --nodes=1
    #SBATCH --ntasks-per-node=4
    #SBATCH --cpus-per-task=8
    #SBATCH --mem=96G
    #SBATCH --gres=gpu:4
    #SBATCH --time=12:00:00
    #SBATCH --output=/export/clusterdata/%u/logs/%x-%j.out
    #SBATCH --error=/export/clusterdata/%u/logs/%x-%j.err

    set -euo pipefail

    cd /export/clusterdata/$USER/proyectos/pytorch
    source /export/clusterdata/$USER/venvs/torch/bin/activate

    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

    torchrun --standalone \
      --nnodes=1 \
      --nproc_per_node=$SLURM_NTASKS_PER_NODE \
      train.py --config configs/experimento.yaml

### Pasar parámetros a un script

    # Enviar parámetros al script:
    sbatch ejemplo_parametros.slurm datos.csv resultado.csv 100

    # ejemplo_parametros.slurm
    #!/bin/bash
    #SBATCH --job-name=parametros
    #SBATCH --partition=main
    #SBATCH --account=general
    #SBATCH --qos=normal
    #SBATCH --time=01:00:00
    #SBATCH --cpus-per-task=2
    #SBATCH --mem=4G

    INPUT="$1"
    OUTPUT="$2"
    ITERACIONES="$3"

    python procesa.py --input "$INPUT" --output "$OUTPUT" --iteraciones "$ITERACIONES"
:::

::: {#comandos .section}
## 8. Comandos básicos de SLURM

### 1) Lanzar scripts

    # Lanzar un script
    sbatch trabajo.slurm

    # Lanzar a una partición concreta
    sbatch --partition=main --account=general --qos=normal trabajo.slurm

    # Lanzar a una partición GPU
    sbatch --partition=gpus --account=gpu --qos=normal_gpu --gres=gpu:1 trabajo.slurm

    # Solicitar un modelo de GPU concreto, si está definido en GRES
    sbatch --partition=gpus --account=gpu --qos=normal_gpu --gres=gpu:MODELO:1 trabajo.slurm

### 2) Consultar estados de trabajo

    # Ver la cola del usuario actual
    squeue -u $USER

    # Ver un trabajo concreto y el motivo de espera
    squeue -j JOBID -o "%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R"

    # Información detallada
    scontrol show job JOBID

::: table-wrap
  Estado                    Significado
  ------------------------- ---------------------------------------------------------------------------------------------
  `PD` / `PENDING`          El trabajo está en cola esperando recursos, prioridad, permisos o condiciones de partición.
  `R` / `RUNNING`           El trabajo está ejecutándose.
  `CG` / `COMPLETING`       El trabajo está finalizando y limpiando recursos.
  `CD` / `COMPLETED`        El trabajo terminó correctamente.
  `F` / `FAILED`            El trabajo falló.
  `CA` / `CANCELLED`        El trabajo fue cancelado.
  `TO` / `TIMEOUT`          El trabajo superó el tiempo máximo asignado.
  `OOM` / `OUT_OF_MEMORY`   El trabajo superó la memoria solicitada.
:::

### 3) Cancelar trabajos

    scancel JOBID
    scancel -u $USER
    scancel --name nombre_del_trabajo

### 4) Información del cluster

    sinfo
    sinfo -Nel
    scontrol show partition
    scontrol show node NOMBRE_NODO

### 5) Historial y contabilidad

    # Trabajos recientes del usuario
    sacct -u $USER --format=JobID,JobName,Partition,Account,QOS,State,Elapsed,MaxRSS,ExitCode

    # Eficiencia de un trabajo terminado
    seff JOBID

### 6) Trabajos interactivos

    # Sesión interactiva CPU
    srun --partition=debug --account=debug --qos=debug \
         --cpus-per-task=2 --mem=4G --time=01:00:00 --pty bash

    # Sesión interactiva GPU
    srun --partition=gpus_debug --account=gpu --qos=debug_gpu \
         --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=01:00:00 --pty bash
:::

::: {#problemas .section}
## 9. Resolución de problemas

### Trabajos en PENDING

Si un trabajo queda en `PENDING`, consulta el motivo exacto con la columna `%R`:

    squeue -j JOBID -o "%.18i %.9P %.30j %.8u %.2t %.10M %.6D %R"
    scontrol show job JOBID

::: table-wrap
  Motivo frecuente                  Qué revisar
  --------------------------------- ----------------------------------------------------------------------------------
  `Resources`                       No hay suficientes CPUs, memoria, nodos o GPUs libres. Reduce recursos o espera.
  `Priority`                        Hay trabajos con mayor prioridad por delante.
  `PartitionTimeLimit`              El tiempo solicitado supera el máximo de la partición/QoS.
  `AssocGrpGRES` o límites de GPU   Se han alcanzado límites de GPUs por QoS, cuenta o usuario.
  `InvalidAccount` / `QOS`          La cuenta o QoS indicada no es válida para esa partición.
:::

### Errores de salida comunes

SLURM escribe la salida estándar y los errores en los ficheros configurados con `--output` y `--error`. Si no se indican, normalmente se genera un fichero `slurm-JOBID.out` en el directorio desde el que se lanzó el trabajo.

    tail -f /export/clusterdata/$USER/logs/mi_trabajo-JOBID.out
    tail -f /export/clusterdata/$USER/logs/mi_trabajo-JOBID.err

::: {.callout .dangerbox}
**Errores típicos:** entorno Python no activado, librerías no instaladas en el nodo de ejecución, rutas relativas incorrectas, falta de permisos, memoria insuficiente, límite de tiempo agotado o solicitud incorrecta de GPUs.
:::
:::

::: {#web .section}
## 10. Interfaces web

::: links
[**SLURM Web**\
Interfaz para ver el estado de las colas de trabajo, procesos en ejecución, recursos usados, recursos disponibles y nodos fuera de línea.](https://subserver3.tsc.uc3m.es/slurm-web) [**Statistics**\
Dashboard con información básica del estado del cluster.](https://subserver3.tsc.uc3m.es/statistics) [**Grafana**\
Dashboards avanzados con información detallada del estado de la granja.](https://subserver3.tsc.uc3m.es/grafana)
:::
:::

::: {#buenas .section}
## 11. Buenas prácticas

::: {.grid .grid-2}
::: card
### Recomendaciones

-   Siempre prueba con `--qos=debug` antes de lanzar trabajos largos.
-   Usa nombres descriptivos para tus trabajos con `--job-name`.
-   Especifica correctamente la memoria y CPUs que necesitas.
-   Utiliza rutas absolutas en tus scripts.
-   Guarda logs en directorios organizados.
-   Implementa checkpoints en entrenamientos largos.
-   Monitoriza el uso de recursos con `seff` después de ejecutar.
-   Usa entornos virtuales en discos compartidos.
-   Documenta tus scripts con comentarios claros.
:::

::: card
### Evita hacer

-   Conectarte directamente por SSH a nodos de cómputo.
-   Ejecutar trabajos pesados en el nodo principal.
-   Solicitar más recursos de los que realmente necesitas.
-   Dejar trabajos en cola indefinidamente si ya no los necesitas.
-   Usar entornos virtuales, datos o resultados en rutas locales no compartidas. Usa `/export/usuarios01/USER` o `/export/clusterdata/USER`.
-   No especificar límites de tiempo.
-   Ignorar los mensajes de error en logs.
-   Lanzar múltiples trabajos idénticos sin control.
-   No limpiar archivos temporales después de terminar.
:::
:::
:::

::: {#plantilla .section}
## 12. Plantilla de script recomendada

    #!/bin/bash
    #SBATCH --job-name=nombre_descriptivo
    #SBATCH --partition=main
    #SBATCH --account=general
    #SBATCH --qos=normal
    #SBATCH --nodes=1
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=4
    #SBATCH --mem=8G
    #SBATCH --time=02:00:00
    #SBATCH --output=/export/clusterdata/%u/logs/%x-%j.out
    #SBATCH --error=/export/clusterdata/%u/logs/%x-%j.err

    set -euo pipefail

    echo "Trabajo: $SLURM_JOB_NAME"
    echo "Job ID: $SLURM_JOB_ID"
    echo "Nodo(s): $SLURM_JOB_NODELIST"
    echo "Directorio inicial: $(pwd)"
    echo "Fecha inicio: $(date)"

    mkdir -p /export/clusterdata/$USER/logs

    cd /export/clusterdata/$USER/proyectos/mi_proyecto

    # Activar entorno, si procede
    source /export/clusterdata/$USER/venvs/mi_entorno/bin/activate

    # Ejecutar aplicación
    python programa.py --input datos/input.dat --output resultados/output.dat

    echo "Fecha fin: $(date)"

::: callout
Para trabajos GPU, cambia `--partition=main` por `--partition=gpus`, `--account=general` por `--account=gpu`, `--qos=normal` por `--qos=normal_gpu`, y añade `--gres=gpu:1` o el número de GPUs necesario.
:::
:::
:::
:::

Manual generado para el Departamento de Teoría de la Señal y Comunicaciones · SLURM / HPC · Versión completa con sección JSON
