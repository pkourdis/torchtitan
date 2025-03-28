# BKMs for running models with `torchtitan` on the Aurora & Borealis systems

## Get the repo
To run the models available in `torchtitan`:
```
$ git clone https://github.com/pkourdis/torchtitan.git
$ cd torchtitan
$ git checkout intel
```
Download a tokenizer following the instructions here https://github.com/pkourdis/torchtitan/tree/aurora?tab=readme-ov-file#downloading-a-tokenizer.

## Running
From inside `torchtitan/intel` directory.
### Batch job
To run a model:
```
$ qsub -v <COMMA SEPARATED OPTIONS> -l nodes=<NUM_NODES>:ncpus=208 -l walltime=<DURATION> -l filesystems=home -q <QUEUE_NAME> -A <ACCOUNT_NAME> ./run_train.sh
```
### Interactive job
Start an interactive session:
```
$ qsub -l nodes=<NUM_NODES>:ncpus=208 -l walltime=<DURATION> -l filesystems=home -q <QUEUE_NAME> -A <ACCOUNT_NAME> -I
```
and to run the model:
```
<SPACE SEPARATED OPTIONS> ./run_train.sh
```
where `<COMMA SEPARATED OPTIONS>` or `<SPACE SEPARATED OPTIONS>` are:
* `<SYSTEM>`: specifies the system and can be one of `aurora` or `borealis` (default: `aurora`).
* `<PT_CONFIG>`: use `pt` only for PyTorch and `pt+ipex` for PyTorch with IPEX (default: `pt+ipex`).
* `<LLAMA_CONFIG>`: one of `debug_model`, `llama3_8b`, `llama3_70b` and `llama3_405b` or the name of train config TOML file in `train_configs` directory (default: `llama3_8b`).
* `<PPN>`: the processes per node (default: `PPN=12`).
* `<ENV>`: the environment to load from `envs` directory (default: `latest`).

For example, to pre-train the `llama3_405b` model on 32 nodes with 8 processes per node for 2hrs using a batch job:
```
qsub -v LLAMA_CONFIG="llama3_405b",PPN=8 -l nodes=32:ncpus=208 -l walltime=120:00 -l filesystems=home -q prod -A Intel-Aurora ./run_train.sh
```
The train configurations for the `Llama` models can be found at `torchtitan/intel/train_configs`:
```
torchtitan/intel/train_configs/
└── llama
    ├── debug_model.toml
    ├── llama3_405b.toml
    ├── llama3_70b.toml
    └── llama3_8b.toml
```
Each job generates a train log, saves the model configuration (i.e. the `*.toml` file) and the environment loaded on system whre the job run. The logs can be found at `torchtitan/aurora/output/logs/<SYSTEM>`:
```
torchtitan/intel/outputs/logs/
└── aurora
    ├── llama3_70b
    │   └── 2025-03-19
    │       ├── llama3_70b_aurora_16n12ppn_pt_3382131pbs_2025-03-19_10:05_train_config.toml
    │       ├── llama3_70b_aurora_16n12ppn_pt_3382131pbs_2025-03-19_10:05_train_env.sh
    │       ├── llama3_70b_aurora_16n12ppn_pt_3382131pbs_2025-03-19_10:05_train_log.txt
    │       ├── llama3_70b_aurora_16n12ppn_pt+ipex_3382122pbs_2025-03-19_09:03_train_config.toml
    │       ├── llama3_70b_aurora_16n12ppn_pt+ipex_3382122pbs_2025-03-19_09:03_train_env.sh
    │       ├── llama3_70b_aurora_16n12ppn_pt+ipex_3382122pbs_2025-03-19_09:03_train_log.txt
    │       ├── llama3_70b_aurora_32n12ppn_pt+ipex_3382147pbs_2025-03-19_09:22_train_config.toml
    │       ├── llama3_70b_aurora_32n12ppn_pt+ipex_3382147pbs_2025-03-19_09:22_train_env.sh
    │       └── llama3_70b_aurora_32n12ppn_pt+ipex_3382147pbs_2025-03-19_09:22_train_log.txt
    └── llama3_8b
        └── 2025-03-19
            ├── llama3_8b_aurora_2n12ppn_pt_3382882pbs_2025-03-19_16:39_train_config.toml
            ├── llama3_8b_aurora_2n12ppn_pt_3382882pbs_2025-03-19_16:39_train_env.sh
            ├── llama3_8b_aurora_2n12ppn_pt_3382882pbs_2025-03-19_16:39_train_log.txt
            ├── llama3_8b_aurora_2n12ppn_pt+ipex_3382098pbs_2025-03-19_08:50_train_config.toml
            ├── llama3_8b_aurora_2n12ppn_pt+ipex_3382098pbs_2025-03-19_08:50_train_env.sh
            └── llama3_8b_aurora_2n12ppn_pt+ipex_3382098pbs_2025-03-19_08:50_train_log.txt
```
