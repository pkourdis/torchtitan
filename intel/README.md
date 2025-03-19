# BKMs for running models with `torchtitan` on the Aurora & Borealis systems
To run the models available in `torchtitan`:
```
git clone  https://github.com/pkourdis/torchtitan.git
cd torchtitan
```
Download a tokenizer following the instructions here https://github.com/pkourdis/torchtitan/tree/aurora?tab=readme-ov-file#downloading-a-tokenizer. Then:
```
git checkout intel
cd intel
```
You can submit using a job via `PBS` to run a model as follows:
```
qsub SYS=<SYS>,PT_CONFIG=<PT_CONFIG>,LLAMA_CONFIG=<LLAMMA_CONFIG>,PPN=<PPN> -l nodes=<NUM_NODES>:ncpus=208 -l walltime=<DURATION> -l filesystems=home -q <QUEUE_NAME> -A <ACCOUNT_NAME> ./run_train.sh
```
where:
* `<SYSTEM>`: specifies the system and can be one of `aurora` or `borealis` (default: `aurora`).
* `<PT_CONFIG>`: use `pt` only for PyTorch and `pt+ipex` for PyTorch with IPEX (default: `pt+ipex`).
* `<LLAMA_CONFIG>`: one of `debug_model`, `llama3_8b`, `llama3_70b` and `llama3_405b` (default: `llama3_8b`).
* `<PPN>`is the processes per node (default: `PPN=12`).

The train configurations for the `Llama` models can be found at `torchtitan/aurora/train_configs`:
```
intel/train_configs/
└── llama
    ├── debug_model.toml
    ├── llama3_405b.toml
    ├── llama3_70b.toml
    └── llama3_8b.toml
```
Each job generates a train log, saves the model configuration (i.e. the `*.toml` file) and the environment loaded on Aurora to run the job. The logs can be found at `torchtitan/aurora/output/logs/<SYSTEM>`.