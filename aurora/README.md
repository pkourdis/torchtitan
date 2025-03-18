# BKMs for running models in `torchtitan` on the Aurora supercomputer
To run:
```
git checkout https://github.com/pkourdis/torchtitan.git
cd torchtitan
git checkout aurora
cd aurora
qsub -l nodes=<NUM_NODES>:ncpus=208 -l walltime=<DURATION> -l filesystems=home -q <QUEUE> -A <ACCOUNT_NAME> ./run_train.sh
```
where inside `run_train.sh` the specific model can be selected:
```
#LLAMA_CONFIG="debug_model"
#LLAMA_CONFIG="llama3_8b"
LLAMA_CONFIG="llama3_70b"
#LLAMA_CONFIG="llama3_405b"
```
The train configurations for each model can be found in `torchtitan/aurora/train_configs`:
```
aurora/train_configs/
└── llama
    ├── debug_model.toml
    ├── llama3_405b.toml
    ├── llama3_70b.toml
    └── llama3_8b.toml
```
Traing logs can be found in ``torchtitan/aurora/output/logs`
```
aurora/outputs/logs
├── llama3_70b
│   └── 20250318
│       ├── llama3_70b_aurora_128n12p_20250318_113737_train_config.toml
│       ├── llama3_70b_aurora_128n12p_20250318_113737_train_log.txt
│       ├── llama3_70b_aurora_16n12p_20250318_132055_train_config.toml
│       ├── llama3_70b_aurora_16n12p_20250318_132055_train_log.txt
│       ├── llama3_70b_aurora_32n12p_20250318_113709_train_config.toml
│       ├── llama3_70b_aurora_32n12p_20250318_113709_train_log.txt
│       ├── llama3_70b_aurora_64N12P_20250318_113351_train_config.toml
│       ├── llama3_70b_aurora_64N12P_20250318_113351_train_log.txt
│       ├── llama3_70b_aurora_64n12p_20250318_123231_train_config.toml
│       └── llama3_70b_aurora_64n12p_20250318_123231_train_log.txt
└── llama3_8b
    └── 20250318
        ├── llama3_8b_aurora_128n12p_20250318_123754_train_config.toml
        ├── llama3_8b_aurora_128n12p_20250318_123754_train_log.txt
        ├── llama3_8b_aurora_32n12p_20250318_123754_train_config.toml
        ├── llama3_8b_aurora_32n12p_20250318_123754_train_log.txt
        ├── llama3_8b_aurora_64n12p_20250318_123638_train_config.toml
```