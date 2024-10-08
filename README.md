# Thesis
## EHR Representation Learning: Diagnosis of Diseases

This is the repository corresponding to the Masters Thesis on EHR Foundation Models.


## Required Packages
Create conda environment
python > 3.9 is needed to install femr
conda create -n ENV_NAME python=3.10
conda activate ENV_NAME
pip install femr


Parts of the code are inspired by https://github.com/sungresearch/femr-on-mimic

The requirements.txt file contains all required packages to run the provided code. 

## MIMIC Demo setup
You can verify that this code is working by using the MIMIC demo data.

### Download Data
Please replace YOUR_DIRECTORY with the corresponding directory. Note that you only have access to the demo data this way. To access the complete dataset you have to apply for access on Physionet.
```bash
wget -r -N -c --no-host-directories --cut-dirs=1 -np -P YOUR_DIRECTORY https://physionet.org/files/mimic-iv-demo/2.2/
```

### Transform Data to MEDS format
```bash
meds_etl_mimic YOUR_DATA_DIR/mimic-iv-demo/ TARGET_DIR
```

### Adjust Configuration
In the config directories, config files are located that determine the execution flow of the scripts. You only have to adjust the parameters meds_path, output_path, and vocab_path.

meds_path is the path that was used for the placeholder TARGET_DIR in the previous command.
output_path can be any directory to which the outputs of intermediate steps are written.

In src.models.clmbr_t_base.py you have to set the HF API Token to use the model

### Run on SLURM Cluster
Load the Python module
```bash
module load devel/python/3.12.3_intel_2023.1.0
```


Copy thesis files
```
rsync -avz --progress /home/niclas/Dokumente/thesis cluster:/home/USER_DIR
```

Load virtualenv with installed dependencies
```bash
source ~/thesis_env/bin/activate
```

Run the demo on the cluster
```bash
python scripts/run_pipeline.py --config-path ~/path_to_this_repo/config/demo_cluster.yaml
```

### Run Local
The file scripts/run_pipeline.py can be used to execute the whole pipeline. It takes the path to a config file as input if no SLURM config is defined in the file, it executed the pipeline locally. The config file decides which steps should be executed. Furthermore, the individual scripts for the pipeline are placed in the scripts directoy. This allows to individually execute a pipeline step without needing to run the complete pipeline or understand the run_pipeline.py file.

Run the demo pipeline
```bash
python scripts/run_pipeline.py --config-path ~/path_to_this_repo/config/demo_conf.yaml
```

### Notebooks
We did not include the outputs of the notebooks as this would risk to expose parts of the MIMIC-IV database to the public. Therefore, all output cells were cleared before uploading.