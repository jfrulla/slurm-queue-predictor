# slurm-queue-predictor

This is a work in progress. It will:
* Extract all job data from slurm and store it as a parquet file
* Preprocess the slurm job data for training
* Train a model to predict slurm queue times
* Make predictions using the model and newly submitted jobs

## Usage (so far):

```
git clone git@github.com:jfrulla/slurm-queue-predictor.git
cd slurm-queue-predictor

# Edit the config file at this point

make install
source .venv/bin/activate

# Extract data from slurm
python slurm_data_extractor.py
```