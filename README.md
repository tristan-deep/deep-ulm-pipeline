## A tracking and visualization pipeline for ULM

> **New:**
Added deepULM localization model! See this [script](./model.py).

Performs the following steps:
- Load localization data from hdf5 file
- Load config file, see example in [tracking_config.yaml](./configs/tracking_config.yaml)
- Track localizations using track parameters from config file
- Save raw tracks to file
- Postprocess tracks
- Convert tracks to maps with [tracks_to_map.py](./tracks_to_map.py)
- Visualize and save maps using [visualization.py](./visualization.py)

Example usage:

```bash
python main.py -c ./configs/tracking_config.yaml
```

```bash
Args:
    -c (--config): path to tracking config file.
    -s (--save): Save raw tracks to file
    --skip: Skip folders that already have tracks.

```

> **Note:**
Make sure the localizations are stored in the correct format (hdf5) and have only a single localization file per folder. You can refer to the `read.py` file and the `ReadLocs` class for more information on the correct format.

### Save localizations
An example of how to save localizations to hdf5 format is shown in [save_localizations.py](./save_localizations.py). This script saves some dummy localizations to a file in the correct format. You should be able to run [main.py](./main.py) with the output of this script.

*Author: Tristan Stevens, 2024, Eindhoven University of Technology*


### Setup environment
Create a conda environment and pip install requirements.

```bash
conda create -n ulm python=3.6 -y
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y
conda activate ulm
pip install -r requirements.txt
```
