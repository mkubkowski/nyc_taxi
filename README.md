# NYC Taxi

## Setting up environment
```
conda create -n env_name -c conda-forge --file requirements.txt
```

## Description of files

* **1. Get_lat_long.ipynb** - Here I prepare data from https://s3.amazonaws.com/nyc-tlc/misc/taxi_zones.zip 
to get longitudes and latitudes.

* **2. NYC Taxi - data to parquet.ipynb** - Here I convert CSV files with yellow taxi trips data 
from January 2020-July 2021 to parquet files and calculate some features for efficiency.

* **3. NYC Taxi - main analyses.ipynb** - Main file with analyses.

* **additional_transformers_sklearn.py**, **utility_functions.py** - files with 
utility functions used in the main notebook

In folders **nyc_taxi_other**, **nyc_taxi_output** I have also included files which are 
optional to download, but can help get results quickly without waiting.