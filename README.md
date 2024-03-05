# GTA_ACLED_TKG

## Sparql Queries:
*TODO Lorenz? 

## Timestep Assignment:
* Run ts_assignment_gta_star.py and ts_assignment_acled.py to read the .nt files and assign timesteps
* This requires the rdflib package, that can be downloaded here https://github.com/XuguangSong98/rdflib and put into the data_preprocessing folder. Processing data with this package very slow and can take hours to days.
* The output are csv files that can be found in /data/acled and /data/gta respectively

## Merge the two datasets:
* Run merke-tkg-from-gta-acled.py to merge both subsets and create train, valid, test.txt
* Specify timerange of interest. In our case this is 2023-01-01 – 2023-12-31
* Split dataset based on timesteps. Specify train/valid/test split. In our case it is 80/10/10
* Automatically stores the resulting files in /data/crisis2023

## Dataset Analysis:
* Run analysis.py to plot figures of graph parameters over time
* Run tkg_analysis.py to extract Parameters such as Recurrency Degree and Consecutiveness Degree

## TKG Forecasting:
TODO describe
