--------------------------------------------------------------------------------
This repository contains the code and data for paper 'Quantifying Community Evolution in Developer Social Networks'.



## Requirements

You need `Python 3.9`, `R`, and `RStudio` to run the code.



## Important files
- The entry of Python code for community detection and index calculation is in `main_calculate_indexes.py`
- Important settings defined in `global_settings.py`
- Community detection code in `community_detection.py`
- Community evolution indices are calculated in `calculate_indexes.py`
- Data are located at `data/`
- After executing `main_calculate_indexes.py`, the results will be recorded in an auto-generated subfolder in the `result/` folder of the project
- The R script `RScript/productivity_analysis.Rmd` is used to analyze the correlation between community evolution indices and team productivity




## How to obtain the results

### Concurrent Validity: Community evolution pattern detection

- Set the threshold of community matching used for existing approach in `global_settings.py`, e.g., `EVOLUTION_PATTERN_THRESHOLD = 0.3`
- Run `main_calculate_indexes.py`
- Find the results in `concurrent_validity.txt` in the auto-generated subfolder under `result/`


### Discriminant Validity: Spearman's Correlations Coefficients between pairs of indices

- Run `main_calculate_indexes.py` （Optional if you've already run the code for concurrent validity）
- Find the results in `discrimant_validity.txt` in the auto-generated subfolder under `result/`


### Regression Analysis: Correlation with team productivity

- Run `main_calculate_indexes.py` （Optional if you've already run the code for concurrent validity）
- Find the path of file `index_productivity.csv` in the auto-generated subfolder under `result/`
  - For example: `../result/2022-03-13T10-07-10Z_interval_7_days_x_12/index_productivity.csv`
- Modify line 22 of `RScript/productivity_analysis.Rmd` to specify the path of the data file.
  - For example: 
```r
table_data<-read.table("../result/2022-03-13T10-07-10Z_interval_7_days_x_12/index_productivity.csv", head=T, sep=',', stringsAsFactors = FALSE)
```
- Run `RScript/productivity_analysis.Rmd` in `RStudio`, and get the results.
