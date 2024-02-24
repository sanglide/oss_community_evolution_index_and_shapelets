# oss_community_evolution_shapelets

Shapelets mining for community evolution indices. This folder contains the code and data for paper 'Measuring and Mining Community Evolution in Developer Social Networks with Entropy-Based Indices'.

--------------------------------------------------------------------------------

## Requirements

You need `Python 3.9`, `R`, and `RStudio` to run the code.

## Important files

- The entry of Python code for shapelets mining and prediction is in `main_shapelets.py`
- Important settings defined in `global_settings.py`
- The shapelets algorithm implemented in `shapelets.py`
- Data are located at `data/`
- After executing `main_shapelets.py`, the results will be recorded in an auto-generated subfolder in the `result/` folder of the project

## How to obtain the results

### Shapelets mining and prediction

- Set the threshold of community matching used for existing approach in `global_settings.py`
- Run `main_shapelets.py`
- Find the results in the auto-generated subfolder under `result/`

### Prediction of ARIMA model

- Run `ARIMA.py`

- Find the results in the auto-generated subfolder `ARIMA\`. 
