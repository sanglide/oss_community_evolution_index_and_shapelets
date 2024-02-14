#!/bin/bash
conda run -n wlenv01 bash -c 'python3 -u ./main_calculate_indexes.py > ../output_main_calculate_indexes.log 2>&1 &'