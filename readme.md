This repository includes two subfolders. Each folder is a `Python` project that can run independently. More detailed running instructions for each project can be found in the `Readme` of the sub files.

# Requirements

You need `Python 3.9`, `R`, and `RStudio` to run the code.

# How to obtain the data and run the project?

- Build the new folder separately named `data/` in `oss_community_evolution_index_and_shapelets/oss_community_evolution_shapelets/` and `oss_community_evolution_index_and_shapelets/oss_community_evolution_indexes/`

- Download the data from [oss_community_evolution_index_and_shapelets_data.rar](https://box.nju.edu.cn/f/f5530c2139ad4095a622/), and copy data separately to `data/`

- Run the projects according to the `readme` of each project. 

- It should be noted that the data of Project `oss_community_evolution_indexes` is raw data obtained through crawling and data preprocessing, while the raw data of Project `oss_community_evolution_shapelets` is generated and filtered based on the results of Project `oss_community_evolution_indexes`. For the convenience of running and reproducing, we have stored a confirmed data in project `oss_community_evolution_shapelets`, instead of running project `oss_community_evolution_indexes` again every time to calculate the index.
