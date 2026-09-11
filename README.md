Work in progress

Lidar-only waypoint prediction for autonomous driving. 

An adjustment for DeepIPCv2.

This project is structured as a package, you need to install the package by `pip install -e .`

There is one global config in common/config.py

`prepossessing` folder is used to develop dataset and do preprocessing like convert lidar to BEV (preprocessing/preprocessing_lidar.py). 

`model` folder contains model, dataloader script, train script, and test script.

O. Natan and J. Miura, “DeepIPCv2: LiDAR-powered Robust Environmental Perception and Navigational Control for Autonomous Vehicle,” IEEE Access, vol. 13, pp. 216290-216301, Dec. 2025.
