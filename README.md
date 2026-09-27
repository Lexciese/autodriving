Work in progress

Lidar-only waypoint prediction for autonomous driving. 

An adjustment for DeepIPCv2.

This project is structured as a package, you need to install the package by `pip install -e .`

There is one global config in common/config.py

`prepossessing` folder is used to develop dataset and do preprocessing like convert lidar to BEV (preprocessing/preprocessing_lidar.py). 

`model` folder contains model, dataloader script, train script, and test script.

O. Natan and J. Miura, “DeepIPCv2: LiDAR-powered Robust Environmental Perception and Navigational Control for Autonomous Vehicle,” IEEE Access, vol. 13, pp. 216290-216301, Dec. 2025.

## Tutorial Generate Dataset dari ROSBAG

1. Tidak perlu replay rosbag, langsung jalankan:
    - python3 ros2_extract_data.py
2. postprocessing, generate pseudolabel, dll
    - python3 create_route.py
    - python3 generate_histo.py
    - python3 generate_camsegdep.py
    - python3 generate_lidsegdep.py
3. join semua data untuk mengecek semuanya termasuk waypoints, route points, dll
    - python3 join_data_all.py