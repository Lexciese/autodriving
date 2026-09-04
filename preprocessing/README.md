# Tutorial Generate Dataset dari ROSBAG

1. Tidak perlu replay rosbag, langsung jalankan:
    - python3 ros2_extract_data.py
2. postprocessing, generate pseudolabel, dll
    - python3 create_route.py
    - python3 generate_histo.py
    - python3 generate_camsegdep.py
    - python3 generate_lidsegdep.py
3. join semua data untuk mengecek semuanya termasuk waypoints, route points, dll
    - python3 join_data_all.py