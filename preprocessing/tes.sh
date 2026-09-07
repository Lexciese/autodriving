for f in datasetx/2026-09-05_route00/lidar/cld/*.pcd; do
  b=$(basename "$f" .pcd)
  touch "datasetx/2026-09-05_route00/meta/$b.yml"
done
