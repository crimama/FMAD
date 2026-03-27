# Dinomaly Bootstrap

This directory is a bootstrap location for the upstream Dinomaly repository:

- upstream: `https://github.com/guojiajeremy/Dinomaly`
- local shell clone was not possible in this session because outbound network access is blocked
- `scripts/run_dinomaly.sh` will clone the upstream repository into `upstream/` when network access is available

Local additions in this repository:

- `patches/dataset_mvtecad2_robustad.patch`: optional dataset extension patch
- runtime result export to `results/phase1/dinomaly_{dataset}.json`
