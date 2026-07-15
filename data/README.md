# Dataset layout

The default configuration expects the following files:

```text
data/
├── River/
│   ├── river_before.mat
│   ├── river_after.mat
│   └── groundtruth.mat
├── Farmland/
│   ├── farm06.mat
│   ├── farm07.mat
│   └── label.mat
└── Hermiston/
    └── Hermiston.mat
```

Expected MATLAB variables:

- River: `river_before`, `river_after`, and `groundtruth`
- Farmland: `imgh`, `imghl`, and `label`
- Hermiston: `hsi_t1`, `hsi_t2`, and `hsi_gt`

The loader also accepts several common alternative variable names.
