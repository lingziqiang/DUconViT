# Data Preparing

1. You can make the dataset by save_data_to_h5.py and save_data_to_npz. And you can make 
the list by make_lists.
2. The directory structure of the whole project is as follows:

```bash
.
├── DUconViT
│   ├──datasets
│   │       └── dataset.py
│   ├──train.py
│   ├──test.py
│   └──...
└── data
    └──dataset_name
        ├── test_vol_h5
        │   ├── case0001.npy.h5
        │   └── *.npy.h5
        └── train_npz
            ├── case0005_slice000.npz
            └── *.npz
```
