# HFSN

This is the code for the work: "HFSN: A Hybrid Frequency-Spatial Network for Content-Adaptive Image Watermarking".

## To train the network

First edit `config.py` to point `DATA_ROOT` to your dataset directory (it must contain `train/` and `val/` sub-folders of images), then run:

```bash
python main.py
```

All training hyper-parameters (batch size, image size, message length, epochs, learning rate, loss weights, noise layer list) are read from `config.py`, so you do not need to pass them on the command line. Useful knobs:

| Parameter | Default | Meaning |
|---|---|---|
| `BATCH_SIZE` | 16 | training batch size |
| `IMAGE_SIZE` | 128 | input image resolution (128x128) |
| `MESSAGE_LENGTH` | 64 | secret message bit length |
| `NITER` | 100 | total training epochs |
| `LR` | 0.001 | Adam learning rate |
| `SAVED_MODE` | False | set True to resume from `SAVED_MODE_PATH` |

Checkpoints, training images, validation images, logs and a copy of the experiment code are saved under `./training/<hostname>_<timestamp>_<remark>/`.

PS: Before training, make sure your dataset is organised like the COCO layout, for example:

```
D:\Python_Dataset\COCO\
├── train\        # training images
└── val\          # validation images
```

The default noise layers enabled in `config.py` are: Resize, GaussianBlur, Dropout, Cropout, MedianBlur, GaussianNoise, SaltAndPepper, JPEG, GridCrop, Identity. You can edit the `NOISE_LAYERS` list to ablate individual attacks.
