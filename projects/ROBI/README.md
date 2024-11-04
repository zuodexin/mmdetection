#### Training commands

In MMDetection's root directory, run the following command to train the model:

```bash
python tools/train.py projects/ROBI/configs/faster-rcnn_dummy-resnet_fpn_1x_robi.py
```


#### Test commands

In MMDetection's root directory, run the following command to test the model:

```bash
python tools/test.py projects/ROBI/configs/faster-rcnn_dummy-resnet_fpn_1x_robi.py  --show --out result.pkl ${CHECKPOINT_PATH}
```