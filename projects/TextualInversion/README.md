#### Training commands

In MMDetection's root directory, run the following command to train the model:

```bash
python tools/train.py projects/TextualInversion/configs/grounding_dino_swin-t_finetune_8xb2_20e_robi.py 
```

```bash
python tools/train.py projects/TextualInversion/configs/grounding_dino_swin-t_finetune_text_8xb2_20e_robi.py 
```


#### Test commands

In MMDetection's root directory, run the following command to test the model:

```bash
python tools/test.py projects/TextualInversion/configs/grounding_dino_swin-t_finetune_8xb2_20e_robi.py  ${CHECKPOINT_PATH} # --show --out result.pkl

``` 
```bash
python tools/test.py projects/TextualInversion/configs/grounding_dino_swin-t_finetune_text_8xb2_20e_robi.py  ${CHECKPOINT_PATH} # --show --out result.pkl
``` 