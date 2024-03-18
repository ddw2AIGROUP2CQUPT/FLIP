# FLIP (Facial Language Image Pretrain)

This repository is the official implementation of [FaceCaption-15M]().

**Overview of FLIP architecture.**

![image-20240318101027127](https://img.yutangli.net/img/202403181010116.png)

 **(a). Same color represents shared parameters. “12x” stands for 12-layer transformer modules. (b), (c) and (d) FLIP-based model are applied to the tasks of text-image retrieval, facial attributes prediction and sketch less facial image retrieval, respectively.**

## Training

Coming soon......（Only for the datasets been published, the code of training is meaningful.）

```shell
python pretrain.py > log.log
```

## Evaluation

Coming soon...... 

## Pre-trained Models

Coming soon......

## Datasets

> **Coming soon......**

**Overview of our proposed FaceCaption-15M containing over 15 million facial image-text (right and left) pairs.**

![image-20240318100601414](https://img.yutangli.net/img/202403181006981.png)

**Comparisons with other popular facial image datasets.**

![image-20240318100734131](https://img.yutangli.net/img/202403181007778.png)

**Image quality score distribution.**

![image-20240318100849106](https://img.yutangli.net/img/202403181008178.png)

**Text distribution.**

![image-20240318100913176](https://img.yutangli.net/img/202403181009312.png)

## Results

### Task1: Text-Image Retrieval

**Comparison with other classical pretrained models. All pretrained model backbones are frozen, with only the linear layer being fine-tuned. † represents the model pretrained on the LAION-Face [86] dataset; * represents the model pretrained on the FaceCaption dataset constructed without using LLM text generation.**

![](https://img.yutangli.net/img/202403181015142.png)

### Task2: Facial Attributes Prediction

**Comparison with other classical models. † represents the model pre-trained on the original LAION-Face dataset.**

![image-20240318101126897](https://img.yutangli.net/img/202403181011115.png)

### Task3: Sketch Less Facial Image Retrieval

**Comparative results with different baseline methods. † represents the model pre-trained on the LAION-Face dataset.**

![image-20240318101633671](https://img.yutangli.net/img/202403181016876.png)

**Performance of early retrieval in SLFIR problem. Instead of showing the complete sketch, we visualized it using the percentage of sketch. A higher value indicates a better early retrieval performance.**

![image-20240318101704679](https://img.yutangli.net/img/202403181017013.png)

## Citations & Contacts

> Coming soon......
