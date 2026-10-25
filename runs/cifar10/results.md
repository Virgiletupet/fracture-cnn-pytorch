# CIFAR-10 from scratch

20 epochs each, AdamW + one-cycle schedule, crop and flip augmentation. Model selection on 5,000 validation images, test (10,000) scored once.

| Model | Parameters | Seconds per epoch | Best val accuracy | Test accuracy |
|---|---|---|---|---|
| basic_cnn | 122,570 | 2.8 | 0.809 | 0.805 |
| vgg_bn | 1,148,874 | 24.8 | 0.913 | 0.912 |
| depthwise_cnn | 140,106 | 19.4 | 0.872 | 0.864 |
| small_resnet | 1,084,586 | 40.1 | 0.916 | 0.913 |
