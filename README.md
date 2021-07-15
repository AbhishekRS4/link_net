# LinkNet implementation on Cityscapes dataset

## Notes
* Implementation of LinkNet
* The image dimension used to train the model is 1024x512
* 15 custom classes used

## Main idea
* Use features from the corresponding encoder stage in decoder [skip connection] by performing element-wise summation

## Intructions to run
* To list training options
```
python3 link_net_train.py --help
```
* To list inference options
```
python3 link_net_infer.py --help
```

## Visualization of results
* [LinkNet](https://youtu.be/qT2-NQb-sec)

## Reference
* [ResNet](https://arxiv.org/abs/1512.03385)
* [LinkNet](https://arxiv.org/pdf/1707.03718.pdf)
* [LinkNet Project](https://codeac29.github.io/projects/linknet/)
* [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
