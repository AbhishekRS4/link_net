# LinkNet implementation on Cityscapes dataset

## Notes
* Implementation of LinkNet
* The image dimension used to train the model is 1024x512
* 15 custom classes used

## Main idea
* Use features from the corresponding encoder stage in decoder [skip connection] by performing element-wise summation

## Intructions to run
> To run training use - **python3 link\_net\_train.py -h**
>
> To run inference use - **python3 link\_net\_infer.py -h**
>
> This lists all possible commandline arguments

## Visualization of results
* [LinkNet](https://youtu.be/qT2-NQb-sec)

## Reference
* [ResNet](https://arxiv.org/abs/1512.03385)
* [LinkNet](https://arxiv.org/pdf/1707.03718.pdf)
* [LinkNet Project](https://codeac29.github.io/projects/linknet/)
* [Cityscapes Dataset](https://www.cityscapes-dataset.com/)

## To do
- [x] LinkNet
- [x] Visualize results
- [ ] Compute metrics

