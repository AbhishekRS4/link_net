# LinkNet implementation on Cityscapes dataset

## Notes
* Implementation of LinkNet
* The image dimension used to train the model is 1024x512
* 15 custom classes used

## Main idea
* Use features from the corresponding encoder stage in decoder [skip connection] by summing them

## To do
- [x] LinkNet
- [ ] Compute metrics
- [ ] Sample output

## Reference
* [ResNet](https://arxiv.org/abs/1512.03385)
* [LinkNet](https://arxiv.org/pdf/1707.03718.pdf)
* [LinkNet Project](https://codeac29.github.io/projects/linknet/)
* [Cityscapes Dataset](https://www.cityscapes-dataset.com/)
