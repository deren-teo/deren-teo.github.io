# Transfer learning using pre-trained networks

*Last updated: May 26, 2023*

Training a neural network is a daunting task. In 2016, ResNet-18 -- a popular convolutional neural network (CNN) for image classification -- took 3.5 days to train on not one, but *four* Nvidia Kepler GPUs[<sup>1</sup>](http://torch.ch/blog/2016/02/04/resnets.html). It was trained on [ImageNet](https://www.image-net.org/), a dataset of over a million images of 1000 different objects[<sup>2</sup>](https://www.image-net.org/download.php). Today, we'll be training ResNet-18 to classify flowers on my own single GPU in under a minute.

How?! You might exclaim. Has technology improved so much in this time? Certainly, GPUs have improved significantly since 2016, yet a single consumer-grade GPU is still no match to four, even from 2016. Instead, the magic lies in the use of *transfer learning* to fine-tune a pre-trained ResNet-18 model. In this post, we'll see an example of fine-tuning ResNet-18 to classify 5 types of flowers.

## Transfer learning in a nutshell

Jason Brownlee provides a beginner-friendly introduction to transfer learning in [this article](https://machinelearningmastery.com/transfer-learning-for-deep-learning/). No need to switch tabs though, as we'll outline the key points here to understand just enough to appreciate the model we'll be fine-tuning.

To quote the opening sentences of the article:

> Transfer learning is a machine learning method where a model developed for a task is reused as the starting point for a model on a second task.
>
> It is a popular approach in deep learning where pre-trained models are used as the starting point on computer vision and natural language processing tasks given the vast compute and time resources required to develop neural network models on these problems...

Image classification is a non-trivial task, and the models which perform well are duly complex. Many highly successful models are based on the convolutional neural network (CNN) architecture. In learning to differentiate objects, CNNs have been observed to build a hierarchy of features of varying complexity, and classify objects based on combinations of these features. Therefore, a model trained (and successful) on a very large data set with many classes contains at a low level a generic image feature extractor. Given a model generic enough, the more specific layers can easily be fine-tuned to identify other objects as combinations of generic features; even objects which the model may not have trained on[<sup>3</sup>](https://machinelearningmastery.com/transfer-learning-for-deep-learning/).

This is the approach we'll be using to apply ResNet-18 to classify our flowers. ResNet-18 is a popular 18-layer CNN trained on the [ImageNet](https://www.image-net.org/) database of over one million images of 1000 different objects. This makes it suitable for transfer learning. We'll be using a pre-trained model and fine-tuning it for our particular flower classification task, and will observe that with less than 10 seconds of training, we will have ourselves a highly effective classifier.

## Classifying flowers using ResNet-18

The 5 flowers we'll classify are: chamomile, tulip, rose, sunflower, dandelion.

We'll get our data from the [Flowers Recognition](https://www.kaggle.com/datasets/alxmamaev/flowers-recognition) dataset on Kaggle, courtesy Alexander Mamaev. The dataset contains 4242 images in total; approximately 800 for each flower. The images aren't split into training and testing sets by default, so I've manually separated the first 50 images of each class to form our testing set. It's important to evaluate the model performance on completely separate data, as this avoids bias occurring from testing the model on data which influenced the model tuning.

We're now ready to create our model. In fast.ai, we start by creating a `DataBlock` to load the training data:

```python
dls = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=[Resize(192, method='squish')]
).dataloaders(path)
```

Note how we specify that the inputs are images and the outputs are categories. We also specify that the training data should be split such that 20% is used for validation and 80% for training. To make sure all our inputs are equal, we resize the images by squishing them before passing them through the model.

Here's a sample of some of the outputs of the `DataBlock`:

![](/images/2023-05-20_flowers_sample.png)

Now, the real magic happens. Let's use the pre-trained `resnet18` model provided by fast.ai and fine tune it over three epochs. This means the training data is passed through the model three times.

```python
learn = vision_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(3)
```

As the model is training, it should show both training and validation loss, and the error rate. The process is stochastic, so sometimes the model trains to perfection whereas other times it may make one or two errors. Regardless, the error should be very low. On a GPU, each epoch should only take a couple of seconds -- 18 on my computer. My model finished tuning with an error rate of 0.0898, representing a 91.02% accuracy on the validation data.

Let's now test it on our withheld test set to see how well the model generalises. We'll load the test set and run it through our model:

```python
test_path = Path(path, '..', 'test')
test_dl = learn.dls.test_dl(get_image_files(test_path), with_labels=True)
_, preds, labels = learn.get_preds(dl=test_dl, with_decoded=True)

preds, labels = preds.numpy(), labels.numpy()
```

Next, we compare the predictions made by the model to the true labels of the each image:

```python
correct = sum(preds == labels)
total = len(labels)
acc = 100 * sum(preds == labels) / len(labels)

print(f'Test accuracy: {acc:.2f}% ({correct}/{total})')
```
```bash
Test accuracy: 100.00% (250/250)
```

That's a pretty good result! Yours may not be exactly the same, but it should nevertheless be rather accurate.

## In summary

You're now witness to the wonder of transfer learning and how it accelerates model training for simple image classification problems. In minimal time and not too much code, we fine-tuned the popular ResNet-18 CNN to classify 5 types of flowers and achieved 100% accuracy on a test set of 250 images. Now equipped with this knowledge, go forth and make the most of the hours people have already spent training deep neural nets!

## References

[1] S. Gross and M. Wilber. "Training and investigating Residual Nets." (2016), [Online]. Available: http://torch.ch/blog/2016/02/04/resnets.html

[2] Stanford Vision Lab. "Download." (2020), [Online]. Available: https://www.image-net.org/download.php

[3] J. Brownlee. "A gentle introduction to transfer learning for deep learning." (2019), [Online]. Available: https://machinelearningmastery.com/transfer-learning-for-deep-learning/

[4] A. Mamaev. "Flowers recognition." (2021), [Online]. Available: https://www.kaggle.com/datasets/alxmamaev/flowers-recognition
