# fastai

## Cats vs Dogs

- labelled dataset of cats and dogs
    - (binary) classification problem
- being able to recongize cats vs dogs
- using a pre-trained model instead of building our own from scratch
- using resnet34, model that won the ImageNet competition in 2015
    - convolutional neural network (CNN)
- relying on fastai python library which provides basic helpers
    - provides models as variables
    - provides classifiers as functions
- recall ml concepts
    - sigmoid : 'S' curve
    - relu    : y = max(0, x) # set all negative values to 0, return positive values

```python
sz = 223 
arch = resnet34
data = ImageClassifier.from_paths(PATH, tfms=tfms_from_model(arch, sz))
learner = ConvLearner.pretrained(arch, data, precompute=True)

# second parameter is number of epochs
# too many epochs may lead to overfitting
# fix this by 'generating' more data
learner.fit(0.01, 2)


# differential learning rate
# Q: how do we know which layers get which learning rate applied?
lr = np.array([1e-4, 1e-3, 1e-2])
learner.fit(lr, cycle_len=1, cycle_mult=2)
```
- resnet34, being trained for ImageNet provides a classification for a 1000 classes
    - for cats vs dogs, we need to modify this to return 2 classes only
- results evaluation – look at items from the dataset for each of the following:
    - few correct labels at random
    - few incorrect labels at random
    - most correct labels for each class
    - most incorrect labels for each class
    - most uncertain labels for each class
- generate more data – data augmentation
    - transform images e.g. horizontal flipping, zooming, rotating
- when creating a learner, it sets all but the last layer to frozen. 
    - means that it's only updating the weights in the last layer when we call fit
    - use `learner.unfreeze()` to continue fine tuning other layers
    - other layers in the model have already been trained to recognize ImageNet photos
    - differential learning rate (diff learning rates for diff layers)
        - first few layers   : 1e-4 (basic geometric features)
        - next few layers    : 1e-3 (more complex features)
        - last few fc layers : 1e-2 (surface level features)
- `cycle_len` parameter
    - stochastic gradient descent with restarts
    - slowing down learning rate as getting closer to optimal point
    - learning rate annealing
        - (naive) pick one and descrease by 10x when you get close
        - (cosine) using one half of a cosine curve, start with high, low as you get close
        - (cycle len) jump to a high learning rate every so often, decrease it... (restarts)
            - cycle len (n) refers to restart after every n epoch
            - snapshot ensemble: snapshots of weights taken at every lowest learning rate point, averaged 
            - 
- `cycle_mult` parameter
    - multiplication factor to increase the cycle len after each cycle
    - more exploring every subsequent turn
- test time augmentation (TTA)
    - data augmentation at testing time
    - make predictions on data in validation set
    - also make predictions on number of randomly augmented versions
    - 10-20% reduction in error rate
- hyperparameters : cannot be determined from the training data
    - learning rate
    - number of epochs
    - cycle_len
    - cycle_mult
- learning rate
    - think about gradient descent and taking steps to get to a optimum
    - let step be x, then x_(n+1) = x_n + (dy/dx * alpha)
        - alpha : learning rate
    - hard to 'set' a learning rate
        - method to find it in a research paper, less people know about it
        - use mini batches – smaller sets of training data
            - learning rate increased
            - plot against loss
            - loss drops until it starts going up
            - pick learning rate which is 1 order of magnitude lower than the one at the lowest loss
- generalized model better because it will be able to apply to wider range of test (new) data
- confusion matrix
    - visualizing classification prediction results
    - cases
        - cats -> predicted cats (correct)
        - cats -> predicted dogs (incorrect)
        - dogs -> predicted cats (incorrect)
        - dogs -> predicted dogs (correct)
