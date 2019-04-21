# course18.fastai.com

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
- keras implementation
    - rather than single data obj, have a data generator (define augmentation, normalization)
        - augmentation params
    - pass directory of input, define batch size, classification type (binary, s/ else)
    - construct a base_model, manually add layers you want
    - freeze by manually iterating layers
    - using `pip install tensorflow-gpu keras`
    - replicate the same state-of-the-art algorithms (diff learning rates, stochastic grad desc w/ restarts, etc)
    - provides data_from_csv that can be helpful
- pandas
    - useful to take predictions and tabularize them
    - to_csv to turn predications into parseable format for eval
- cnn
    - convolution : a little matrix, 3x3, modifies every 3x3 part of an image with this matrix, add them all together to get the result
    - relu : throw away negative, only keep positive (relu(x) = max(0, x))
    - apply convlutional filter at each later, apply relu, max pool and repeat
    - 'activating' at the edges
        - some numbers from input, some kernel operation, get the output
    - second convolutional filter (different numbers in the matrix)
        - pytorch stores it as tensor, so stacking each filter together
    - first finds horizontal edges, second finds vertical edges
    - the two resulting layers then get modified by two matrices, twice
    - higher dimensional linear spaces
    - an architecture
        - how big is your kernel, 
          e.g. starts of with 2 3x3 convolutional kernels, result conv1
          then second layer, 2 2x3x3 kernel, result conv2
        - do you have max pooling, where does it happen
        - fully connected layer : give each activate a weight, compute sum product (every one of activation with its weight)
          - can have lots of weights e.g. VGG 4096x4096, 300M weights
          - resnets don't use very large fully connected layers
    - your image/dataset may have more channels than the model is intended for, in that case, youc an add a convolutional kernel that 'ignores' the extra channel, or reconciles data to compress that information into the number of channels needed
    - in reality, the matrices that define convolutional kernels are computed by starting with random numbers and doing sgd (stochastic gradient descent)
    - activation function : function applied to activates, f(x) = y, e.g. relu
        - stacking linear functions will give linear result
        - we add non-linearity with activation function
    - softmax : outputs number between 0 and 1, set of numbers that add up to 1
        - softmax(x) = e ^ (output) / sum_{k} (e ^ (output_k)) 
        - useful for predicting singular this, not good at predicting multiple things
        - not useful for multi-label classification problem
    - softmax is compared to one-hot-encoded label (0 or 1 : only one label is true)
    - sigmoid is the activation fucntion used for multi-label
        - sigmoid(x) = x / 1+x
        - s shaped graph, asymptotes top to 1, and bottom to 0 (so good for prob)
- pytorch
    - dl : gives back a minibatch (the next one)
    - ds : data set, single data item
    - dl is a python iterator
    - more pythonic
- planet images
    - aren't like imagenet
    - nothing in imagenet looks like satellite images
    - earlier layers are unlike imagenet
- metrics
    - accuracy
    - f2 : beta = 2 for f_beta
    - f_beta : how much do you way false negatives against false positives
    - confusion matrix
    - custom (can write your own – f(predictions, targets) = ... )
- dropout
    - if you output `learner` you'll see the layers that it has
    - dropout (p = 0.5)
        - go through, with a 50% chance just delete them
        - probability of deleting a cell in a layer
        - throwing away half of the activations
        - for each minibatch, throw away a different layer
        - prevents overfitting
        - critical in making modern dl work (hinton's work)
        - pytorch doubles activations when dropping half to keep the mean same
    - learner can have multiple linear layers, we only need 1, size of data set to number of classes
- overfitting : training loss much lower than validation

## Structured data and time series
- relation data
    - pandas merge to join datasets together
    - categories : data that is unrelated e.g. store id 1 and id 2 
        different level of 'year' (2016, 17, 18), treat it differently
        cadinality : number of levels or different types
        modeling decision (age, or week etc)
    - continuous : variables that are feeded into fully connected layers
        often time, floating point numbers, hard to make categorical
- starting with smaller data set to experiment
    - images : resized to 64x64
    - structured data : random sample
- categories may be mapped to 0-based integers e.g. years (2016, 17, 18) => (0, 1, 2) so that matrix formed is 1 or 2 in size rather than 2017/18
- embeddings
    - forget about categorical, think about continuous vars, e.g. 20 continuous vars
    - take 1d array, vector or rank 1 tensor (1x20)
    - put it through matrix product 1x20 X 20x100 => 1x100 => (relu) => X 100x1 => single number
    - above is a 1-layer neural net, in practice, have multiple layers, e.g.
        1x100 X 100x50 => 1x50 X 50x1 => single number
    - categorical
        - days of the week 0 (Sat) - 6 (Fri)
        - 1x7 X 7x4 => each day : 1x4 which we can then plug into our neural net
        - repeat until optimal weights are found using gradient descent on loss
        - just another set of weights
        - embedding matrix : the 7x4 matrix
        - appended to our continuous parameters
        - build embedding matrix for each categorical matrix
        - once you have an embedding matrix, you can choose to share and reused (pinterest, instacart)
        - higher dimensionality vector gives dl a chance to learn deeper representations
        - concept in neural network is higher dimensions
        - embedding is suitable for categorical variables
            - hard to work for something that has very high cardinality
- neural net
    - rank 1 tensor input -> lin -> relu -> lin -> s/max -> output
    - can add more lin layers
    - can add dropout
    - don't need as much feature engineering, time series are simply catgeories via embedding matrices
    - steps
        - 1 : list out cat/cont variables, pandas df
        - 2 : list of which row indexes you want in validation set
        - 3 : Columnar model from data
        - 4 : how big you want embedding matrix to be
        - 5 : call get_learner
        - 6 : call .fit()
    - data augmentation : can be done, but haven't seen it being done for structured data dl
    - pinterest, instacart have some work published, yoshua bengio's group about kaggle comp

## NLP
- language modeling : given few words of a sentence, can you predict the next word of the sentence
- trained on arxiv papers, can learn not only english language, but also how to construct parts of the paper
- replicate the idea from cats vs dog
    - pretrain a model to do one thing
    - use it to do something else
- tokenize : separate words 
- language movies
    - all reviews are concated
    - 64 million words into 64 sections
    - a matrix which is 64 columns, 10 million block, so 10M x 64, each represents a batch
    - batch is length 64, each sequence of up to length 70 (this is randomized to "shuffle" in each epoch)
    - next(iter(...)) gives a 75x64
    - each column contains multiple sentences
    - vocab is a tensor - all the possilbe tokens
    - embedding matrix for the vocab, where size(rows) = size(vocab)
    - words are categorical variables, high cardinality, only variables
    - the only categorical variable
    - length of embedding = 200 (between 50 and 600), has a lot more nuance than e.g. day of week
    - we use an Adam optimizer, will later learn what that is
    - there are existing pre-trained embedding matrices that you can download/use
        - e.g. word2vec
        - building a whole trained language model worked more effectively
        - model : rnn using lstm
    - min freq = n := if word occurs less than n times, ignore
    - words are mapped to integers (vocab read stored array index, sorted by freq)
        - turned into vectors
        - deep learning helps build context into understanding language
    - each word in vocab becomes an embedding matrix, so if 34945 in vocab, we end up with 34945 x n matrix where n is the size of the embedding matrix for each of the words in the vocab
        - just like a categorical variable 
        - set to 200, > 50 since a word has a lot more nuance to it than a particular concept in a data
    - dropout : spread over the model, will be covered later, prevents overfitting
    - learning rate clipping : early short circuit preventing large learning rate
- using the pretrained model
    - using the same vocab, each word maps to the earlier integer
    - do the same thing is cats vs dogs
        - unfreeze and retrain specific layers
- idea : training language models on medical journals, make it downloadable, let someone finetune it for prostrate cancer
