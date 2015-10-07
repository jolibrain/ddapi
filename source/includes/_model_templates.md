# Model Templates

> Example of a 3-layer MLP with 512 hidden units in each layer and PReLU activations:

```json
{"parameters":{"mllib":{"template":"mlp","nclasses":9,"layers":[512,512,512],"activation":"PReLU","nclasses":9}}
```

> Example of GoogleNet for 1000 classes of images:

```json
{"parameters":{"input":{"connector":"image","width":224,"height":224},"mllib":{"template":"googlenet","nclasses":1000}}
```

The DeepDetect server and API come with a set of Machine Learning model templates.

At the moment these templates are available for the [Caffe]() Deep Learning library. They include some of the most powerful deep neural net architectures for image classification, and other customizable classic and useful architectures.

## Neural network templates

All models below are used by passing their id to the `mllib/template` parameter in `PUT /services` calls:

Model ID | Type | Input | Description
-------- | ---- | ----- | -----------
lregression | linear | CSV | logistic regression
mlp | neural net | CSV | multilayer perceptron, fully configurable from API, see parameters below
convnet | convolutional neural net | Images | convolutional neural net, with layers configurable from API, see parameters below
alexnet | deep neural net | Images 227x227 | 'AlexNet', convolutional deep neural net, good accuracy, fast
cifar | deep neural net | Images 32x32 | Convolutional deep neural net, very good for small images
nin | deep neural net | Images 224x224 | 'Network in Network' convolutional deep neural net, good accuracy, very fast
googlenet | deep neural net | Images 224x224 | 'GoogleNet', convolutional deep neural net, best accuracy

## Parameters

- Caffe

Parameter | Type | Optional | Default | Description
--------- | ---- | -------- | ------- | -----------
nclasses | int | no | N/A | Number of output classes ("supervised" service type)
template | string | yes | empty | Neural network template, from "lregression", "mlp", "convnet", "alexnet", "googlenet", "nin"
layers | array of int | yes | [50] | Number of neurons per layer ("mlp" only)
layers | array of string | yes | [1000] | Type of layer and number of neurons peer layer: XCRY for X successive convolutional layers of Y filters followed by a max pooling layer, an int as a string for specifying the final fully connected layers size, e.g. \["2CR32","2CR64","1000"\] ("convnet" only)
activation | string | yes | relu | Unit activation ("mlp" only), from "sigmoid","tanh","relu","prelu"
dropout | real | yes | 0.5 | Dropout rate between layers ("mlp" only)
regression | bool | yes | false | Whether the model is a regressor ("mlp" and "convnet" only)
rotate | bool | yes | false | Whether to apply random rotations to input images ("convnet" only)
mirror | bool | yes | false | Whether to apply random mirroring of input images ("convnet" only)