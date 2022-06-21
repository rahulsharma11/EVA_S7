# EVA_S7

```
# Training of CFAR10 dataset with use of DepthWise Convolution and Dilation Convolution
```

```
**DIALATION CONVOLUTION** - 
Dilated convolution is a way of increasing the receptive view (global view) of the network
exponentiallyand linear parameter accretion. With this purpose, it finds usage in applications
cares more about integrating the knowledge of the wider context with less cost.
```
```
Advantage - this convolution helps to increase the global receptive feild of the network. Thus helps not even classifying the object but also localizing them
```





```
** DEPTHWISE CONVOLUTION **- 
For a depthwise separable convolution on the same example, we first traverse the input channels
with 1 3x3 kernel each, giving us same feature maps output. Now, before merging anything, we traverse
these feature maps with <desired output channel> 1x1 convolutions each and only then start to them add together.
```

```
Advantage - this can be helpful to reduce the number of parameters where large parameters are contraint. Thus by reducing the number of parameters you can add few more convolutions to increase the model efficiency.
```






```
I Created Different Folders to make the training modular.

config         - contains the basic configuration parameters
dataset        - contains code to create the dataloaders for training and testing
model          - contains the model skeleton build code
pytorch-output - contains the output logs and saved model with best val accuracy
utils          - contains the basic utility like logging and trainer class.

```
