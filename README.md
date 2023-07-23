# DESIGN A CNN USING DILATION CONVOLUTION AND DEPTH WISE CONVOLUTION

## TARGET :
* The architecture must have 4 blocks (C1,C2,C3,C40) (No MaxPooling,  3 convolutions per block, where the last one has a stride of 2 )
* RF must be more than 44
* one of the layers must use Depthwise Separable Convolution
* one of the layers must use Dilated Convolution
* use albumentation library (horizontal flip, shiftScaleRotate, coarseDropout)
* Total Params to be less than 200k
* Acheive 85% accuracy

## CONTENTS :
- [DATASET](#dataset)
- [IMPORTING_LIBRARIES](#importing_libraries)
- [SET_THE_ALBUMENTATIONS](#set_the_albumentations)
- [DATA_AUGMENTATIONS](#data_augmentations)
- [SET_DATA_LOADER](#Set_Data_Loader)
- [CNN_MODEL](#cnn_model)
- [TRAINING_THE_MODEL](training_the_model)
- [RESULTS](results)


## DATASET 
### CIFAR DATASET
CIFAR-10 is an established computer-vision dataset used for object recognition. It is a subset of the 80 million tiny images dataset and consists of 60,000 32x32 color (RGB) images containing one of 10 object classes, with 6000 images per class.

## IMPORTING_LIBRARIES
Import the required libraries. 
* NumPy is used for for numerical operations.The torch library is used to import Pytorch.
* Pytorch has an nn component that is used for the abstraction of machine learning operations. 
* The torchvision library is used so that to import the CIFAR-10 dataset. This library has many image datasets and is widely used for research. The transforms can be imported to resize the image to equal size for all the images. 
* The optim is used train the neural Networks.
* MATLAB libraries are imported to plot the graphs and arrange the figures with labelling
* Albumenations are imported for Middle Man's Data Augmentation Strategy
* cv2 is imported 

## SET_THE_ALBUMENTATIONS
* cv2.setNumThreads(0) sets the number of threads used by OpenCV to 0. This is done to avoid a deadlock when using OpenCV’s resize method with PyTorch’s dataloader1.

* cv2.ocl.setUseOpenCL(False) disables the usage of OpenCL in OpenCV2 and is used when you want to disable the usage of OpenCL.

* The  class is inherited from torchvision.datasets.CIFAR10. It overrides the __init__ and __getitem__ methods of the parent class. The __getitem__ method returns an image and its label after applying a transformation to the image3. (This is to be done while using Albumenations)


## DATA_AUGMENTATIONS
For this import albumentations as A

Middle-Class Man's Data Augmentation Strategy is used. Like
### Normalize
<pre>
Syntax:
     A.Normalize(
            mean = (0.4914, 0.4822, 0.4465),
            std = (0.2470, 0.2435, 0.2616), always_apply = True)

Normalization is a common technique used in deep learning to scale the pixel values of an image to a standard range. This is done to ensure that the input features have similar ranges and are centered around zero. 
Normalization is done with respect to mean and standard Deviation.
For CIFAR10 (RGB) will have 3 means and 3 standard devivation that is equal to 
(0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
Normalize all the iamges
Applied to training and test data
</pre>
### HorizontalFlip :
<pre>
Syntax : A.HorizontalFlip()
Flip the input horizontally around the y-axis.
Args:     
p (float): probability of applying the transform. Default: 0.5.
Applied only to Training data
</pre>
### ShiftScaleRotate :
<pre>
Syntax:
A.ShiftScaleRotate (shift_limit=(-0.2,0.2), scale_limit=(-0.2,0.2), rotate_limit=(-15, 15), p=0.5)
Randomly apply affine transforms: translate, scale and rotate the input.

    Args:
        shift_limit ((float, float) or float): shift factor range for both height and width. If shift_limit
            is a single float value, the range will be (-shift_limit, shift_limit). Absolute values for lower and
            upper bounds should lie in range [0, 1]. Default: (-0.0625, 0.0625).
        scale_limit ((float, float) or float): scaling factor range. If scale_limit is a single float value, the
            range will be (-scale_limit, scale_limit). Note that the scale_limit will be biased by 1.
            If scale_limit is a tuple, like (low, high), sampling will be done from the range (1 + low, 1 + high).
            Default: (-0.1, 0.1).
        rotate_limit ((int, int) or int): rotation range. If rotate_limit is a single int value, the
            range will be (-rotate_limit, rotate_limit). Default: (-45, 45).
        p(float): probability of applying the transform. Default: 0.5.

Applied only to Training data
</pre>
### PadIfNeeded
<pre>
Syntax:
    A.PadIfNeeded(min_height=36, min_width=36, p=1.0),
PadIfNeeded is an image augmentation technique that pads the input image on all four sides if the side is less than the desired number. The desired number is specified by the min_height and min_width parameters. In this case padding is equal to 4.

</pre>
### RandomCrop
<pre>
Syntax:
  A.RandomCrop(height=32, width=32, always_apply = False,p=1.0),

RandomCrop is an image augmentation technique that crops a random part of the input and rescales it to some size without loss of bounding boxes. The height and width parameters specify the size of the crop. In this case iamge is cropped to size 32 X 32
</pre>
### CenterCrop
<pre>
A.CenterCrop(32, 32, always_apply=False, p=1.0)
It crops the center square of an image with a side length of 32 pixels. The always_apply parameter is set to False by default, which means that the transformation will not be applied to all images in the dataset. The p parameter is set to 1.0 by default, which means that the transformation will be applied to all images with a probability of 100%

</pre>
### Cutout
<pre>
Syntax:
 A.CoarseDropout(max_holes = 1, max_height=8, max_width=8, min_holes = 1, min_height=8, min_width=8,
                        fill_value=(0.4914, 0.4822, 0.4465), always_apply = True)
 It is similar to cutout

    Args:
        max_holes(int): The maximum number of rectangular regions to be masked. (for CIFAR10 Dataset its 32X32)
        max_height(int): The maximum height of the rectangular regions. 
        max_width(int): The maximum width of the rectangular regions.
        min_holes(int): The minimum number of rectangular regions to be masked.
        min_height(int): The minimum height of the rectangular regions.
        min_width(int): The minimum width of the rectangular regions.
        fill_value(float): The value to be filled in the masked region. It can be a tuple or a single value . 
            It is usually equal to the mean of dataset for CIFAR10 its (0.4914, 0.4822, 0.4465)
        always_apply = True - Applies to all the images
       
Applied only to Training data 
</pre>

### ToTensorV2
<pre>
Syntax:
    ToTensorV2()

To make this function work we need to ToTensorV2 from albumentations.pytorch.transforms
It is a class in the PyTorch library that converts an image to a PyTorch tensor. It is part of the torchvision.transforms module and is used to preprocess images before feeding them into a neural network. 

Applied to training and test data
</pre>
 #### PRINTED TRAIN_TRANSFORMS and TEST_TRANSFORMS 
 <pre>
 Files already downloaded and verified
Files already downloaded and verified
Compose([
  HorizontalFlip(always_apply=False, p=0.5),
  ShiftScaleRotate(always_apply=False, p=0.5, shift_limit=(-0.05, 0.05), scale_limit=(-0.050000000000000044, 0.050000000000000044), rotate_limit=(-15, 15), interpolation=1, border_mode=4, value=None, mask_value=None),
  PadIfNeeded(always_apply=False, p=1.0, min_height=36, min_width=36, border_mode=4, value=None, mask_value=None),
  RandomCrop(always_apply=False, p=1.0, height=32, width=32),
  CenterCrop(always_apply=False, p=1.0, height=32, width=32),
  CoarseDropout(always_apply=False, p=0.5, max_holes=1, max_height=16, max_width=16, min_holes=1, min_height=16, min_width=16),
  Normalize(always_apply=True, p=1.0, mean=(0.4914, 0.4822, 0.4465), std=(0.247, 0.2435, 0.2616), max_pixel_value=255.0),
  ToTensorV2(always_apply=True, p=1.0),
], p=1.0, bbox_params=None, keypoint_params=None, additional_targets={})
Compose([
  Normalize(always_apply=True, p=1.0, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.201), max_pixel_value=255.0),
  ToTensorV2(always_apply=True, p=1.0),
], p=1.0, bbox_params=None, keypoint_params=None, additional_targets={})
</pre>

## SET_DATA_LOADER
* Batch Size = 128
* Number of Workers = 2
* CUDA is used

#### PRINTED TRAIN and TEST LOADER:
<pre>
<torch.utils.data.dataloader.DataLoader object at 0x7b78d473b2b0>
length of train_loader 391
<torch.utils.data.dataloader.DataLoader object at 0x7b78d473b730>
length of test_loader 79
</pre>
#### SAMPLE IMAGES IN TRAIN LOADER
![alt text]sample images train loader

## CNN_MODEL

#### MODEL
<pre>
Net(
  (convblock1): Sequential(
    (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): ReLU()
    (2): BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout(p=0.01, inplace=False)
  )
  (convblock2): Sequential(
    (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): ReLU()
    (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout(p=0.01, inplace=False)
  )
  (convblock3): Sequential(
    (0): Conv2d(64, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=(2, 2), bias=False)
  )
  (convblock4): Sequential(
    (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): ReLU()
    (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout(p=0.01, inplace=False)
  )
  (convblock5): Sequential(
    (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), groups=64, bias=False)
    (1): ReLU()
    (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout(p=0.01, inplace=False)
  )
  (convblock6): Sequential(
    (0): Conv2d(64, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), dilation=(2, 2), bias=False)
  )
  (convblock7): Sequential(
    (0): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): ReLU()
    (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout(p=0.01, inplace=False)
  )
  (convblock8): Sequential(
    (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    (1): ReLU()
    (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout(p=0.01, inplace=False)
  )
  (convblock9): Sequential(
    (0): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), groups=64, bias=False)
    (1): ReLU()
    (2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    (3): Dropout(p=0.01, inplace=False)
  )
  (gap): Sequential(
    (0): AdaptiveAvgPool2d(output_size=1)
  )
  (convblock10): Sequential(
    (0): Conv2d(64, 10, kernel_size=(1, 1), stride=(1, 1), bias=False)
  )
  (dropout): Dropout(p=0.01, inplace=False)
)
</pre>



## TRAINING_THE_MODEL
The train function takes the model, device, train_loader, optimizer, and epoch as inputs. It performs the following steps:

* Sets the model to train mode, which enables some layers and operations that are only used during training, such as dropout and batch normalization.
* Creates a progress bar object from the train_loader, which is an iterator that yields batches of data and labels from the training set.
* Initializes two variables to keep track of the number of correct predictions and the number of processed samples.
* Loops over the batches of data and labels, and performs the following steps for each batch:
* Moves the data and labels to the device, which can be either a CPU or a GPU, depending on what is available.
* Calls optimizer.zero_grad() to reset the gradients of the model parameters to zero, because PyTorch accumulates them on subsequent backward passes.
* Passes the data through the model and obtains the predictions (y_pred).
* Calculates the loss between the predictions and the labels using the negative log-likelihood loss function (F.nll_loss).
* Appends the loss to the train_losses list for later analysis.
* Performs backpropagation by calling loss.backward(), which computes the gradients of the loss with respect to the model parameters.
* Performs optimization by calling optimizer.step(), which updates the model parameters using the gradients and the chosen optimization algorithm (such as SGD or Adam).
* Updates the progress bar with the current loss, batch index, and accuracy. The accuracy is computed by comparing the predicted class (the index of the max log-probability) with the true class, and summing up the correct predictions and processed samples.
* Appends the accuracy to the train_acc list for later analysis.

The test function takes the model, device, and test_loader as inputs. It performs the following steps:

* Sets the model to eval mode, which disables some layers and operations that are only used during training, such as dropout and batch normalization.
* Initializes two variables to keep track of the total test_loss and the number of correct predictions.
* Uses a torch.no_grad() context manager to disable gradient computation, because we don’t need it during testing and it saves memory and time.
* Loops over the batches of data and labels from the test set, and performs the following steps for each batch:
* Moves the data and labels to the device, which can be either a CPU or a GPU, depending on what is available.
* Passes the data through the model and obtains the output (predictions).
* Adds up the batch loss to the total test loss using the negative log-likelihood loss function (F.nll_loss) with reduction=‘sum’, which means it returns a scalar instead of a vector.
* Compares the predicted class (the index of the max log-probability) with the true class, and sums up the correct predictions.
* Divides the total test loss by the number of samples in the test set to get the average test loss, and appends it to the test_losses list for later analysis.

* creates an instance of the Adam optimizer, which is a popular algorithm that adapts the learning rate for each parameter based on the gradient history and the current gradient. You pass the model parameters, the initial learning rate (lr), and some other hyperparameters to the optimizer constructor. 
* creates an instance of the OneCycleLR scheduler, which is a learning rate policy that cycles the learning rate between two boundaries with a constant frequency. You pass the optimizer, the maximum learning rate (0.01), the number of epochs (30), and the number of steps per epoch (len(train_loader)) to the scheduler constructor.
* Defines a constant for the number of epochs = 30, which is the number of times you iterate over the entire training set.
* Prints out a summary of the average test loss, accuracy, and number of samples in the test set. 


## [RESULTS]

I achieved 85% of accuracy at 50th epoch. Initially model is under fitting. As number of epochs increased it worked good.

I missed the screen shot of result.


#### MODEL SUMMARY
<pre>
Requirement already satisfied: torchsummary in /usr/local/lib/python3.10/dist-packages (1.5.1)
cuda
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 32, 32]             864
              ReLU-2           [-1, 32, 32, 32]               0
       BatchNorm2d-3           [-1, 32, 32, 32]              64
           Dropout-4           [-1, 32, 32, 32]               0
            Conv2d-5           [-1, 64, 32, 32]          18,432
              ReLU-6           [-1, 64, 32, 32]               0
       BatchNorm2d-7           [-1, 64, 32, 32]             128
           Dropout-8           [-1, 64, 32, 32]               0
            Conv2d-9           [-1, 32, 15, 15]          18,432
           Conv2d-10           [-1, 64, 15, 15]          18,432
             ReLU-11           [-1, 64, 15, 15]               0
      BatchNorm2d-12           [-1, 64, 15, 15]             128
          Dropout-13           [-1, 64, 15, 15]               0
           Conv2d-14           [-1, 64, 15, 15]             576
             ReLU-15           [-1, 64, 15, 15]               0
      BatchNorm2d-16           [-1, 64, 15, 15]             128
          Dropout-17           [-1, 64, 15, 15]               0
           Conv2d-18             [-1, 32, 7, 7]          18,432
           Conv2d-19             [-1, 64, 7, 7]          18,432
             ReLU-20             [-1, 64, 7, 7]               0
      BatchNorm2d-21             [-1, 64, 7, 7]             128
          Dropout-22             [-1, 64, 7, 7]               0
           Conv2d-23             [-1, 64, 7, 7]          36,864
             ReLU-24             [-1, 64, 7, 7]               0
      BatchNorm2d-25             [-1, 64, 7, 7]             128
          Dropout-26             [-1, 64, 7, 7]               0
           Conv2d-27             [-1, 64, 5, 5]             576
             ReLU-28             [-1, 64, 5, 5]               0
      BatchNorm2d-29             [-1, 64, 5, 5]             128
          Dropout-30             [-1, 64, 5, 5]               0
AdaptiveAvgPool2d-31             [-1, 64, 1, 1]               0
           Conv2d-32             [-1, 10, 1, 1]             640
================================================================
Total params: 132,512
Trainable params: 132,512
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 4.19
Params size (MB): 0.51
Estimated Total Size (MB): 4.70
----------------------------------------------------------------
</pre>

* Total number of Convolution Layers = 13

* Total Receptive Field is greater than 44 (49 in this case)
![alt text]RF_FIELD

# ANALYSIS:
* Usage of albumenations is understood.
* This model took more time/epoch compared to previous models.
* Deptwise Convoltions, Dilation is used with maxpooling and striding.
* This model can be trained further.
* Acheived No of parametrs < 200K
* Accuracy > 85%
* RF > 44



