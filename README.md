# Ripik_DataHack_Challenge_AV

Image Classification with Data Augmentation and EfficientNet 

This project performs image classification using transfer learning with EfficientNet models. The goal is to improve classification accuracy by using extensive data augmentation and model finetuning techniques.

The dataset consists of 7,200 images across different classes. Data augmentation is used to increase the number of training images from 7,200 to 108,000. Two EfficientNetB0 models are then trained on the original and augmented datasets separately. Finally, the models are finetuned and combined using an ensemble to produce the final predictions.

Data Augmentation
The following augmentations were applied to the original 7,200 images to produce 108,000 augmented images:

A.Resize(width=512, height=512), # Scale all images to 512x512

A.Flip(), # Random flip half the images horizontally

A.Transpose(), # Random transpose half the images  

A.RandomBrightnessContrast(),

A.MedianBlur(blur_limit=3),

A.MotionBlur(),

A.GaussNoise(), 

A.OpticalDistortion(),

A.GridDistortion(),

A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),  

A.HueSaturationValue(),

A.CLAHE(),

A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45), 

A.CoarseDropout()
The Albumentations library was used to perform the augmentations.

Model Training

Two EfficientNetB0 models were trained on the original and augmented datasets using the Adam optimizer and categorical cross entropy loss. Transfer learning was used by initializing the models with ImageNet weights. The last classification layer was updated during training while the other layers were frozen.

Model 1 Accuracy (original images): 80%

Model 2 Accuracy (augmented images): 82%

Model Finetuned Model : 93%

The original model (Model 1) was further finetuned by unfreezing all layers and retraining with a lower learning rate.

Finetuning significantly improved performance over the non-finetuned model.
