# Semantic Segmentation with Pre-trained SegFormer and FCN Models

This project focuses on the task of semantic segmentation using two different datasets: **OxfordIIITPet** and **Cityscapes**. The goal is to correctly segment objects in images, such as pets and street scenes, into their respective classes.

## Datasets

- **OxfordIIITPet**: This dataset contains images of pets, with the task of segmenting the pet, its boundary, and the background into three different classes.
- **Cityscapes**: A dataset of urban street scenes with 19 classes used for segmentation. The dataset has been preprocessed to exclude certain classes.

## Tasks

1. **Metrics Implementation**: We implemented the `SegMetrics` class to compute the mean Intersection over Union (mIoU), a critical metric for evaluating semantic segmentation models, as it handles class imbalances effectively.

2. **Training Loop**: The project adapts a training loop to accommodate the new segmentation task, tracking both training and validation metrics, specifically focusing on mIoU.

3. **Pre-trained Model Usage**: We employed the `fcn_resnet50` model from PyTorch's segmentation library, training it both from scratch and using pre-trained weights for the encoder. The performance was tracked using train and validation loss, as well as mIoU metrics.

4. **SegFormer Implementation**: The project included the completion of the SegFormer model, particularly focusing on the Transformer blocks and the lightweight decoder. The SegFormer was pre-trained on the Cityscapes dataset and fine-tuned on the OxfordIIITPet dataset to compare performance.

## Results

- **Model Comparison**: We compared models trained from scratch and those using pre-trained weights for both FCN and SegFormer models. Additionally, we explored fine-tuning with and without freezing the encoder.

- **Visualization**: The best-performing model's predictions were visualized, showing segmented masks for a subset of the OxfordIIITPet validation set.

This project demonstrates the effectiveness of using pre-trained models and fine-tuning in improving semantic segmentation tasks across different datasets.
