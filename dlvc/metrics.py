import torch
from abc import ABCMeta, abstractmethod

class PerformanceMeasure(metaclass=ABCMeta):
    '''
    A performance measure.
    '''

    @abstractmethod
    def reset(self):
        '''
        Resets internal state.
        '''
        pass

    @abstractmethod
    def update(self, prediction: torch.Tensor, target: torch.Tensor):
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        Raises ValueError if the data shape or values are unsupported.
        '''
        pass

    @abstractmethod
    def __str__(self) -> str:
        '''
        Return a string representation of the performance.
        '''
        pass


class SegMetrics(PerformanceMeasure):
    '''
    Mean Intersection over Union.
    '''

    def __init__(self, classes):
        self.classes = classes
        self.confusion_matrix = torch.zeros(len(classes), len(classes))
        self.reset()

    def reset(self) -> None:
        '''
        Resets the internal state.
        '''
        self.confusion_matrix.zero_()

    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        '''
        Update the measure by comparing predicted data with ground-truth target data.
        prediction must have shape (b,c,h,w) where b=batchsize, c=num_classes, h=height, w=width.
        target must have shape (b,h,w) and values between 0 and c-1 (true class labels).
        Raises ValueError if the data shape or values are unsupported.
        Make sure to not include pixels of value 255 in the calculation since those are to be ignored. 
        '''
        if len(prediction.shape) != 4 or len(target.shape) != 3:
            raise ValueError("Unsupported data shape. Expected prediction shape (b,c,h,w) and target shape (b,h,w).")

        if prediction.shape[1] != len(self.classes):
            raise ValueError(f"Number of classes in prediction ({prediction.shape[1]}) does not match with provided classes ({len(self.classes)}).")

        if not torch.all(torch.eq(target, torch.floor(target))) or torch.any(target < 0) or torch.any(target >= len(self.classes)):
            raise ValueError("Unsupported target values. Expected values between 0 and c-1 (true class labels).")

        mask = target != 255
        ###########
        #target_out = target[mask]
        #if not (torch.all(target_out >= 0) and torch.all(target_out < len(self.classes-1))):    
        #    raise ValueError("Unsupported target values. Expected values between 0 and c-1 (true class labels).")
        ###################################
        prediction = torch.argmax(prediction, dim=1)

        for cls in range(len(self.classes)):

            TP = torch.sum((prediction[mask] == cls) & (target[mask] == cls)).item()
            FP = torch.sum((prediction[mask] == cls) & (target[mask] != cls)).item()
            FN = torch.sum((prediction[mask] != cls) & (target[mask] == cls)).item()
            self.confusion_matrix[cls, cls] += TP
            self.confusion_matrix[cls, :] += FP
            self.confusion_matrix[:, cls] += FN

    def __str__(self):
        '''
        Return a string representation of the performance, mean IoU.
        e.g. "mIou: 0.54"
        '''
        return f"mIoU: {self.mIoU():.4f}"#\nConfusion Matrix:\n{self.confusion_matrix}

    def mIoU(self) -> float:
        '''
        Compute and return the mean IoU as a float between 0 and 1.
        Returns 0 if no data is available (after resets).
        If the denominator for IoU calculation for one of the classes is 0,
        use 0 as IoU for this class.
        '''
        intersection = torch.diag(self.confusion_matrix)
        union = torch.sum(self.confusion_matrix, dim=0) + torch.sum(self.confusion_matrix, dim=1) - intersection
        iou = intersection / union
        #Replace Na values with 0
        iou[torch.isnan(iou)] = 0
        mean_iou = torch.mean(iou)
        return mean_iou.item()
    
    def get_confusion_matrix(self):
        '''
        Return the confusion matrix.
        '''
        #return self.confusion_matrix
        return f"Confusion Matrix:\n{self.confusion_matrix.numpy()}"
    

'''
# Example usage:
# Initialize SegMetrics with classes
classes = ["A", "B", "C"]
seg_metrics = SegMetrics(classes)

# Update with prediction and target tensors
#prediction = torch.tensor([[[[0.5, 0.3, 0.2], [0.1, 0.8, 0.1]], [[0.2, 0.5, 0.3], [0.4, 0.3, 0.3]], [[0.3, 0.2, 0.5], [0.1, 0.2, 0.7]]]])
#target = torch.tensor([[0, 1], [2, 1]])

# Example tensors with correct shapes
prediction = torch.randn(2, 3, 4, 4)  # 2 samples, 3 classes, 4x4 images
target = torch.randint(0, 3, (2, 4, 4))  # 2 samples, 4x4 images with class labels 0, 1, 2

seg_metrics.update(prediction, target)
print("Prediction Tensor:")
print(prediction)

print("\nTarget Tensor:")
print(target)
# Print performance (mean IoU) and confusion matrix
print(seg_metrics)
print(seg_metrics.get_confusion_matrix())#check1
#okay :))

#################check2
import torch

# Define the classes
classes = ["A", "B", "C"]

# Initialize SegMetrics
seg_metrics = SegMetrics(classes)

# Example tensors with correct shapes
#prediction = torch.zeros(3, 3, 4, 4)  # All predictions are initialized to zero
prediction = torch.zeros(3 ,3, 4, 4, dtype=torch.long)  # All target labels are initialized to zero

target = torch.tensor([[[0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0]],

                       [[1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1],
                        [1, 1, 1, 1]],

                       [[2, 2, 2, 2],
                        [2, 2, 2, 2],
                        [2, 2, 2, 2],
                        [2, 2, 2, 2]]], dtype=torch.long)  # Ground truth labels

# Assigning labels to target tensor
#prediction[:, 0, :, :] = 0  # First sample's labels
#prediction[:, 1, :, :] = 1  # Second sample's labels
#prediction[:, 2, :, :] = 2  # Third sample's labels

# Assigning labels to prediction tensor
#for i in range(3):  # Adjusted loop for 3 samples
#    for j in range(4):
#        for k in range(4):
#            prediction[i, target[i, j, k], j, k] = 1.0  # Assigning 1.0 to the predicted class label
for batch_idx in range(target.shape[0]):
    for i in range(target.shape[1]):
        for j in range(target.shape[2]):
            class_label = target[batch_idx, i, j]
            prediction[batch_idx, class_label, i, j] = 1.0

print("Prediction Tensor:")
print(prediction)

print("\nTarget Tensor:")
print(target)

# Update SegMetrics
seg_metrics.update(prediction, target)

# Print performance (mean IoU) and confusion matrix
print(seg_metrics)
print(seg_metrics.get_confusion_matrix())


'''