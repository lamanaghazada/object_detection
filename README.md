# Object Detection with Yolov8n (Nano model) on custom dataset

## Overview:

This project focuses on object detection in urban environments.Autonomous system operating in urban environments using only one of 2 pretrained models and a custom dataset. Dataset coverage was designed to include diverse urban scenarios: daytime/nighttime, different weather conditions, crowded streets,.The model should accurately detect key urban objects,specifically cars,buses,motorbikes,bicycles.Photos were deliberately chosen to include multiple and diverse vehicles in the same image. This ensures the model learns to detect different object types simultaneously and handle scenarios where vehicles are overlapping or in close proximity, which is crucial for urban object detection.

## Setup

Follow the steps below.

### Clone the repository

git clone https://github.com/lamanaghazada/object_detection.git

### Install dependencies

pip install -r requirements.txt

### Additional Set up (in case any problem) 

### 1. Open the Notebook

Open the provided notebook in Google Colab and run the cells sequentially.

https://drive.google.com/drive/folders/1oKZxORvL57z_kCBRsrOTok1qZrwajV_C?usp=drive_link

### 2.Dataset

The dataset used in this project can be used from the following link:

https://drive.google.com/drive/folders/1pswcM2FMdZTOY_XRjb4zyk-cpRL9aPLS?usp=drive_link

### Installation

Clone the repository and install dependencies:

pip install -r requirements.txt

## Annotation process:

| Class     | Number of Annotations |
| --------- | --------------------- |
| Car       | 118                   |
| Bus       | 56                    |
| Motorbike | 57                    |
| Bicycle   | 69                    |
| **Total** | **300**               |

- Annotation done in CVAT about 7-8 hours;
- Some images had overlapping vehicles,special attention to them (cars behind buses or motorbikes beside bicycles), each was annotated separately with its own bounding box;
- Small motorbikes or bicycles near cars were annotated to ensure class distinction.

<img width="835" height="571" alt="Image" src="https://github.com/user-attachments/assets/d62bf6c1-e422-4163-bdaa-8ab5183c5167" />

<img width="1168" height="635" alt="Image" src="https://github.com/user-attachments/assets/383619c4-a591-4f9a-877c-04926e0cc06c" />

<img width="1262" height="588" alt="Image" src="https://github.com/user-attachments/assets/6e8946c1-4b06-4381-ac16-557e72832512" />

<img width="1207" height="640" alt="Image" src="https://github.com/user-attachments/assets/3b72d48f-c6fb-4158-b800-c5ce2d11745e" />

<img width="1124" height="497" alt="Image" src="https://github.com/user-attachments/assets/cc203385-2daa-4d4d-a18e-2b07286886eb" />

<img width="1278" height="579" alt="Image" src="https://github.com/user-attachments/assets/1bacfe93-39d3-4a24-be75-0d51df5157c6" />

The dataset annotations were exported in the Ultralytics YOLO Detection 1.0 format to ensure compatibility with the YOLOv8 training pipeline. The dataset configuration is defined in a YAML file, which specifies the paths to the train, validation, and test sets, as well as the class names used for object detection (car, bus, motorbike, bicycle).

### The annotated dataset exported from CVAT is available in the `dataset/` folder shared in Set Up.

## Model Training

Model Choice for Autonomous Driving

For urban autonomous driving scenarios, YOLOv8 is often the more practical choice.
Reasons that have been taken into consideration:
-Urban environments do not require extremely long-distance detection
-Most relevant objects are nearby (cars, buses, bicycles,motorbikes).
-Real-time performance is critical
-Faster inference improves reaction time. (3 times faster)
-Computational efficiency (2 times faster training)
Therefore, YOLOv8 provides a better speed–accuracy trade-off for city driving use cases, even though RT-DETR has **stronger transformer-based perception capabilities**.
RT-DETR Advantages:
-Better global scene understanding
-More robust for complex scenes
-Better detection of distant or small objects
For training I used 120 images,

## Data Augementation techniques:

Two types of augmentation were applied during training: custom augmentations and built-in YOLO augmentations. Ultralytics YOLOv8 already includes several built-in augmentation techniques. Based on the characteristics of the collected images: many photos in the dataset contain varying lighting conditions, strong contrast differences, and slight geometric variations, so augmentations were chosen to simulate these real-world conditions.
-A.Blur(blur_limit=7, p=0.5)
--Simulates camera motion or focus imperfections so the model can still detect objects in slightly blurred images.
-A.GaussNoise(std_range=(0.04, 0.2), p=0.3)
--Adds random sensor-like noise to help the model become robust to camera noise and low-quality images.
-A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5)
--Adjusts brightness and contrast to make the model robust to different lighting conditions such as shadows, strong sunlight, or overexposed photos.
-A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5)
--Randomly changes color properties so the model learns to recognize objects despite color variations caused by lighting or different cameras.
-Horizontal Flip (fliplr = 0.5)
--Many objects in the dataset can appear in either left or right orientation.
-Scaling (scale = 0.5)
--Objects may appear at different distances from the camera.
--Scaling augmentation helps the model learn to detect objects at various sizes.

## Hyperparameter Selection:

The hyperparameters were selected considering the small size of the dataset and the goal of achieving stable training while avoiding overfitting.

## Dataset Size Consideration

The dataset is relatively small:
Training set: 120 images (≈70%)
Validation set: 18 images (≈10%)
Test set: 34 images (≈20%)
Each class contains at least 30 annotations, which helps maintain a minimum level of class representation despite the limited number of images.
Because of this limited dataset size, several hyperparameters were adjusted to stabilize training and improve generalization.
Learning Rate — lr0 = 0.0001
Initially, the default learning rate (0.01) was tested.
The validation loss fluctuated significantly instead of decreasing steadily.
The learning rate was reduced to 0.0001, which stabilized training and produced better validation behavior.
Smaller datasets often require lower learning rates to prevent unstable parameter updates.

## Optimizer = 'AdamW'

The number of experiments conducted was limited, so using a reliable predefined optimizer was preferable.
In practice, Adam-based optimizers generally perform better than SGD when learning rate tuning is limited.
AdamW also provides better weight regularization compared to standard Adam.

## Layer Freezing (freeze=15)

When training on a small dataset, updating these layers can cause the model to overfit or destroy useful pretrained representations.
Training was first attempted without freezing any layers.This resulted in poorer performance, including lower F1-score, precision, and recall.
Then the first 15 layers of the backbone network were frozen.

## Early Stopping

(patience=10),
(patience=12)
After testing both values, the training behavior and final performance were nearly identical.

## Epochs

Training was initially started with 50 epochs as a baseline.Then 75 and 100.
It took approximately 30 minutes on a GPU environment.(considering early stopping).

## Model evaluation:

Although the overall results between the models were not significantly different, several experiments were conducted with different hyperparameter settings (learning rate, patience, freezing layers, etc.). The model trained for 100 epochs showed more stable and better performance based on metrics such as validation loss, training loss, and F1-score. Additionally, visual inspection indicated fewer incorrect predictions, so this model was selected as the final model.

<img width="2400" height="1200" alt="Image" src="https://github.com/user-attachments/assets/394fd09c-4f9a-443a-a643-72066c752f54" />

### Validation Set Results

| Class       | Images | Instances | Precision | Recall    | F1-score | mAP@0.5   | mAP@0.5:0.95 |
| ----------- | ------ | --------- | --------- | --------- | -------- | --------- | ------------ |
| car         | 6      | 8         | 0.930     | 0.750     | 0.830    | 0.880     | 0.638        |
| bus         | 5      | 5         | 0.944     | 1.000     | 0.971    | 0.995     | 0.848        |
| motorbike   | 5      | 7         | 1.000     | 0.856     | 0.922    | 0.995     | 0.680        |
| bicycle     | 4      | 6         | 0.745     | 1.000     | 0.854    | 0.972     | 0.789        |
| **Overall** | **18** | **26**    | **0.905** | **0.901** | —        | **0.961** | **0.739**    |

### Test Set Results

| Class       | Images | Instances | Precision | Recall    | mAP@0.5   | mAP@0.5:0.95 |
| ----------- | ------ | --------- | --------- | --------- | --------- | ------------ |
| car         | 16     | 27        | 0.821     | 0.704     | 0.809     | 0.471        |
| bus         | 10     | 12        | 0.951     | 0.833     | 0.893     | 0.762        |
| motorbike   | 8      | 11        | 0.832     | 0.900     | 0.894     | 0.667        |
| bicycle     | 9      | 12        | 0.857     | 0.833     | 0.896     | 0.597        |
| **Overall** | **34** | **62**    | **0.865** | **0.817** | **0.873** | **0.624**    |

### Inference Performance

| Metric                            | Value |
| --------------------------------- | ----- |
| Average inference time (ms/image) | 11.50 |

<img width="494" height="333" alt="Image" src="https://github.com/user-attachments/assets/9be46ec9-d7e0-4e26-a3a6-07df7f50fbfa" />

<img width="486" height="325" alt="Image" src="https://github.com/user-attachments/assets/2f717407-343b-4572-9a01-72e29cdfcb3e" />

### Error Analysis

Overall, the model performs well across all classes, but some common error patterns were observed.These results are based on a **small dataset**, so on a larger and more diverse dataset, the model is expected to achieve even better overall performance.

- **Small or partially visible objects:** Classes like **bicycle** and **motorbike** are sometimes missed when only a small portion of the object is visible or the object is very close to the camera.
- **Overlapping objects:** In cases where multiple objects overlap (e.g., a **motorbike** partially behind a **car**), the model occasionally detects only one of the overlapping objects.
- **Class confusion:** Occasionally,some images have been misclassified.
- **Occlusion and complex scenes:** Objects partially covered by others (e.g., a **car** behind a **bus**) can lead to incorrect or missed detections.

## Missed detection cases(also overlapping)

<img width="500" height="374" alt="Image" src="https://github.com/user-attachments/assets/d392cbad-b073-463a-a238-12584baddc92" />

<img width="496" height="245" alt="Image" src="https://github.com/user-attachments/assets/ef983baf-6a3a-40cd-9e9b-ef61216580de" />

### Use of LLM for Project Assistance

Throughout this project, ChatGPT(GPT-5 mini.), Claude were used as a supportive tool to streamline workflow and improve efficiency. The LLM helped in several ways:

- **Code assistance:** Provided quick solutions for repetitive tasks, such as **converting model metric values into Markdown tables for the README.**
- **Data handling guidance:** Offered advice on filtering and selecting relevant images from a large dataset (17,125 images), including which images were more suitable for training an urban object detection model.
- **General insights:** Gave recommendations on building an effective object detection model in urban environments, such as important considerations for camera placement, object sizes, occlusion, and dataset coverage.
- **Time management:** Helped prioritize tasks and reduce trial-and-error by providing quick, informed guidance for both coding and experimental decisions.

### Tools and Libraries

The following tools and libraries were used throughout the project:

- **YOLOv8n (Ultralytics):** Main object detection framework for model training and evaluation.
- **CVAT:** Used for annotating images and preparing the dataset for training.
- **Google Colab:** Development and experimentation environment with GPU support.
- **Python libraries:**
  - `torch` / `torchvision` – for deep learning and model backbone support
  - `opencv-python` – for image processing and visualization
  ### Challenges Faced
  Challenge was to select a **small subset** from **17,125 images** with only **four specific types of images** . Initially, I started manually selecting images without considering other folders, which was **time-consuming**. Later, I noticed that the **annotation folder** contained labels for each image, where images were assigned values of -1 or 1. Using this information, I was able to **extract only the images labeled as 1** from the four relevant `.txt` files via a **Python script**, which significantly **saved time** and improved workflow efficiency.

### Future Improvements

- **Larger and more diverse dataset:** Expanding the dataset with more urban scenes and varied object instances would likely improve overall accuracy and generalization.
- **Hyperparameter tuning:** More extensive experimentation with learning rate schedules,optimizer selection, and layer freezing strategies could further optimize performance.
- **Evaluation on real-world footage:** Testing the model on video streams from urban cameras to assess detection in dynamic, overlapping, or occluded scenarios.
