# FastMONAI Course Project - Group 4
Sander Braastad, Tarald Skaar, Matias Holmemo Mardal

## Context
Deep learning has shown a lot of promise in the field of medical image analysis, and many researchers are exploring its potential in areas such as disease diagnosis and treatment planning. However, developing deep learning models for medical imaging can be challenging due to the complexity of the data and the need for accurate and reliable results.

FastMONAI is a deep learning library that provides a high-level interface for developing and training deep learning models for medical imaging tasks. It is built on top of PyTorch and provides a range of features for processing medical images, including data loading, preprocessing, and visualization. In this assignment we will focus on the medMNIST and decathlon datasets, which we will go through thoroughly in the data section of this paper. 

Overall, working on a deep learning project using fastMONAI requires a strong understanding of deep learning concepts and techniques, as well as expertise in data preprocessing and analysis. It is a challenging but rewarding area of research that has the potential to make a real impact on the field of medical imaging. 


## Data

### MedMNIST
MedMNIST had several datasets containing 28x28x28 images. Advantages with this dataset were equally formatted images and big datasets. The images are small which makes them easier to process, but that also means they contain less information. MedMINST’s dataset’s contained several different types of problems to solve. The group started experimenting with this dataset, but had huge difficulties converting the images to NIfTI and using FastMONAI on those images. None of the models from this dataset did well

### The Medical Decathlon
The Medical Decathlon is a benchmark dataset for evaluating AI algorithms for medical image segmentation. The dataset includes 10 different medical imaging tasks, each with its own training and testing sets. The tasks include brain tumor segmentation, liver segmentation, and heart segmentation, among others.

The dataset contains 60 patients, with 10 patients per task. Each patient has multiple MRI sequences, and each sequence has a corresponding ground-truth segmentation. The images are provided in NIfTI format, and the ground-truth segmentations are provided as binary masks in NIfTI format as well.

The dataset is diverse and challenging, with variations in image quality, resolution, and pathology across different patients and imaging tasks. This makes it an ideal benchmark dataset for evaluating the performance of AI algorithms for medical image segmentation.

Decathlon had several datasets great for learning. All the images are in a compressed NIfTI format (nii.gz) making it easy to use FastMONAI to explore the dataset’s. Decathlon only has datasets made for semantic segmentation, but with different challenges within each dataset. The group found it easier to work with.

Consists of 10 datasets:
1. BraTS: A dataset of brain MRI scans for glioma tumor segmentation.
2. CANDI: A dataset of cardiac MR images for left ventricle segmentation.
3. CDMRI: A dataset of cardiac MRI scans for cardiac function quantification.
4. CREMIS: A dataset of chest CT scans for lung nodule detection.
5. CQ500: A dataset of chest CT scans for lung nodule classification.
6. ISIC: A dataset of skin lesion images for melanoma classification.
7. KiTS: A dataset of kidney tumor CT scans for tumor segmentation.
8. LiTS: A dataset of liver CT scans for liver and lesion segmentation.
9. PANDA: A dataset of prostate biopsy images for cancer diagnosis.
10. RibFrac: A dataset of chest CT scans for rib fracture detection.


## What is FastMONAI


## The goal
The goal of the proposed idea is to develop an AI-powered system that can analyze MRI scans in real-time and automatically identify any tumors present. The system would then send this information to the doctor who is responsible for reviewing the scan, providing an objective and automated assessment of the scan before the doctor has had a chance to review it themselves.

The primary aim of this system is to improve the speed and accuracy of medical diagnosis, potentially saving valuable time and identifying tumors or other abnormalities that may have been missed by human review alone. By providing doctors with an objective assessment of the MRI scans, the system could also help to reduce the risk of human error and ensure that patients receive prompt and accurate diagnoses.



## Results / Notebooks

### Nodules (MedMNIST) 

<a target="_blank" href="https://colab.research.google.com/github/MatiasHolmemoMardal/FastMONAI/blob/main/notebooks/V2_FastMONAI_MedMNIST_Single_label_classification.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

### Brain tumor (Decathlon)

<a target="_blank" href="https://colab.research.google.com/github/MatiasHolmemoMardal/FastMONAI/blob/main/notebooks/Braintumor_model.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
In the brain tumor dataset we did not use the existing method DecathlonDataset() from FastMONAI for downloading the datasets due to not being aware of this method existing. Instead we accessed the dataset through Google Drive, giving us the chance to explore the content of the dataset, and decide how best to make a data frame for ourselves. It would definitely have been less time-consuming to use FastMONAI for this part of the project, but we got a good insight on how the dataset was constructed this way. 

We used MedDataset from FastMONAI to give us a great overview of the size and shape of the images. We found that there was no need to resample or reorder the images. MedDataBlock() was a great help when making a dataloader. We used MONAI’s model UNet() to make a model. There was some difficulty with the spatial dimension of the UNet() model since it did not take any higher than 3 dimensions.

We used FastMONAI both for making the lossfunction and the learner.

Using the lr_find() method we found that the optimal learning rate was somewhere between 0.01 and 0.1

We got a binary dice score of 0.9742, which seems too high even though the model made several correct predictions

### Heart (Decathlon)

<a target="_blank" href="https://colab.research.google.com/github/MatiasHolmemoMardal/FastMONAI/blob/main/notebooks/heart_semantic.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

We used the DecathlonDataset() method from FastMONAI for downloading the dataset. 

Got binary dice score of 0.9110 which we are happy with

### Colon (Decathlon)

<a target="_blank" href="https://colab.research.google.com/github/MatiasHolmemoMardal/FastMONAI/blob/main/notebooks/colon_b_s_semantic.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

We used the DecathlonDataset() method from FastMONAI for downloading the dataset. 

This model was made with the same structure as the heart-model, but did not perform as expected. When training the pipeline we got “nan” on the dice score and tried to fix this several times and didn’t manage to fix this before there was no time left. This problem also occurred when working on the lung-model. We tried several times to fix this on this dataset and the lung-dataset with no luck. We tried to use Google and ChatGPT, but none of the suggestions we got helped us even with hours upon hours trying to fix the problem.

### Lung (Decathlon)

### Spleen (Decathlon) 

<a target="_blank" href="https://colab.research.google.com/github/MatiasHolmemoMardal/FastMONAI/blob/main/notebooks/Spleen_semantic.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>
In the spleen model we had the same problem as with the colon model

## What we could have done differently
In hindsight, there are several aspects of our project that we could have approached differently to improve our outcomes. Firstly, we should have established stricter schedules and worked more evenly throughout the project. As a group, we were also writing our bachelor thesis at the same time, which took up a significant amount of our time and energy. This made it challenging to allocate enough time to the project and contributed to some delays in our progress.

Another area where we could have improved is in our initial choice of dataset. We started with the MedMNIST dataset, which presented some compatibility issues with FastMONAI, the deep learning library we were using for our project. As a result, we spent a considerable amount of time on trial and error, attempting to work with this dataset. Ultimately, we switched to the Medical Decathlon dataset, which was much more compatible with FastMONAI and allowed us to make faster progress. If we had started with the Medical Decathlon dataset from the outset, we could have saved a lot of time and completed the Gradio app.

Additionally, we could have made better use of our teacher, who was the creator of the FastMONAI library. His expertise would have been invaluable in helping us troubleshoot issues and optimize our models more efficiently. By seeking his guidance more proactively, we might have been able to resolve the compatibility issues with the MedMNIST dataset more quickly or determine sooner that we should switch to the Medical Decathlon dataset.

Speaking of the Gradio app, one significant mistake we made was starting its development too late. We should have started developing the app earlier in the project to ensure that it was completed by the end of our timeline. As it stands, we were unable to complete the app due to time constraints.

Overall, if we had established stricter schedules, started with the Medical Decathlon dataset from the outset, consulted our teacher more frequently for assistance, and begun development of the Gradio app earlier in the project, we could have achieved more successful outcomes. This serves as a valuable lesson for future projects, emphasizing the importance of careful planning and execution to ensure the best possible results.


## Further work

One potential direction for future work is to implement the deep learning models developed in this project into a user-friendly web application using Gradio. The Gradio app could allow users to select a binary semantic segmentation model that represents the type of MRI scan (e.g., heart, brain, lung, etc.), and then upload an MRI scan to obtain an output image that marks the location of any tumors present.

By providing users with an intuitive and easy-to-use interface for running these deep learning models, the Gradio app could help to facilitate the use of AI-powered medical imaging analysis by clinicians and researchers who may not have extensive experience with deep learning. This could ultimately help to improve the accuracy and speed of medical diagnoses, potentially leading to better patient outcomes.

To implement this further work, additional development would be required to integrate the deep learning models into the Gradio app and to ensure that the app is user-friendly and accessible to a wide range of users. Additionally, further testing and validation would be needed to ensure that the app is accurate and reliable for a variety of different MRI scans and clinical scenarios.


