# FastMONAI

### Data - The Medical Decathlon
The Medical Decathlon is a benchmark dataset for evaluating AI algorithms for medical image segmentation. The dataset includes 10 different medical imaging tasks, each with its own training and testing sets. The tasks include brain tumor segmentation, liver segmentation, and heart segmentation, among others.

The dataset contains 60 patients, with 10 patients per task. Each patient has multiple MRI sequences, and each sequence has a corresponding ground-truth segmentation. The images are provided in NIfTI format, and the ground-truth segmentations are provided as binary masks in NIfTI format as well.

The dataset is diverse and challenging, with variations in image quality, resolution, and pathology across different patients and imaging tasks. This makes it an ideal benchmark dataset for evaluating the performance of AI algorithms for medical image segmentation.

### What is FastMONAI


### The goal
The goal of the proposed idea is to develop an AI-powered system that can analyze MRI scans in real-time and automatically identify any tumors present. The system would then send this information to the doctor who is responsible for reviewing the scan, providing an objective and automated assessment of the scan before the doctor has had a chance to review it themselves.

The primary aim of this system is to improve the speed and accuracy of medical diagnosis, potentially saving valuable time and identifying tumors or other abnormalities that may have been missed by human review alone. By providing doctors with an objective assessment of the MRI scans, the system could also help to reduce the risk of human error and ensure that patients receive prompt and accurate diagnoses.



## Results

### Notebooks

#### Nodules (MedMNIST) 

#### Brain tumor (Decathlon)
In the brain tumor dataset we did not use the existing method DecathlonDataset() from FastMONAI for downloading the datasets due to not being aware of this method existing. Instead we accessed the dataset through Google Drive, giving us the chance to explore the content of the dataset, and decide how best to make a data frame for ourselves. It would definitely have been less time-consuming to use FastMONAI for this part of the project, but we got a good insight on how the dataset was constructed this way. 

We used MedDataset from FastMONAI to give us a great overview of the size and shape of the images. We found that there was no need to resample or reorder the images. MedDataBlock() was a great help when making a dataloader. We used MONAI’s model UNet() to make a model. There was some difficulty with the spatial dimension of the UNet() model since it did not take any higher than 3 dimensions.

We used FastMONAI both for making the lossfunction and the learner.

Using the lr_find() method we found that the optimal learning rate was somewhere between 0.01 and 0.1

We got a binary dice score of 0.9742, which seems too high even though the model made several correct predictions

#### Heart (Decathlon)
<a target="_blank" href="https://colab.research.google.com/github/MatiasHolmemoMardal/FastMONAI/blob/main/notebooks/heart_semantic.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

We used the DecathlonDataset() method from FastMONAI for downloading the dataset. 

Got binary dice score of 0.9110 which we are happy with

#### Colon (Decathlon)
We used the DecathlonDataset() method from FastMONAI for downloading the dataset. 

This model was made with the same structure as the heart-model, but did not perform as expected. When training the pipeline we got “nan” on the dice score and tried to fix this several times and didn’t manage to fix this before there was no time left. This problem also occurred when working on the lung-model. We tried several times to fix this on this dataset and the lung-dataset with no luck. We tried to use Google and ChatGPT, but none of the suggestions we got helped us even with hours upon hours trying to fix the problem.

#### Lung (Decathlon)

