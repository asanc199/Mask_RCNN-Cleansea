# Cleansea Project
## Mask R-CNN for Underwater Debris Detection and Segmentation
This project uses the Mask R-CNN algorithm to detect underwater debris. The goal is to train the Mask R-CNN neural network algorithm using our hand-labeled dataset of underwater debris in order to achieve a successful detection of segmentation of debris under the sea.

The main purpose of this research consists:
* **New Dataset generation:** Since most existing datasets for object detection are not applicable in this type of task, a dataset of underwater captured debris has been labeled with up to 19 different labels.

* **Trainning and fine-tunning Mask R-CNN model:** Once we have the data needed for this task, a fine-tunning process is needed for achieving good results-

* **Image, Video and Real Time Detection:** As an evaluation phase, we test our trained model analicing different images, videos and real time detections using a simulated enviroment.

## Cleansea Dataset Image examples
Hand-labeled dataset of underwater debris is shown below:

![Debris Detection Sample 1](/assets/detection_0.png)
![Debris Detection Sample 2](/assets/detection_1.png)

## Citation
Use this bibtex to cite this repository:
```
@inproceedings{asferrer2022,
      author    = {Alejandro Sanchez Ferrer, Antonio Javier Gallego, Jose Javier Valero-Mas, and Jorge Calvo-Zaragoza},
      title     = {Underwater Debris Detection using Regression Neural Networks},
      booktitle = {Iberian Conference on Pattern Recognition and Image Analysis},
      year      = {2022}
    }
```

## Requirements
Python 3.7.11, TensorFlow 2.4.1, Keras 2.4.3 and other common packages listed in `requirements.txt`.

## Installation
Follow the `installation/cleansea_installation.md` file for further instructions on installing dependencies on a conda enviroment.

(Optional) To train or test on MS COCO install `pycocotools` from one of these repos. They are forks of the original pycocotools with fixes for Python3 and Windows (the official repo doesn't seem to be active anymore).

    * Linux: https://github.com/waleedka/coco
    * Windows: https://github.com/philferriere/cocoapi.
    You must have the Visual C++ 2015 build tools on your path (see the repo for additional details)

# Example of usage
### [Cleansea Project](https://youtu.be/nQbFYz0dRno) - Underwater Debris Detection as Bachelor's Final Project by [Alejandro Sanchez Ferrer](https://github.com/asanc199)
![Projecto Cleansea](assets/project_cleansea.gif)
