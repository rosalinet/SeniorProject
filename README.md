# Fall 2021 - Senior Project
by Jared Baumann, Rosaline Tep, Mina Wahib

### Abstract
This project serves to implement neural network techniques for inferring secondary structure
classes from data about the Cα backbone of an amino acid. It’s purpose is not to determine the
entire protein structure, but instead serve as a stepping stone for these efforts. This system is
implemented such that it will be easy to utilize and provide fast and accurate predictions about
secondary structures. In order to accomplish this task, we've implemented multiple networks that
can be used to predict structures, including Fully Connected Neural Networks (FCNNs) and Deep
Convolutional Neural Networks (DCNNs) to make predictions both for the secondary structure
classification of a single amino acid through its Cα characteristics as well as the transition type if
applicable. For both cases we were able to exceed 80% accuracy, with the simple secondary
structure classification reaching an accuracy of 93%, and the transition type reaching an accuracy
of 84%.

*The neural network is within the Python-constructed GUI. Some packages used in this project are tkinter, numpy, tensorflow, and pandas; within the report, you can also see results of PCA, ROC, and the confusion matrices.*
