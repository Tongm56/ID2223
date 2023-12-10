
# ID2223 - Lab1 - Iris & Wine

There are two parts in this lab. The first step is to run the Feature, Training (KNN), Online/Batch Inference Pipelines and build a Serverless ML system for Iris Flower dataset. Another part is to do the similar work for the Wine Quality Dataset(https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/wine.csv). In this case, we use Random Forests to classify the wine dataset into two main categories, average quality and high quality. Developers can easily develop more types of classifiers or different models based on this project. 

The focus of our lab is to learn and demonstrate how to build a serverless machine learning system, emphasizing the importance of scalability for both code and ideas.

<!-- PROJECT SHIELDS -->
<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->
<!-- PROJECT LOGO -->
<br />

<p align="center">
  <a href="https://github.com/Tongm56/ID2223/tree/main/Lab1">
    <img src="https://raw.githubusercontent.com/bokuan/ID2223/main/average.png?token=GHSAT0AAAAAACHL4DAOO2XQVNTSNYIPH476ZKY6VSQ" alt="Logo" width="300" height="300">
  </a>

  <h3 align="center">ID2223 - Lab1</h3>
  <p align="center">
    Serverless ML System
    <br />
    <a href="https://github.com/Tongm56/ID2223/tree/main/Lab1"><strong>feel free to try »</strong></a>
    <br />
    <br />
    <a href="https://huggingface.co/spaces/momowanwu/iris">Iris_Prediction</a>
    ·
    <a href="https://huggingface.co/spaces/momowanwu/wine_quality_pre">Wine_Prediction</a>
    ·
    <a href="https://huggingface.co/spaces/momowanwu/iris_monitor">Iris_Prediction_Monitor</a>
    ·
    <a href="https://huggingface.co/spaces/momowanwu/wine_quality_monitor">Wine_Prediction_Monitor</a>
    ·
    <a href="https://github.com/Tongm56/ID2223/issues">report Bug</a>
    ·
    <a href="https://github.com/Tongm56/ID2223/issues">propose new features</a>
  </p>

</p>

This README.md is for developers who want to try to quickly develop machine learning model user interfaces online instead of just using ipynb for classification or regression.
<span style="color:red">**Just click the link below the picture to see our demo.**</span>
 
## Table of contents

- [Background](#Background)
  - [Requirements](#Requirements)
  - [install](#install)
- [Structure](#Structure)
  - [Iris](#Iris)
  - [WineQuality](#WineQuality)
- [Authors](#Authors)

## Background
Usually, ordinary machine learning process is always based on local, such as directly using .ipynb for model data processing, training and prediction. Our lab this time used several advanced websites to build two serverless machine learning systems. They are hopsworks.ai, modal.com and huggingface.com

Overall, the functions of the three different websites are as follows: 

Tasks
a. Build and run a feature pipeline on Modal
b. Run a training pipeline
c. Build and run an inference pipeline with a Gradio UI on Hugging Face Spaces.

For more information, please see this document below. This document introduces all websites and processes that require registration and the structure of the whole serverless ML system. 

(https://id2223kth.github.io/assignments/lab1/id2223_kth_lab1_2023.pdf)

### Requirements
1. If you have windows, install twofish
2. hopsworks
3. joblib
4. scikit-learn==1.1.1
5. seaborn
6. dataframe-image
7. modal
8. gradio

### **install**
Clone the repo

```sh
git clone https://github.com/Tongm56/ID2223/tree/main/Lab1.git
```
### Structure
#### Iris 

[![Iris](https://github.com/Tongm56/ID2223/blob/main/Lab1/Iris.png)](https://github.com/Tongm56/ID2223/blob/main/Lab1/Iris.png)

#### Wine Quality 

[![WineQuality](https://github.com/Tongm56/ID2223/blob/main/Lab1/wine.png)](https://github.com/Tongm56/ID2223/blob/main/Lab1/wine.png)

### Authors
- [TongMo](https://github.com/Tongm56)
- [BokuanLi](https://github.com/bokuan)

### Details for wine quality dataset
| quality | count |
|---------|-------|
| 6       | 2311  |
| 5       | 1745  |
| 7       | 852   |
| 4       | 204   |
| 8       | 148   |
| 3       | 30    |
| 9       | 5     |

After removing duplicates and null values, we get the different qualities of wine, ranging from 3 to 9 respectively. When directly divided into 7 categories and do classification, we found that the model classification performance was very poor, mainly predicting 5, 6, and 7 with the largest amount of data. The reason for this result is that we found that this data set is very unbalanced, which may cause overfitting. 

Therefore, we use a classification method for the wine quality dataset and use the random forest model to classify the quality of 3, 4, and 5 as average quality, and the quality of 6, 7, 8, and 9 as high quality. In this case, the two types of labels are basically balanced.

Of course, developers can be divided into more categories, such as 3, 4, and 5 are low quality, 6 is medium quality, and 7, 8, and 9 are high quality, depending on the quality of the data and their ideas. In any case, the principles and ideas are the same, and only a small part of the code needs to be changed.
### License

 [LICENSE.txt](https://github.com/Tongm56/ID2223/blob/main/LICENSE)

### Reference

- [lab requirements are from ID2223](https://id2223kth.github.io/assignments/lab1/id2223_kth_lab1_2023.pdf)
- [GitHub Pages](https://pages.github.com)




