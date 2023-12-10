
# ID2223 - Lab2 - Fine-Tune a Transformer For Language Transcription to Chinese

There are two tasks in this lab. The first step is to fine-tune a model for language transcription, add a UI. The second task is to improve pipeline scalability and model performance. The data is from https://huggingface.co/datasets/mozilla-foundation/common_voice_11_0/viewer/zh-CN


<!-- PROJECT SHIELDS -->
<!-- @import "[TOC]" {cmd="toc" depthFrom=1 depthTo=6 orderedList=false} -->
<!-- PROJECT LOGO -->
<br />

<p align="center">
  <a href="https://github.com/Tongm56/ID2223/tree/main/Lab2">
    <img src="https://www.thetype.com/wp-content/uploads/2013/05/renzheng.jpg" alt="Logo" width="300" height="300">
  </a>

  <h3 align="center">ID2223 - Lab2</h3>
  <p align="center">
    fine tune whisper-small for Chinese
    <br />
    <a href="https://github.com/Tongm56/ID2223/tree/main/Lab2"><strong>feel free to try »</strong></a>
    <br />
    <br />
    <a href="https://huggingface.co/spaces/momowanwu/whisper-small-zh-CN">whisper-small-zh-CN</a>
    ·
    <a href="https://huggingface.co/momowanwu/checkpoint2">Best Model</a>
    ·
    <a href="https://huggingface.co/momowanwu/checkpoint">Model in Task1</a>
    ·
    <a href="https://github.com/Tongm56/ID2223/issues">report Bug</a>
    ·
    <a href="https://github.com/Tongm56/ID2223/issues">propose new features</a>
  </p>

</p>

This README.md is for developers who want to try to quickly fine tune whisper-small model for Chinese.

<span style="color:red">**Just click the link below the picture to see our demo.**</span>
 
## Table of contents

- [Tasks](#tasks)
  - [Requirements](#requirements)
  - [Install](#install)
- [What our Gradio app can do](#what-our-gradio-ap-can-do)
  - [Transcribe speech via microphone or upload a file](#transcribe-speech-via-microphone-or-upload-a-file)
  - [Transcription via URL](#transcription-via-URL)
  - [Input text and output Chinese](#input-text-and-output-chinese)
- [Answers to tasks](#answers-to-tasks)
- [Authors](#authors)

## Tasks
- a. Fine-Tune a pre-trained transformer model and build a serverless UI for using that model
- b. Communicate the value of your model to stakeholders with an app/service that uses the ML model to make value-added decisions
- c. Refactor the program into a feature engineering pipeline, training pipeline, and an inference program (Hugging Face Space) to enable you to run feature engineering on CPUs and the training pipeline on GPUs. You should save checkpoints when training, so that you can resume again from the checkpoint.
- d. Describe in your README.md program ways in which you can improve model performance are using
(1) model-centric approach - e.g., tune hyperparameters, change the fine-tuning model architecture, etc
(2) data-centric approach - identify new data sources that enable you to train a better model that one provided in the blog post

For more information, please see this document below. 
(https://id2223kth.github.io/assignments/lab2/id2223_kth_lab2_2023.pdf)

### Requirements
- yt-dlp
- gradio
-  moviepy
- transformers
- torch
- gtts
### **install**
Clone the repo

```sh
git clone https://github.com/Tongm56/ID2223/tree/main/Lab2.git
```
### What our Gradio app can do
#### Transcribe speech via microphone or upload a file

[![Transcribe speech via microphone or upload a file](https://github.com/Tongm56/ID2223/blob/main/Lab2/pictures/Record%20your%20speech%20or%20upload%20an%20audio%20file.png)](https://github.com/Tongm56/ID2223/blob/main/Lab2/pictures/Record%20your%20speech%20or%20upload%20an%20audio%20file.png)

#### Transcription via URL
<span>**Users can paste Bilibili or YouTube videos for Chinese voice transcription. Note that the whisper-small model can only read the first 30 seconds of content. There are currently no tests on other video platforms.**</span>

[![Transcription via URL](https://github.com/Tongm56/ID2223/blob/main/Lab2/pictures/Transcribe%20from%20URL.png)](https://github.com/Tongm56/ID2223/blob/main/Lab2/pictures/Transcribe%20from%20URL.png)

#### Input text and output Chinese
<span>**Using Google's gtts API, you can convert Chinese text into speech to facilitate learning and imitation.**</span>
[![Input text and output Chinese](https://github.com/Tongm56/ID2223/blob/main/Lab2/pictures/Text%20to%20Speech%20Synthesis.png)](https://github.com/Tongm56/ID2223/blob/main/Lab2/pictures/Text%20to%20Speech%20Synthesis.png)

### Answers to tasks
- a. Fine-Tune a pre-trained transformer model and build a serverless UI for using that model

I keep checkpoints in both Google Cloud Drive and hugging face hub to facilitate training and loading.

When maintaining the model parameters in the Hindi blog (see references for details), the WER is 190%, which is obviously not a satisfactory result.
- b. Communicate the value of your model to stakeholders with an app/service that uses the ML model to make value-added decisions

The value of models is shown in the previous section, "What our Gradio application can do." This can greatly improve text recording efficiency when transcribing Chinese via microphone or file. Pasting the URL is not only convenient for text extraction of videos, but also some podcasts and other applications that do not have text records can easily form text records. I also implemented the functions of inputting Chinese characters and outputting audio through Google API, which can make it easier for people to learn Chinese. But it should be pointed out that this model is only a whisper-small tweak and mainly handles audio within 30 seconds. If it takes longer, consider a model like Whisper-large-v2. But the principle is the same, these models require more powerful GPUs. In this experiment, I mainly used V100 and A100 GPUs.

- c. Refactor the program into a feature engineering pipeline, training pipeline, and an inference program (Hugging Face Space) to enable you to run feature engineering on CPUs and the training pipeline on GPUs. You should save checkpoints when training, so that you can resume again from the checkpoint.

I divided the code into three main parts. The first part is feature extraction. This file is: https://github.com/Tongm56/ID2223/blob/main/Lab2/feature_pipeline_whisper_small_zh.ipynb. The CPU can be used to process this file, which mainly includes file downloading, file feature processing, and uploading to the cloud disk. The second part is:https://github.com/Tongm56/ID2223/blob/main/Lab2/training_pipeline_whisper_small_zh.ipynb. In this file, the parameters of model training are mainly adjusted, and appropriate checkpoints are saved to Google Cloud Disk and hugging face hub. The third file builds the UI of gradient: https://github.com/Tongm56/ID2223/blob/main/Lab2/Build%20a%20demo%20UI%20on%20hugging%20face.ipynb. It should be noted that this file only creates a temporary demo in colab. If you need to create a space in hugging face, you need to remove the first line in this file and name it app.py and name the dependent package as requirements.txt file and then upload both files in the gradio app.

- d. Describe in your README.md program ways in which you can improve model performance are using
(1) model-centric approach - e.g., tune hyperparameters, change the fine-tuning model architecture, etc
(2) data-centric approach - identify new data sources that enable you to train a better model that one provided in the blog post

In my model, by adjusting gradient_accumulation_step to 2, I reduced the WER of my model from 190% to 148.5%. Although this is still not a very good WER, we tested it and found that the recognition rate is very high. Compared with some of my models with WER around 100%, this model has a better ability to recognize Chinese in real case e.g. paste a Chinese news video URL or speak by my self, so I decided to use this model. It is worth noting that generally Chinese can be evaluated with CER, but I don’t have enough time, so there is no way to use CER for evaluation.

The reason for the poor results may be that some of the test data sets itself are very unclear, making it difficult for even the human ear to distinguish. At the same time, the epochs I ran were not enough. Maybe more epochs are needed to learn the features. Another reason is that Chinese itself is very complex. I have not found a Chinese whisper-small benchmark, but I have found that some can only achieve 100% WER using whisper-large-v2.

[![model parameters](https://github.com/Tongm56/ID2223/blob/main/Lab2/pictures/model%20para.PNG
)](https://github.com/Tongm56/ID2223/blob/main/Lab2/pictures/model%20para.PNG
)

In terms of model-centric approach, we can consider using larger models such as whisper-medium or large-v2 models. If we focus on this whisper-small model, we can consider adjusting the learning rate and batch size to obtain different epochs, and we can also consider using various methods to prevent overfitting e.g.use Dropout layers or apply regularization. If our GPU is powerful enough, we can use grid search to compare different hyperparameter combinations.

For data-centric approach. We can consider using the AISHELL data set. This is a Chinese speech data set released by Beijing Hill Company, which contains about 178 hours of open source version data. The data set contains the voices of 400 people with different accents from different regions in China.

Future improvements can consider both aspects at the same time and use more data sets for training and testing. Perhaps another testing method is to conduct WER or CER test in real life by inputting a large number of videos.

### Authors
- [TongMo](https://github.com/Tongm56)
- [BokuanLi](https://github.com/bokuan)

### License

 [LICENSE.txt](https://github.com/Tongm56/ID2223/blob/main/LICENSE)

### Reference

- [lab requirements are from ID2223](https://id2223kth.github.io/assignments/lab2/id2223_kth_lab2_2023.pdf)
- [Hugging face Pages](https://huggingface.co/blog/fine-tune-whisper)




