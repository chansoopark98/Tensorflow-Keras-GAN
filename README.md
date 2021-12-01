<h1>CSNet-colorization</h1>

> CSnet colorization: based on Deep Learning  

CNN(Convolutional Neural Network) 기반의 colorization을 구현한 레포지토리입니다.

광운대학교 홀로그램 프로젝트

<hr/>

## Table of Contents 

 1. [Preferences](#Preferences)
 2. [Install requirements](#Install-requirements)
 3. [Preparing datasets](#Preparing-datasets)
 4. [Train](#Train)
 5. [Eval](#Eval)
 6. [Predict](#Predict)
 7. [Reference](#Reference)

<hr/>

## Preferences

ESDet은 Tensorflow 기반 코드로 작성되었습니다. 코드는 **Windows** 및 **Linux(Ubuntu)** 환경에서 모두 동작합니다.
<table border="0">
<tr>
    <tr>
        <td>
        OS
        </td>
        <td>
        Ubuntu 20.10
        </td>
    </tr>
    <tr>
        <td>
        TF version
        </td>
        <td>
        2.6.2
        </td>
    </tr>
    <tr>
        <td>
        Python version
        </td>
        <td>
        3.9.0
        </td>
    </tr>
    <tr>
        <td>
        CUDA
        </td>
        <td>
        11.1
        </td>
    </tr>
    <tr>
        <td>
        CUDNN
        </td>
        <td>
        cuDNN v8.0.5 (November 9th, 2020), for CUDA 11.1
        </td>
    </tr>
    <tr>
        <td>
        GPU
        </td>
        <td>
        NVIDIA RTX3090 24GB
        </td>
    </tr>
</table>

<hr/>

## Install requirements

학습 및 평가를 위해 **Anaconda(miniconda)** 가상환경에서 패키지를 다운로드 합니다.

Download the package from the **Anaconda (miniconda)** virtual environment for training and evaluation.

    conda create -n envs_name python=3.9

    pip install tensorflow==2.6.2
    pip install tensorflow_datasets

<hr/>


## Prediction Demo
사전 저장된 모델로 이미지 추론을 predict.py로 실행합니다. 테스트에 사용할 이미지 파일과 출력 결과를 저장할 디렉토리를 지정해야 합니다.  
Run image inference with predict.py with pre-saved models. You must specify an image file to use for testing and a directory to store the output results.

Demo 프로그램은 이미지 해상도를 224x224로 고정합니다.

The demo program fixes the image resolution to 224x224.

```plain
└── root
       ├── demo_images/  # This is the image directory to use for testing.
       |   ├── image_1.jpg 
       |   └── image_2.jpg
       └── checkpoint/  # This is the directory to save the inference result image.    
           └── results/ 
               └── image_2_output.jpg
```  
<br/>

아래와 같이 실행할 수 있습니다.  

    python predict.py
