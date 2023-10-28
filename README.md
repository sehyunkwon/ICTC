# Image Clustering Conditioned on Text Criteria

## Contents
- [Install](#install)
- [Dataset Prep](#dataset-prep)
- [Usage](#usage)

## Install

1. Clone this repository
```bash
git clone https://github.com/sehyunkwon/ICTC.git
cd ICTC
```

2. Install Packages
```Shell
conda create -n ictc python=3.10 -y
conda activate ictc
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

## Dataset Prep
This section explains how to prepare the images that we used. The images have to be in **.jpg**, **.JPEG**, or **.png** format (you can add more in **load_image_paths_from_folder** function from **models.py** file). The image names have to be in the following format: {IMG_NUMBER}_{CLASS_NAME}.{FORMAT}.

### MNIST
```
mkdir -p ~/data/mnist
cd ~/data/mnist
git clone https://github.com/rasbt/mnist-pngs.git
mv mnist-pngs/test/* .
rm -rf mnist-pngs
cd ~/ICTC/scripts
./mnist.sh
```

### CIFAR10
```
mkdir -p ~/data/cifar10
cd ~/data/cifar10
wget http://pjreddie.com/media/files/cifar.tgz
tar xzf cifar.tgz
mv ./cifar/test/* .     # only use test set
rm -rf cifar*
```

### CIFAR100
```
pip install cifar2png
cifar2png cifar100superclass ~/data/cifar100/
cd ~/data/cifar100/
rm -rf train/
mv ./test/* .     # only use test set
rm -rf test/
cd ~/ICTC/scripts
./cifar100.sh            # change the file format
```
### STL10
```
mkdir -p ~/data/stl10/
cd ~/ICTC/scripts
python stl10_download.py
cd ~/data/stl10/
rm -rf stl10_binary*
cd ~/ICTC/scripts
python stl10.py         # change the file format
cd ~/data/stl10/
rm -rf test/*/
```

### Stanford-40-actions
```
mkdir -p ~/data/stanford-40-actions
cd ~/data/stanford-40-actions
wget http://vision.stanford.edu/Datasets/Stanford40_JPEGImages.zip
unzip Stanford40_JPEGImages.zip
rm -rf Stanford40_JPEGImages.zip
cd ~/ICTC/scripts
./stanford.sh
```
### PPMI (People Playing Musical Instruments)
The 7 subclasses are: saxophone, guitar, trumpet, cello, flute, violin and harp
```
mkdir -p ~/data/ppmi
cd ~/data/ppmi
wget http://vision.stanford.edu/Datasets/norm_ppmi_12class.zip
unzip norm_ppmi_12class.zip
rm -rf norm_ppmi_12class.zip README
rm -rf norm_image/with_instrument
rm -rf ./*/*/train
mv ./norm_image/play_instrument/*/test/* .
rm -rf norm_image/
cd ~/ICTC/scripts
./ppmi.sh
cd ~/data/ppmi
mkdir 12_classes
mkdir 7_classes
mkdir 2_categories
cd ~/ICTC/scripts
cp * ~/data/ppmi./12_classes
cp *_Saxophone.jpg *_Guitar.jpg *_Trumpet.jpg *_Cello.jpg *_Flute.jpg *_Violin.jpg *_Harp.jpg ./7_classes
cp ./7_classes/* ./2_categories
cd ~/ICTC/scripts
./2_categories.sh
```

## Usage
Assuming the current directory is **~/ICTC**.

#### **1. Obtain image description from VLM** (Step 1)
Image descriptions will be saved in **'./{dataset}/initial_answer.jsonl'**.
```
python llava/eval/model.py --dataset cifar10
or
python blip2/model.py --dataset cifar10
```

#### **2. Obtain labels from VLM descriptions using LLM** (Step 2a)
From the description of the image provided by VLM, ask LLM for a possible label.
```
python ictc/llm_step2a.py --dataset cifar10
```
To use GPT-4, enable **args.use_gpt4**. Or to use Llama-2, enable **args.llama**

#### **3. Cluster possible labels to N labels using LLM** (Step 2b)
Using the saved file containing initial labels, ask LLM to summarize them into K classes.
```
python ictc/llm_step2b.py --dataset cifar10
```

#### **4. Final classification(=clustering)** (Step 3)
Now you obtained K(e.g., K=10) classes. Feed LLM with image description and a list of possible classes. Obtain the final classification.
```
python ictc/llm_step3.py --dataset cifar10
```

#### **5. Measuring acc, ari, nmi using the Hungarian Matching algorithm**
Classic metrics used in clustering literature.
```
python ictc/measuring_acc.py --dataset cifar10
```

**Example** : PPMI 7 classes; `'ACC': 96.29, 'ARI': 91.81, 'NMI': 92.64`

<img src="./ictc/ppmi/gpt3.5/7_classes_v1/confusion_matrix.png" width="400px" height="400px" title="Confusion matrix of PPMI 7 classes"/>
</img>


#### **6. Launch the entire Pipeline**
If you wish to launch the entire pipeline with a one-liner, use
```
python ictc/launch_full_pipeline.py
```
Note: this may not work if GPT fails to follow the specified output formatting.

## Results

### CIFAR10
- GPT 3.5: 91.37% `~/ICTC/ictc/cifar10/gpt3.5/final_old_prompt`
- GPT 4: 90.75% `~/ICTC/ictc/cifar10/gpt4/final_old_prompt`

### CIFAR100
- GPT 3.5: 51.34% `~/ICTC/ictc/cifar100/gpt3.5/orig`
- GPT 4: 58.89% `~/ICTC/ictc/cifar100/gpt4/final`

### STL10
- GPT 3.5: 97.7% `~/ICTC/ictc/stl10/gpt3.5/final_prompt`
- GPT 4: 98.63% `~/ICTC/ictc/stl10/gpt4/final_prompt`

### Stanford-40-actions
#### Actions
- GPT 3.5: 71.44% `~/ICTC/ictc/stanford-40-actions/gpt3.5/actions_40_classes_final_prompt`
- GPT 4: 77.39% `~/ICTC/ictc/stanford-40-actions/gpt4/actions_40_classes_final_prompt`

#### Mood
- GPT 4: 76.0% `~/ICTC/ictc/stanford-40-actions/gpt4/mood_4_classes`

#### Location - 2 classes
- GPT 4: `~/ICTC/ictc/stanford-40-actions/gpt4/location_2_classes`

#### Location - 10 classes
- GPT 4: 86.0% `~/ICTC/ictc/stanford-40-actions/gpt4/location_10_classes`

### PPMI
#### 2-classes
Ground truth: wind instruments vs string instruments
- GPT 3.5: 93.71% `~/ICTC/ictc/ppmi/gpt3.5/2_classes_v1`
- GPT 4: 97.71% `~/ICTC/ictc/ppmi/gpt4/2_classes_final_prompt`
#### 7-classes
- GPT 3.5: 96.29% `~/ICTC/ictc/ppmi/gpt3.5/7_classes_v1`
- GPT 4: 96.43% : `~/ICTC/ictc/ppmi/gpt4/7_classes_final_prompt`

## Ablation Results
### Ablate Step 3:  
#### CIFAR10
- LLaMa 7b: 87.73% `~/ICTC/ictc/cifar10/llama_7b/gpt_classes`
- LLaMa 13b: 87.36% `~/ICTC/ictc/cifar10/llama_13b/gpt_classes`
- LLaMa 70b: 88.41% `~/ICTC/ictc/cifar10/llama_70b/gpt_classes`

#### CIFAR100
- LLaMa 7b: 46.53% `~/ICTC/ictc/cifar100/llama_7b/gpt_classes`
- LLaMa 13b: 45.44% `~/ICTC/ictc/cifar100/llama_13b/gpt_classes`
- LLaMa 70b: 52.6% `~/ICTC/ictc/cifar100/llama_70b/gpt_classes`

#### STL10
- LLaMa 7b: 94.06% `~/ICTC/ictc/stl10/llama_7b/gpt_classes`
- LLaMa 13b: 95.49% `~/ICTC/ictc/stl10/llama_13b/gpt_classes`
- LLaMa 70b: 97.41% `~/ICTC/ictc/stl10/llama_70b/gpt_classes`

### Ablate Step 1+3: 
#### CIFAR10
- LLaMa 7b:
- LLaMa 13b:
- LLaMa 70b:

#### CIFAR100
- LLaMa 7b:
- LLaMa 13b:
- LLaMa 70b:

#### STL10
- LLaMa 7b:
- LLaMa 13b:
- LLaMa 70b:

### LLaMa only
#### CIFAR10
- LLaMa 7b:
- LLaMa 13b:
- LLaMa 70b:

#### CIFAR100
- LLaMa 7b:
- LLaMa 13b:
- LLaMa 70b:

#### STL10
- LLaMa 7b:
- LLaMa 13b:
- LLaMa 70b:

### LLaVa only
- CIFAR10:
- CIFAR100:
- STL10:
- Stanford-40-actions:
- PPMI 2 classes:
- PPMI 7 classes: