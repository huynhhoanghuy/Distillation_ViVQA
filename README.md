# Distillation_ViVQA
I have a large pretrained ViVQA model, I have a small model. Boom!!! I have a small ViVQA model with high precision.


Models for Visual Question Answering on ViVQA dataset

# dataset:

https://drive.google.com/file/d/1X5eORxfRTJ_UN26OWHjZnhsQKpo9euiG/view?usp=sharing

# pretrained weight (DucAnh's paper): 

https://drive.google.com/file/d/1eNSfuHDsf5TriGt-OjbAZ88sTQvrZzwt/view?usp=sharing

# knowledge-distillation-pytorch
virtualenv -p python3.8 venv3.8
source venv3.8/bin/activate

pip3 install -r requirements.txt

* Please install torch follow up CUDA version.

HOW TO RUN:

bash train_tunning.sh
