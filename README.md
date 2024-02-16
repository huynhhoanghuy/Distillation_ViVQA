# Distillation_ViVQA
Distill Visual Question Answering model on ViVQA dataset.
Built up a MoVE system based on distilled-model for precision improvement.

# dataset:

https://drive.google.com/file/d/1X5eORxfRTJ_UN26OWHjZnhsQKpo9euiG/view?usp=sharing

# pretrained weight (DucAnh's paper): 

https://drive.google.com/file/d/1eNSfuHDsf5TriGt-OjbAZ88sTQvrZzwt/view?usp=sharing

# pretrained weight (our paper): 

https://drive.google.com/file/d/1umQkszA6JEP2l9Wvwukbu1uuzwjw3ihm/view?usp=sharing

# knowledge-distillation-pytorch
virtualenv -p python3.8 venv3.8
source venv3.8/bin/activate

pip3 install -r requirements.txt

* Please install torch follow up CUDA version.

HOW TO RUN:

bash test_MoVE.sh
