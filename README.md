# Distillation_ViVQA
I have a large pretrained ViVQA model, I have a small model. Boom!!! I have a small ViVQA model with high precision.


Models for Visual Question Answering on ViVQA dataset

# Requirement
Python package:
+ timm
+ torchinfo
+ transformers
# Training

**To train Cross Attention model**

!python3 train.py --model 'CrossAtt' \
                 --backbone 'vit' --image_pretrained 'google/vit-base-patch16-224-in21k'\
                 --bert_type 'phobert' --bert_pretrained 'vinai/phobert-base' \
                 --q_dim 768 --v_dim 768 --question_len 20 \
                 --f_mid_dim 1024 --joint_dim 768 \
                 --n_coatt 2 \
                 --data_dir '/content/dataset' \
                 --output "/content/drive/MyDrive/Colab Notebooks/vivqa-models/CrossAtt/trained_models" \
                 --init_lr 1e-4 --warmup_steps 5 --weight_decay 1e-5 \
                 --label_smooth 0.15 --dropout 0.2 \
                 --batch_size 32 --nepochs 40

**To train Guide Attention model**
!python3 train_2.py --model 'GuidedAtt' \
                    --vit_backbone 'vit' --vit_image_pretrained 'google/vit-base-patch16-224-in21k'\
                    --cnn_backbone 'resnet34' --cnn_image_pretrained 'resnet34'\
                    --bert_type 'phobert' --bert_pretrained 'vinai/phobert-base' \
                    --q_dim 768 --v_vit_dim 768 --v_cnn_dim 512 --question_len 20 \
                    --glimpse 1 --joint_dim 1024 \
                    --data_dir '/content/dataset' \
                    --output "/content/drive/MyDrive/Colab Notebooks/vivqa-models/trained_models/GuidedAtt" \
                    --init_lr 1e-4 --warmup_steps 5 --weight_decay 1e-5 \
                    --label_smooth 0.0 --dropout 0.3 \
                    --batch_size 32 --nepochs 40

# Evaluate

**Evaluate Cross Attention model**
!python3 test.py --model 'CrossAtt' \
                 --use_spatial --use_cma \
                 --backbone 'vit' --image_pretrained 'google/vit-base-patch16-224-in21k'\
                 --bert_type 'phobert' --bert_pretrained 'vinai/phobert-base' \
                 --q_dim 768 --v_dim 768 --question_len 20 \
                 --n_coatt 2 --joint_dim 768 \
                 --data_dir '/content/dataset' \
                 --input "/content/drive/MyDrive/Colab Notebooks/vivqa-models/trained_models/CrossAtt/GuidedAtt_vit_resnet34_phobert_25_06_2022__05_36_35.pt" \
                 --label_smooth 0.15 --dropout 0.2

**Evaluate Guided Attention**
!python3 test.py --model 'GuidedAtt' \
                 --use_spatial --use_cma \
                 --vit_backbone 'vit' --vit_image_pretrained 'google/vit-base-patch16-224-in21k'\
                 --cnn_backbone 'resnet34' --cnn_image_pretrained 'resnet34'\
                 --bert_type 'phobert' --bert_pretrained 'vinai/phobert-base' \
                 --seed 1 \
                 --q_dim 768 --v_vit_dim 768 --v_cnn_dim 512 --question_len 20 \
                 --glimpse 1 --joint_dim 1024 \
                 --data_dir '/content/dataset' \
                 --input "/content/drive/MyDrive/Colab Notebooks/vivqa-models/trained_models/GuidedAtt/GuidedAtt_vit_resnet34_phobert_28_06_2022__10_41_16.pt" \
                 --label_smooth 0.15 --dropout 0.2