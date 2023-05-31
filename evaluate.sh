## Evaluate Cross Attention model

# python3 test.py --model 'CrossAtt' \
#                  --use_spatial --use_cma \
#                  --backbone 'vit' --image_pretrained 'google/vit-base-patch16-224-in21k'\
#                  --bert_type 'phobert' --bert_pretrained 'vinai/phobert-base' \
#                  --q_dim 768 --v_dim 768 --question_len 20 \
#                  --n_coatt 2 --joint_dim 768 \
#                  --data_dir '/content/dataset' \
#                  --input "/content/drive/MyDrive/Colab Notebooks/vivqa-models/trained_models/CrossAtt/GuidedAtt_vit_resnet34_phobert_25_06_2022__05_36_35.pt" \
#                  --label_smooth 0.15 --dropout 0.2

## Evaluate Guided Attention

python3 test.py --model 'GuidedAtt' \
                 --use_spatial --use_cma \
                 --vit_backbone 'vit' --vit_image_pretrained 'google/vit-base-patch16-224-in21k'\
                 --cnn_backbone 'resnet34' --cnn_image_pretrained 'resnet34'\
                 --bert_type 'phobert' --bert_pretrained 'vinai/phobert-base' \
                 --seed 1 \
                 --q_dim 768 --v_vit_dim 768 --v_cnn_dim 512 --question_len 20 \
                 --glimpse 1 --joint_dim 1024 \
                 --data_dir '/data/huy.hhoang/ViVQA25' \
                 --input "/data/huy.hhoang/ViVQA_model/output/trained_models/GuidedAtt/GuidedAtt_vit_resnet34_phobert_11_04_2023__11_31_10.pt" \
                 --label_smooth 0.0 --dropout 0.3