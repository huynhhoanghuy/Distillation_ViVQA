python3 inference.py --model 'GuidedAtt' \
                    --use_spatial --use_cma \
                    --vit_backbone 'vit' --vit_image_pretrained 'google/vit-base-patch16-224-in21k'\
                    --cnn_backbone 'resnet34' --cnn_image_pretrained 'resnet34'\
                    --bert_type 'phobert' --bert_pretrained 'vinai/phobert-base' \
                    --seed 1 \
                    --q_dim 768 --v_vit_dim 768 --v_cnn_dim 512 --question_len 20 \
                    --glimpse 1 --joint_dim 1024 \
                    --data_dir '/content/dataset' \
                    --input '/content/drive/MyDrive/Colab Notebooks/vivqa-models/trained_models/GuidedAtt/GuidedAtt_vit_resnet34_phobert_24_06_2022__14_51_56.pt' \
                    --indices 100