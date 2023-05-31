export CUDA_VISIBLE_DEVICES=2

python3 train_4_distill.py --model 'GuidedAtt' \
                    --print_summary \
                    --vit_backbone 'vit' --vit_image_pretrained 'google/vit-base-patch16-224-in21k'\
                    --cnn_backbone 'resnet34' --cnn_image_pretrained 'resnet34'\
                    --bert_type 'phobert' --bert_pretrained 'vinai/phobert-base' \
                    --q_dim 768 --v_vit_dim 768 --v_cnn_dim 512 --question_len 20 \
                    --glimpse 1 --joint_dim 1024 \
                    --data_dir '/data/huy.hhoang/ViVQA25' \
                    --output "/data/huy.hhoang/ViVQA_model/output/trained_models/GuidedAtt" \
                    --init_lr 1e-4 --warmup_steps 5 --weight_decay 1e-5 \
                    --label_smooth 0.0 --dropout 0.3 \
                    --batch_size 32 --nepochs 40

python3 train_4_distill.py --model 'GuidedAtt' \
                    --print_summary \
                    --vit_backbone 'vit' --vit_image_pretrained 'google/vit-base-patch16-224-in21k'\
                    --cnn_backbone 'resnet34' --cnn_image_pretrained 'resnet34'\
                    --bert_type 'phobert' --bert_pretrained 'vinai/phobert-base' \
                    --q_dim 768 --v_vit_dim 768 --v_cnn_dim 512 --question_len 20 \
                    --glimpse 1 --joint_dim 1024 \
                    --data_dir '/data/huy.hhoang/ViVQA25' \
                    --output "/data/huy.hhoang/ViVQA_model/output/trained_models/GuidedAtt" \
                    --init_lr 5e-5 --warmup_steps 5 --weight_decay 1e-5 \
                    --label_smooth 0.0 --dropout 0.3 \
                    --batch_size 32 --nepochs 40

python3 train_4_distill.py --model 'GuidedAtt' \
                    --print_summary \
                    --vit_backbone 'vit' --vit_image_pretrained 'google/vit-base-patch16-224-in21k'\
                    --cnn_backbone 'resnet34' --cnn_image_pretrained 'resnet34'\
                    --bert_type 'phobert' --bert_pretrained 'vinai/phobert-base' \
                    --q_dim 768 --v_vit_dim 768 --v_cnn_dim 512 --question_len 20 \
                    --glimpse 1 --joint_dim 1024 \
                    --data_dir '/data/huy.hhoang/ViVQA25' \
                    --output "/data/huy.hhoang/ViVQA_model/output/trained_models/GuidedAtt" \
                    --init_lr 1e-4 --warmup_steps 5 --weight_decay 1e-5 \
                    --label_smooth 0.15 --dropout 0.3 \
                    --batch_size 32 --nepochs 40

python3 train_4_distill.py --model 'GuidedAtt' \
                    --print_summary \
                    --vit_backbone 'vit' --vit_image_pretrained 'google/vit-base-patch16-224-in21k'\
                    --cnn_backbone 'resnet34' --cnn_image_pretrained 'resnet34'\
                    --bert_type 'phobert' --bert_pretrained 'vinai/phobert-base' \
                    --q_dim 768 --v_vit_dim 768 --v_cnn_dim 512 --question_len 20 \
                    --glimpse 1 --joint_dim 1024 \
                    --data_dir '/data/huy.hhoang/ViVQA25' \
                    --output "/data/huy.hhoang/ViVQA_model/output/trained_models/GuidedAtt" \
                    --init_lr 1e-4 --warmup_steps 5 --weight_decay 1e-5 \
                    --label_smooth 0.15 --dropout 0.2 \
                    --batch_size 32 --nepochs 40