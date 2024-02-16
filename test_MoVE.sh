source ../../venv3.8/bin/activate
export CUDA_VISIBLE_DEVICES=0
#AUGMENTATION


python3 test_MoVE.py --model 'GuidedAtt' \
                    --print_summary \
                    --vit_backbone 'vit' --vit_image_pretrained 'google/vit-base-patch16-224-in21k'\
                    --cnn_backbone 'resnet34' --cnn_image_pretrained 'resnet34'\
                    --bert_type 'phobert' --bert_pretrained 'vinai/phobert-base' \
                    --q_dim 768 --v_vit_dim 768 --v_cnn_dim 512 --question_len 20 \
                    --glimpse 1 --joint_dim 1024 \
                    --data_dir '/home/dmp/Desktop/1.Users/3.huy.hhoang/input/VQA_raw/ViVQA25' \
                    --output "/home/dmp/Desktop/1.Users/3.huy.hhoang/input/out" \
                    --init_lr 5e-5 --warmup_steps 5 --weight_decay 5e-6 \
                    --label_smooth 0 --dropout 0 \
                    --batch_size 1 --nepochs 1
