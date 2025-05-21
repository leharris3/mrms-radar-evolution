cd diffusers
export CUDA_VISIBLE_DEVICES=0,1,2,3

accelerate launch train_controlnet_tornado.py \
    --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
    --output_dir outputs/tornado-controlnet \
    --train_batch_size 2 --num_train_epochs 5