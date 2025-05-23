cd diffusers
export CUDA_VISIBLE_DEVICES=0
accelerate launch examples/controlnet/train_controlnet_sdxl.py \
    --pretrained_model_name_or_path stabilityai/stable-diffusion-xl-base-1.0 \
    --output_dir outputs/tornado-controlnet \
    --train_batch_size 1 \
    --num_train_epochs 1 \
    --train_data_dir "" \
    --use_toy_dataset \