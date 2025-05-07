## 12. Let's create a shell script to run the extended models:

```bash
#!/bin/bash

# Script to run the extended MedFuse models
# Usage: ./run_extensions.sh [task] [fusion_module] [data_ratio]

# Default parameters
TASK=${1:-"in-hospital-mortality"}
FUSION_MODULE=${2:-"attention"}
DATA_RATIO=${3:-0.2}

# Create output directory
OUTPUT_DIR="checkpoints/${TASK}/${FUSION_MODULE}_ratio_${DATA_RATIO}"
mkdir -p $OUTPUT_DIR

# Run training
echo "Training ${FUSION_MODULE} model for ${TASK} with uni-modal ratio ${DATA_RATIO}"
python extended_main.py \
    --fusion_module ${FUSION_MODULE} \
    --task ${TASK} \
    --data_ratio ${DATA_RATIO} \
    --mode train \
    --epochs 50 \
    --batch_size 16 \
    --dim 256 \
    --dropout 0.3 \
    --layers 2 \
    --vision-backbone resnet34 \
    --visualize_attention \
    --save_dir ${OUTPUT_DIR}

# Run evaluation
echo "Evaluating ${FUSION_MODULE} model for ${TASK}"
python extended_main.py \
    --fusion_module ${FUSION_MODULE} \
    --task ${TASK} \
    --data_ratio ${DATA_RATIO} \
    --mode eval \
    --load_state ${OUTPUT_DIR}/best_checkpoint.pth.tar \
    --save_dir ${OUTPUT_DIR}

echo "Completed running ${FUSION_MODULE} model for ${TASK} with uni-modal ratio ${DATA_RATIO}"