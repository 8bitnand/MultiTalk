#!/bin/bash

# MultiTalk Video Generation Script
# Usage: ./run_multitalk.sh

set -e  # Exit on error

# =============================================================================
# CONFIGURATION - Edit these values
# =============================================================================

# Model paths
CKPT_DIR="weights/Wan2.1-I2V-14B-480P"
WAV2VEC_DIR="weights/chinese-wav2vec2-base"
LORA_DIR="weights/FusionX_LoRa/Wan2.1_I2V_14B_FusionX_LoRA.safetensors"

# Input files
PROMPT="A person talking"
IMAGE="examples/multi/1/ref.jpg"
AUDIO1="examples/multi/1/human1.wav"
AUDIO2=""  # Leave empty for single person, or add path for multi-person
BBOX1=""   # Optional: "x1,y1,x2,y2"
BBOX2=""   # Optional: "x1,y1,x2,y2"

# Output settings
SAVE_FILE="output_video"
AUDIO_SAVE_DIR="save_audio"

# Generation settings
SAMPLE_STEPS=8          # 8 for FusionX LoRA, 40 for standard
TEXT_GUIDE_SCALE=1.0    # 1.0 for LoRA, 5.0 for standard
AUDIO_GUIDE_SCALE=2.0   # 2.0 for LoRA, 4.0 for standard
SAMPLE_SHIFT=2          # 2 for FusionX, 7 for standard
MODE="streaming"        # "clip" or "streaming"
FRAME_NUM=81

# Intelligent chunking settings
USE_INTELLIGENT_CHUNKING=true
SAVE_CHUNKS=true
SILENCE_THRESH_DB=-40
MIN_SILENCE_LEN=0.5

# Memory settings
NUM_PERSISTENT_PARAM=0  # 0 for low VRAM, higher for more VRAM
USE_TEACACHE=true

# =============================================================================
# SCRIPT - Don't modify below unless you know what you're doing
# =============================================================================

echo "======================================"
echo "MultiTalk Video Generation"
echo "======================================"

# Create output directory
mkdir -p "$AUDIO_SAVE_DIR"

# Build command
CMD="python generate_multitalk.py \
    --ckpt_dir $CKPT_DIR \
    --wav2vec_dir $WAV2VEC_DIR \
    --prompt \"$PROMPT\" \
    --image $IMAGE \
    --audio1 $AUDIO1 \
    --sample_steps $SAMPLE_STEPS \
    --sample_text_guide_scale $TEXT_GUIDE_SCALE \
    --sample_audio_guide_scale $AUDIO_GUIDE_SCALE \
    --sample_shift $SAMPLE_SHIFT \
    --mode $MODE \
    --frame_num $FRAME_NUM \
    --num_persistent_param_in_dit $NUM_PERSISTENT_PARAM \
    --save_file $SAVE_FILE \
    --audio_save_dir $AUDIO_SAVE_DIR \
    --base_seed 42"

# Add LoRA if specified
if [ -n "$LORA_DIR" ] && [ -f "$LORA_DIR" ]; then
    CMD="$CMD --lora_dir $LORA_DIR --lora_scale 1.0"
    echo "Using LoRA: $LORA_DIR"
fi

# Add second audio if specified
if [ -n "$AUDIO2" ]; then
    CMD="$CMD --audio2 $AUDIO2"
    echo "Multi-person mode with 2 audio files"
fi

# Add bounding boxes if specified
if [ -n "$BBOX1" ]; then
    CMD="$CMD --bbox1 $BBOX1"
fi
if [ -n "$BBOX2" ]; then
    CMD="$CMD --bbox2 $BBOX2"
fi

# Add intelligent chunking
if [ "$USE_INTELLIGENT_CHUNKING" = true ]; then
    CMD="$CMD --intelligent_chunking --silence_thresh_db $SILENCE_THRESH_DB --min_silence_len $MIN_SILENCE_LEN"
    echo "Intelligent chunking enabled"
fi

# Add chunk saving
if [ "$SAVE_CHUNKS" = true ]; then
    CMD="$CMD --save_chunks"
    echo "Saving chunks enabled"
fi

# Add TeaCache
if [ "$USE_TEACACHE" = true ]; then
    CMD="$CMD --use_teacache"
    echo "TeaCache acceleration enabled"
fi

echo ""
echo "Input:"
echo "  Prompt: $PROMPT"
echo "  Image: $IMAGE"
echo "  Audio1: $AUDIO1"
[ -n "$AUDIO2" ] && echo "  Audio2: $AUDIO2"
echo ""
echo "Settings:"
echo "  Steps: $SAMPLE_STEPS"
echo "  Mode: $MODE"
echo "  Output: $SAVE_FILE"
echo ""
echo "Starting generation..."
echo "======================================"

# Run the command
eval $CMD

echo ""
echo "======================================"
echo "Generation complete!"
if [ "$SAVE_CHUNKS" = true ] && [ "$USE_INTELLIGENT_CHUNKING" = true ]; then
    echo "Output saved in: ${SAVE_FILE}_chunks/"
    echo "  - Individual chunks: ${SAVE_FILE}_chunks/chunk_*.mp4"
    echo "  - Final video: ${SAVE_FILE}_chunks/${SAVE_FILE}_final.mp4"
else
    echo "Output: ${SAVE_FILE}.mp4"
fi
echo "======================================"
