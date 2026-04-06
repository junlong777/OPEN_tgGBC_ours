export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONPATH="$(pwd):$PYTHONPATH"

# Evaluate OPEN wiht tgGBC using a single GPU.
python tools/test.py \
    projects/configs/open_r101_1408_90e.py \
    ckpts/open_r101_1408.pth \
    --eval bbox \
    --gbc -r 12000 -n 2 -k 175