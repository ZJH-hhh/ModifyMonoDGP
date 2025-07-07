CFG=$1
cat "${CFG}"  
echo ""

CUDA_VISIBLE_DEVICES=0 python tools/train_val.py --config $@ -e