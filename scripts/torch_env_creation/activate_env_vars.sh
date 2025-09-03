export UUID_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES

# Must set CUDA_VISIBLE_DEVICES to device integers in place of UUID's for compatibility with PyTorch 1.X
# See https://github.com/pytorch/pytorch/issues/90543
for d in ${CUDA_VISIBLE_DEVICES//,/ }
do
   DEV_NUM=$(cut -d ' ' -f 2 <<< `nvidia-smi -L | grep $d`)
   DEV_NUM=${DEV_NUM/:/}
   export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES/$d/$DEV_NUM}
done