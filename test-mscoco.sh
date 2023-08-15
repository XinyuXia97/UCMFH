# 0 is train real, 1 is train hash, 2 is test real, 3 is test hash
TASK=1
if [ $TASK == 1 ] || [ $TASK == 3 ]
then
    for hash in 16 32 64 128
    do 
        python demo.py --dataset mscoco  --epoch 500 --device cuda:2 --hash_lens $hash --task $TASK
    done
elif [ $TASK == 0 ] || [ $TASK == 2 ]
then
    python demo.py --dataset mscoco --epoch 100 --device cuda:2 --task $TASK
fi

