# 0 is train real, 1 is train hash, 2 is test real, 3 is test hash
TASK=3
if [ $TASK == 1 ] || [ $TASK == 3 ]
then
    for hash in 16 32 64 128
    do 
        python demo.py --dataset nus-wide --epoch 150 --device cuda:1 --hash_lens $hash --task $TASK
    done
elif [ $TASK == 0 ] || [ $TASK == 2 ]
then
    python demo.py --dataset nus-wide --epoch 150 --device cuda:1 --task $TASK
fi