path1="1"
path2="1 2 3 4"
try="1 2 3"

# 学習とテストの経路が同じ
for t in $try
do
    for train_path in $path1
    do
        for i in $(seq 1 13)
        do
            python main.py --config config/CTRGCN_VITPOSE_aug0.yaml \
            --work_dir ./results/walkpath/CTRGCN/VITPOSE0/$t \
            --phase train \
            --evaluation_method walk_path_leave_pair_out \
            --train_data_path ./data/VITPOSE/$t/data.npy \
            --train_label_path ./data/VITPOSE/$t/label.npy \
            --train_walkpath $train_path \
            --test_data_path ./data/VITPOSE/$t/data.npy \
            --test_label_path ./data/VITPOSE/$t/label.npy \
            --test_walkpath $train_path \
            --leave_pair $(($i*4-3)) $(($i*4-2)) $(($i*4-1)) $(($i*4))
        done
    done
done


for t in $try
do
    for train_path in $path1
    do
        for test_path in $path2
        do
            for i in $(seq 1 13)
            do
                python main.py --config config/CTRGCN_VITPOSE_aug0.yaml \
                --work_dir ./results/walkpath/CTRGCN/VITPOSE0/$t \
                --phase test \
                --evaluation_method walk_path_leave_pair_out \
                --train_data_path ./data/VITPOSE/$t/data.npy \
                --train_label_path ./data/VITPOSE/$t/label.npy \
                --train_walkpath $train_path \
                --test_data_path ./data/VITPOSE/$t/data.npy \
                --test_label_path ./data/VITPOSE/$t/label.npy \
                --test_walkpath $test_path \
                --leave_pair $(($i*4-3)) $(($i*4-2)) $(($i*4-1)) $(($i*4))
            done
        done
    done
done