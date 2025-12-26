path1="1 2 3 4"
path2="1 2 3 4"
try="1 2 3"
augument="1"

for aug in $augument
do
    # 学習とテストの経路が同じ
    for t in $try
    do
        for leave_path in $path1
        do
            for i in $(seq 1 13)
            do
                python main.py --config config/MB_3DP_aug$aug.yaml \
                --work_dir ./results/val2/STGCN/MB_3DP_$aug/$t \
                --phase train \
                --evaluation_method val2 \
                --train_data_path ./data/MB_3DP/$t/data.npy \
                --train_label_path ./data/MB_3DP/$t/label.npy \
                --train_walkpath $(( leave_path % 4 + 1 )) $(( (leave_path + 1) % 4 + 1 )) $(( (leave_path + 2) % 4 + 1 ))\
                --test_data_path ./data/MB_3DP/$t/data.npy \
                --test_label_path ./data/MB_3DP/$t/label.npy \
                --test_walkpath $(( leave_path % 4 + 1 )) $(( (leave_path + 1) % 4 + 1 )) $(( (leave_path + 2) % 4 + 1 ))\
                --leave_pair $(($i*4-3)) $(($i*4-2)) $(($i*4-1)) $(($i*4))
            done
        done
    done
done

for aug in $augument
do
    for t in $try
    do
        for leave_path in $path1
        do
            for test_path in $path2
            do
                for i in $(seq 1 13)
                do
                    python main.py --config config/MB_3DP_aug$aug.yaml \
                    --work_dir ./results/val2/STGCN/MB_3DP_$aug/$t \
                    --phase test \
                    --evaluation_method val2 \
                    --train_data_path ./data/MB_3DP/$t/data.npy \
                    --train_label_path ./data/MB_3DP/$t/label.npy \
                    --train_walkpath $(( leave_path % 4 + 1 )) $(( (leave_path + 1) % 4 + 1 )) $(( (leave_path + 2) % 4 + 1 ))\
                    --test_data_path ./data/MB_3DP/$t/data.npy \
                    --test_label_path ./data/MB_3DP/$t/label.npy \
                    --test_walkpath $test_path \
                    --leave_pair $(($i*4-3)) $(($i*4-2)) $(($i*4-1)) $(($i*4))
                done
            done
        done
    done
done