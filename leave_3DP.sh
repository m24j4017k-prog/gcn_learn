try="1 2 3"
augument="0"

for aug in $augument
do
    for t in $try	
    do
        for i in $(seq 1 13)
        do
            python main.py --config config/default.yaml \
            --work_dir ./results/leave_4_pair_out/time/MB_3DP$aug/$t/$i \
            --train_data_path ./data/MB_3DP/$t/data.npy \
            --train_label_path ./data/MB_3DP/$t/label.npy \
            --test_data_path ./data/MB_3DP/$t/data.npy \
            --test_label_path ./data/MB_3DP/$t/label.npy \
            --leave_pair $(($i*4-3)) $(($i*4-2)) $(($i*4-1)) $(($i*4))
        done
    done
done