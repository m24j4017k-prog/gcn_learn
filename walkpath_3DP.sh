try="1 2 3"



for tr in $path1
do
    for te in $path2
        do
            for t in $try	
            do
                for i in $(seq 1 13)
                do
                    python main.py --config config/default.yaml \
                    --evaluation_method walk_path_leave_pair_out \
                    --train-data-path ./data/MB_3DP/$t/data.npy \
                    --train-label-path ./data/MB_3DP/$t/label.npy \
                    --train-walk-path $tr \
                    --test-data-path ./data/MB_3DP/$t/data.npy \
                    --test-label-path ./data/MB_3DP/$t/label.npy \
                    --test-walk-path $te \
                    --leave-pair $(($i*4-3)) $(($i*4-2)) $(($i*4-1)) $(($i*4))
                done
            done
        done
    done
done