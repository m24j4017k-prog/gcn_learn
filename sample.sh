for i in 1 2 3 4
do
    a=$(( i % 4 + 1 ))
    b=$(( (i + 1) % 4 + 1 ))
    c=$(( (i + 2) % 4 + 1 ))

    echo "$a $b $c"
done
