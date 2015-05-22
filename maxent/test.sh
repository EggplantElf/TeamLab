for i in 0 1 2 3 4 5 6 7
do
    echo $i
    liblinear/train -s $i train.pos.inst models/$i.nopos.model 
done

for i in 0 1 2 3 4 5 6 7
do
    echo $i
    liblinear/predict dev.pos.inst models/$i.pos.model output
done