# for i in 11 12 13
# do
#     echo $i
#     liblinear/train -s $i train.nopos.inst models/$i.nopos.model 
# done

for i in 11 12 13
do
    echo $i
    liblinear/predict -b 1 dev.nopos.inst models/$i.nopos.model output
done