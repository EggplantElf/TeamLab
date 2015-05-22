python maxent.py -i

liblinear/train -s 0 train.nopos.inst models/m0.model
liblinear/train -s 0 train.pos.inst models/m1.model

python maxent -p
