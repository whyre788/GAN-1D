# GAN-1D
using WGAN to generate fault bearing vibration signals

request:

python 3.5+

tensorflow-gpu

numpy scipy os

open cmd and cd to the folder 

  $ python train.py 

  --learning_rate 0.0000001 #change the learning rate,default 0.0000001
                  
  --epoch 2000001 #how much epochs to train,default 2000001
                  
  --sample_rate 50000 #how many epochs you want to sample once,default 50000
                  
  --train_data x1 #there 9 kinds of signals you can choose,default x1
                  
your model will be saved at ./checkpoint/ per 100 epochs

  $ python test.py 

test you model, output will be saved at ./output/

It takes about 11 hours to run 2000 epochs in Titan XP.
