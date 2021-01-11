How to set up the environment:

1. Load project in PyCharm
2. Add a new Run/Debug configuration with the main.py.
3. Add one of the following as parameters, under script path:

#used for training
--dataset=crackconcrete --arch=unet --dip=crackconcrete --gpu --train

#used for testing
--dataset=crackconcrete --arch=unet --dip=crackconcrete --gpu --test
