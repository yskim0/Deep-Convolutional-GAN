# Deep-Convolutional-GAN

This repositiory is for implementing `Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks(DCGAN)`

## Environment

- CUDA Version: 10.2 

```
torch==1.5.0
torchvision==0.6.0
numpy==1.19.2
```

## Usage

To train a model, run `train.py`.
If you need to speicfy the model, just use some args.

```
# training with gpu.
$ python train.py --gpu
```

optional&required arguments

```
--data_dir      default='./data/',
                help="Directory containing the dataset"
--lr            type=float, default=0.0002,
                help="Learning rate"
--b1            type=float, default=0.5,
                help="Momentum decay rate"
--b2            type=float, default=0.9,
                help="Adaptive term decay rate"
--latent_dim    type=int, default=100,
                help="Dimensionality of the latent space"
--epoch         type=int, default=100,
                help="Total training epochs"
--batch_size    type=int, default=64,
                help="batch size"
--img_ch        type=int, default=1,
                help="image channel size(MNIST: 1, CIFAR-10: 3)"
--gpu           action='store_true', default='False',
                help="GPU available"
```

## Results

[40 epochs]

![ezgif-3-4eff45838982](https://user-images.githubusercontent.com/48315997/103209926-95d1ea80-4947-11eb-9fc1-7e28da82424c.gif)


![42400](https://user-images.githubusercontent.com/48315997/103209819-599e8a00-4947-11eb-83f7-5914622f8ddc.png)
![44800](https://user-images.githubusercontent.com/48315997/103209805-4ee3f500-4947-11eb-9325-92a56c612a7e.png)


### 왜 이럴까?
![](https://im3.ezgif.com/tmp/ezgif-3-9d4af2fcecf7.gif)

44에폭부터 이렇게 Generator가 완전 맛이 가버림...
loss
```
[Epoch 44/100] [D loss: 1.7774] [G loss: 0.6982]
[Epoch 45/100] [D loss: 0.0000] [G loss: 14.7746]
[Epoch 46/100] [D loss: 0.0001] [G loss: 12.6780]
[Epoch 47/100] [D loss: 0.0000] [G loss: 13.6344]
[Epoch 48/100] [D loss: 0.0000] [G loss: 30.7463]
[Epoch 49/100] [D loss: 0.0000] [G loss: 53.4993]
[Epoch 50/100] [D loss: 0.0000] [G loss: 53.8629]
[Epoch 51/100] [D loss: 0.0000] [G loss: 80.5456]
[Epoch 52/100] [D loss: 0.0000] [G loss: 68.9882]
[Epoch 53/100] [D loss: 0.0000] [G loss: 80.7234]
[Epoch 54/100] [D loss: 0.0000] [G loss: 73.4675]
[Epoch 55/100] [D loss: 0.0000] [G loss: 60.6370]
[Epoch 56/100] [D loss: 0.0000] [G loss: 77.4076]
[Epoch 57/100] [D loss: 0.0000] [G loss: 81.6535]
[Epoch 58/100] [D loss: 0.0000] [G loss: 74.7338]
[Epoch 59/100] [D loss: 0.0000] [G loss: 77.7797]
[Epoch 60/100] [D loss: 0.0000] [G loss: 79.5218]
[Epoch 61/100] [D loss: 0.0000] [G loss: 79.2897]
[Epoch 62/100] [D loss: 0.0000] [G loss: 78.8147]
[Epoch 63/100] [D loss: 0.0000] [G loss: 64.2516]
[Epoch 64/100] [D loss: 0.0000] [G loss: 79.5293]
[Epoch 65/100] [D loss: 0.0000] [G loss: 75.6623]
[Epoch 66/100] [D loss: 0.0000] [G loss: 72.8966]
```


## Reference

- [Pytorch DCGAN Tutorial](https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html)