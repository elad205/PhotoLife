# Introduction
this is a project that I have developed for my final
project at high school.  
I have strated from scracth with no knowlage about 
machine learning and moved my way up during research and development.   
I am setisfied with  my final result but I will still try to improve things.

# Usage
## Neural network usage
in order to use the gans network:  
```
python main.py <mode> <batch_size>
```  
there are 4 avilable modes:   
- train mode: used for training the model  
- test mode: used for testing the model  
- eval mode: used for colorizing single or few images  
- standby mode: a mode where the script waits for files from the stdin, colorizes them, rinse and repeat.  
**extra usage**
```
python main.py <test dataset> <train dataset>
<epochs>
<learning rate> <visdom ip and port> <decy lr> <checkpoint> <save weights> <save location> <beat0>  <eval images>
```  

## Server usage
the server is a simple server built with flask. Its
purpose is to make the network accessible and easy to use.
 
```
python server.py <host> <port> <upload loc> <save loc> <checkpoint>
```


# Built with

[PyTorch](https://pytorch.org/) - a machine learning framework  
[Flask](http://flask.pocoo.org/) - open source web application framework  
[torchvision](https://pytorch.org/docs/stable/torchvision/index.html) - a  package that consists of popular datasets, model architectures, and common image transformations for computer vision.  

# Exampales
![image1](/tests/tests_black/test8.jpg?raw=true)
![image1r](/tests/test/images/colored/r8.jpg?raw=true

# Credits
Thanks to @Or Gani for helping me with the amazing designs for this projects.

I also want to thank my mentor Shai who helped me through the whole project.

During my project I have taken insperation from a number of projects and I would like to mention them here:

https://arxiv.org/pdf/1803.05400.pdf  
https://github.com/jantic/DeOldify
