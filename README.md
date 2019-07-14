# Face-Verification-based-on-DeepID-and-Joint-Beyesian
Face Verification based on DeepID and Joint Beyesian 

## 0、背景
这个项目是我研究生课程《机器学习》的大作业。主要是做人脸验证：输入两幅人脸图像，判断是否属于同一个人。


## 1、整体流程：
> 1\) 利用DeepID提取每幅人脸图像的特征，分别得到两幅输入的人脸图像的特征x1和x2.

> 2\) 利用联合贝叶斯来根据特征x1和x2判断是否是否属于同一个人。

## 2、联合贝叶斯算法：       
![image](https://github.com/ShirleyGxd/Face-Verification-based-on-DeepID-and-Joint-Beyesian/blob/master/ImagesForReadMeFile/Beiyes1.PNG)
![image](https://github.com/ShirleyGxd/Face-Verification-based-on-DeepID-and-Joint-Beyesian/blob/master/ImagesForReadMeFile/Beiyes2.PNG)

## 3、使用说明
> 1\) 安装caffe

> 2\) 将data文件夹内的same_pairs.txt的内容改为待测试的属于同一人的图像对的地址，diff_pairs.txt的内容改为待测试的不属于同一人的图像对的地址

> 3\) 运行src文件夹内的verif.py文件 

## 参考文献

Sun Y, Wang X, Tang X. Deep learning face representation from predicting 10,000 classes[C]//Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition. 2014: 1891-1898.

Chen D, Cao X, Wang L, et al. Bayesian face revisited: a joint formulation[C]// European Conference on Computer Vision. Springer-Verlag, 2012:566-579.
