# 第四次作业说明 HW4

## 任务简述

### 实现变分自编码器(VAE)

要求学生使用PyTorch实现一个VAE来学习手写数字的MNIST数据集的概率模型。其中包括实现重参数化技巧，负ELBO（证据下界）的计算，以及在测试子集上评估平均负ELBO、KL项和重构损失。



---

## Usage (使用方法)

请大家稍微探索一下这些文件的用法。作业要求需要改的文件有：

1. `codebase/utils.py`
1. `codebase/models/vae.py`
1. `codebase/models/gmvae.py`
run_vae.py


不要修改其他文件。我们已经设置好了所有默认超参数，大家可以不要改

这些模型可能需要一段时间才能在CPU上运行，因此请做好相应的准备。在Macbook Pro，运行`vae.py`和`gmvae.py`各需约几分钟。
运行`ssvae.py`大约需要几十分钟。`fsvae.py`需要很长时间才能完成，在50000次迭代左右就可以停止训练了，
不然确实太久了哈哈哈。请注意，我们会定期为您保存模型。

您也可以看情况创建新文件或jupyter notebook来帮你搞事情。小提示：你可能会觉得这些函数有用:

1. `codebase.utils.load_model_by_name` (用于加载模型。请参阅`run_vae.py`中的示例用法)
1. `vae.py`/`gmvae.py`/`ssvae.py`/`fsvae.py`中的采样
1. `numpy.swapaxes` and/or `torch.permute` (用于在表示为numpy数组时平铺图像)
1. `matplotlib.pyplot.imshow` (从numpy数组创建图片)

以下是你需要在代码库中实现的各种函数，按时间顺序排列:

1. `sample_gaussian` in `utils.py`
1. `negative_elbo_bound` in `vae.py`
1. `log_normal` in `utils.py`
1. `log_normal_mixture` in `utils.py`
1. `negative_elbo_bound` in `gmvae.py`
1. `negative_elbo_bound` in `ssvae.py`
1. `negative_elbo_bound` in `fsvae.py` (bonus)

请大家要填的代码写在`待办`下面的一堆井号和`# 代码修改结束`上面一行的一堆井号中间。

你写完这几个函数并且没问题之后, 可以用git bash/bash (linux)/zsh (linux) 打开本文件夹，运行以下代码然后上传`hw4.zip`：
```
zip -r hw4.zip codebase/utils.py codebase/models/vae.py codebase/models/gmvae.py codebase/models/ssvae.py codebase/models/fsvae.py
```

如果没有git bash之类的那就手动打包~

大家有问题就在群里提出。作业看着很多但是真的不难，希望大家能认真做，然后希望大家玩数学和编程能玩的开心！期待大家提交的作业:-)

---

## Dependencies

代码只需要一下几个库，默认的版本应该是没有问题的，如有问题请在群里提出~ 

```
tqdm
numpy
torchvision
torch
matplotlib
```
