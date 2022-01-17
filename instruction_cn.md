## 人智大作业二：表情识别
将data.h5文件放入data文件夹下，data.h5在如下链接中。
https://cloud.tsinghua.edu.cn/f/02dd63f57cc947518c46/

第一问会将50个epoch的checkpoint存入ResNet50_ckpt文件夹下，并依次进行测试。效果较好的模型是第44epoch，在如下链接中，将其放入ResNet50_ckpt文件夹下，修改main.py的主函数，即可进行单独的测试。
checkpoint_epoch_44.pth
https://cloud.tsinghua.edu.cn/f/afb4a704d756459ebe2b/

第二问做了一点改进，效果较好的模型是第48个epoch，在如下链接中，将其放入ResNet50_ckpt文件夹下，修改main.py的主函数，即可进行单独的测试。
checkpoint_epoch_48.pth
https://cloud.tsinghua.edu.cn/f/986103237aa94b49a953/

第三问需要将checkpoint_epoch_44.pth放入interface文件夹下。打开interface文件夹，运行interface.py即可看到图象的效果，将py文件中相应的代码取消注释，运行interface.py即可看到视频的效果。