# VAE
### 环境配置
并没有用到额外的包，可以将其它任务的环境重命为common

或者运行如下脚本
```bash
conda env create -f environment.yml
conda activate common
```

### 代码:
主体代码在main.py中

可视化代码在visualize.py中

### 运行:
```bash
sbatch run.sh
```
可根据下面说明选择run.sh中想要的部分
#### Baseline
run.sh中的下面这部分
```angular2html
echo "Baseline"
echo "--batch_size 64 --z_dim 1 --l1 2 --epoch 100"
python -u main.py --cuda --batch_size 64 --z_dim 1 --l1 1 --epoch 100 --save z_dim_2
echo "--batch_size 64 --z_dim 2 --l1 1 --epoch 100"
python -u main.py --cuda --batch_size 64 --z_dim 2 --l1 1 --epoch 100 --save z_dim_1
```
#### 添加噪声的尝试（DVAE）
只需要在脚本中加入
```
--add_noise
```
对应run.sh中的下面这部分
```angular2html
echo "DVAE"
echo "Add_noise --batch_size 64 --z_dim 1 --l1 2 --epoch 100"
python -u main.py --cuda --batch_size 64 --z_dim 1 --l1 1 --epoch 100 --save z_dim_2 --add_noise
echo "Add_noise --batch_size 64 --z_dim 2 --l1 1 --epoch 100"
python -u main.py --cuda --batch_size 64 --z_dim 2 --l1 1 --epoch 100 --save z_dim_1 --add_noise
```
#### 添加BN的尝试（BN-VAE）
只需要在脚本中加入
```
--add_BN
```
对应run.sh中的下面这部分
```angular2html
echo "BN VAE"
echo "Add_BN --batch_size 64 --z_dim 1 --l1 2 --epoch 100"
python -u main.py --cuda --batch_size 64 --z_dim 1 --l1 1 --epoch 100 --save z_dim_2 --add_BN
echo "Add_BN --batch_size 64 --z_dim 2 --l1 1 --epoch 100"
python -u main.py --cuda --batch_size 64 --z_dim 2 --l1 1 --epoch 100 --save z_dim_1 --add_BN
```
