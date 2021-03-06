# README


## *源码的运行环境为Matlab*

在文件learn1_1000.m文件中，源码中定义了一个贝叶斯网，其中共有四个结点，分别为C、S、R和W，构成了一张有向无环图，各节点之间的其关系如下：


```
dag(C,[R S]) = 1;
```

```
dag(R,W) = 1;
```

```
dag(S,W)=1;
```

各节点CPT值如下：

*C结点CPT值*
 
 |   |   Ground truth
 ---- | -----  
 C(0)  | 0.5  
 C(1)  | 0.5  
 
 *R结点CPT值*
 
 |    | C(0)  | C(1)
 ---- | ----- | ------  
 R(0)  | 0.8 | 0.2 
 R(1)  | 0.2 | 0.8 
 
  *S结点CPT值*
 
 |    | C(0)  | C(1)
 ---- | ----- | ------  
 S(0)  | 0.5 | 0.9 
 S(1)  | 0.5 | 0.1 
 
   *W结点CPT值*
 
 |  R  | R(0) | R(1) | R(1) | R(1) 
  | ---- | ----- | ------ | ------| ------
 |  S  | S(0) | S(1) | S(0) | S(1) 
 | W(0)  | 1.0 | 0.9 | 0.1 | 0.01 
 | W(1)  | 0.0 | 0.1 | 0.9 | 0.99 
 
第一次运行将创建1000个随机数据集，并得出个CPT值
 
通过导入data3000.mat和data5000.mat输出CPT值
 
***一共会输出两个CPT值，分别为最大似然估计`learn_params()`和贝叶斯方法`bayes_update_params()`*** 
 
***输出时间为耗时统计，单位是ms***
 
