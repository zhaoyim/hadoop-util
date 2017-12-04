# hadoop-util
此项目包含两部分内容：<br/>
1. 收集hadoop集群的集群、队列以及job信息。都是通过restfulw接口获取， <br/>
   请先确定是否这些restful接口正确存在。具体接口下面给出解释。<br/>
2. 利用采集到的集群与队列信息进行LSTM模型的训练 <br/>

## hadoop集群的信息采集：
### 修改conf文件夹下的properties.conf文件：
* hadoop_url: hadoop集群信息获取的restful url
* application_url: hadoop集群中applications信息获取的restful url
* job_url: mapreduce的job信息获取的restful url
* job_metrices: 根据所列内容对获取到的spark job的信息内容进行提取相应的字段
### 运行hadoop.py脚本：
获取帮助请运行 python hadoop.py --help, 下面给出一些重要解释：<br/>
* --file_path ：采集到的信息输出目录：其中cluster信息保存在$file_path/cluster.csv<br/>
此文件写入参数为'a',保存所有历史数据。同时还有$file_path/cluster2.csv写入参数为'w'，只保存当前信息。<br/>
* --time_format与--time_interval需联合使用，time_format指定时间单位:周，天，小时，分钟，秒。<br/>
而time_interval则为int型数字，time_format为m，time_interval为10则指定采集最新10s的application信息。<br/>
当time_interval<0时则time_format不起作用，采集所有的application信息。<br/>
* --state 则指定要采集的application状态:finished, accepted, running<br/>
* --time_period: 指定采集周期，单位为s。<br/>
## LSTM模型的训练与使用
  python train_lstm.py --help 获取更多帮助。
* --time_period: 模型迭代训练周期。
* --train_step: 训练step
* -- predict_step: 预测step <br/>
### 
模拟训练请运行python test/lstm_test.py
