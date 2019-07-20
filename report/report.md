- Temporarily stop diacritics_19-05-28_13_14_57
+ Model `Seq2Seq`

```buildoutcfg
INFO:root:Step: 250000
INFO:root:Number of batchs: 157
INFO:root:w_a: 0.3831±0.0231 	 s_a: 0.0082±0.0154 	 Duration: 33.4596 s/step
INFO:root:New best score: 0.38311907877453927
INFO:root:Saved model at /source/main/train/output/saved_models//Seq2Seq/2019-05-28T11:15:09
INFO:root:Current best score: 0.38311907877453927 recorded at step 250000

```

- Bug in inference of seq2seq_attn. Stop `diacritics_19-05-29_18_55_07`
+ Saved at: `/Seq2SeqAttn/2019-05-29T16:55:22`, step 30000

- Still going down, but need stopping. Stop `diacritics_19-06-01_18_23_30`
+ 
```buildoutcfg
------------------ 	Evaluation	------------------
INFO:root:Step: 470000
INFO:root:Number of batchs: 157
INFO:root:w_a: 0.9735±0.0059 	 s_a: 0.3734±0.0824 	 Duration: 37.7257 s/step
INFO:root:New best score: 0.9734695462548292
INFO:root:Saved model at /source/main/train/output/saved_models//Seq2SeqAttnWithSrc/2019-06-01T16:23:45
INFO:root:Current best score: 0.9734695462548292 recorded at step 470000
INFO:root:Sample 1 

```


- Temporarliy stop `diacritics_19-06-05_18_09_26`
------------------ 	Evaluation	------------------
INFO:root:Step: 90000
INFO:root:Number of batchs: 157
INFO:root:w_a: 0.9535±0.0084 	 s_a: 0.1992±0.0772 	 Duration: 38.0996 s/step
INFO:root:New best score: 0.9534970709813919
INFO:root:Saved model at /source/main/train/output/saved_models//Seq2SeqChunk/2019-06-05T16:09:41
INFO:root:Current best score: 0.9534970709813919 recorded at step 90000


- Temporarily stop `diacritics_19-06-06_12_01_28` because its coverage was too slow
+ Monitoring: Seq2SeqChunk/2019-06-06T10:01:42
```buildoutcfg
------------------ 	Evaluation	------------------
INFO:root:Number of batchs: 157
INFO:root:New best score: 0.9621806538241117
INFO:root:Saved model at /source/main/train/output//saved_models/Seq2SeqChunk/2019-06-06T10:01:42/210000.pt
INFO:root:Current best score: 0.9621806538241117 recorded at step 210000

```


- Stop container: diacritics_19-06-14_19_03_47
```
------------------ 	Evaluation	------------------
INFO:root:Number of batchs: 157
INFO:root:New best score: 0.8941512289477335
INFO:root:Saved model at /source/main/train/output//saved_models/Seq2SeqFeedingAttn/2019-06-14T17:04:06/220000.pt
INFO:root:Current best score: 0.8941512289477335 recorded at step 220000

```
- Temporarily stop `diacritics_19-06-20_15_49_56`
```buildoutcfg
------------------      Evaluation      ------------------
INFO:root:Number of batchs: 79
INFO:root:New best score: 0.9152970683550201
INFO:root:Saved model at /source/main/train/output//saved_models/MainModel/2019-06-20T13:50:12/40000.pt
INFO:root:Current best score: 0.9152970683550201 recorded at step 40000


```

- Stop `diacritics_19-06-20_15_49_56` overfitting
INFO:root:Current best score: 0.9384526835146263 recorded at step 150000

- Temporarily stop `diacritics_19-06-24_18_37_33`
+ `2019-06-24T16:37:50`
+ 
```buildoutcfg
------------------      Evaluation      ------------------
INFO:root:Number of batchs: 79
INFO:root:New best score: 0.9685444852313875
INFO:root:Saved model at /source/main/train/output//saved_models/Simple/2019-06-24T16:37:50/80000.pt
INFO:root:Current best score: 0.9685444852313875 recorded at step 80000
```
- Temporarily stop `diacritics_19-06-25_11_05_29`
+ 
+
```buildoutcfg
------------------      Evaluation      ------------------
INFO:root:Number of batchs: 79
INFO:root:New best score: 0.97243390156583
INFO:root:Saved model at /source/main/train/output//saved_models/Simple/2019-06-25T09:06:11/210000.pt
INFO:root:Current best score: 0.97243390156583 recorded at step 210000

```

- Stop `diacritics_19-06-26_06_02_00` because of over-fitting
NFO:root:Current best score: 0.973669634538235 recorded at step 360000


```yaml
------------------      Evaluation      ------------------
INFO:root:Number of batchs: 79
INFO:root:Current best score: 0.9751268797402585 recorded at step 730000
Simple/2019-06-27T16:33:36
```

- Stop `diacritics_19-07-01_08_43_03`
```yaml
2019-07-02T06:04:04.036735001Z ------------------       Evaluation      ------------------
2019-07-02T06:04:04.036739742Z INFO:root:Number of batchs: 79
2019-07-02T06:04:04.037234929Z INFO:root:Current best score: 0.9763880471105716 recorded at step 870000

```
Simple/2019-07-01T06:43:19


# Training Transformer
- Container: `diacritics_19-07-17_11_07_35`
- 
```yaml
------------------      Evaluation      ------------------
INFO:root:Number of batchs: 157
INFO:root:Current best score is 0.9706060603762169 at /source/main/train/output//saved_models/Model/1.0/185000.pt
INFO:root:Current best score: 0.9706060603762169 recorded at step 185000

```
- Dataset: `dataset.py`
- Log: `Model/1.0`

Temporarily stop to train with augmentation data

Temporarily stop `diacritics_19-07-18_17_26_33` (13h)
2019-07-19T04:14:09.509074488Z ------------------       Evaluation      ------------------
2019-07-19T04:14:09.509080059Z INFO:root:Number of batchs: 157
2019-07-19T04:14:09.510004842Z INFO:root:Current best score is 0.9717414296294907 at /source/main/train/output//saved_models/Model/1.1/225000.pt
2019-07-19T04:14:09.510025346Z INFO:root:Current best score: 0.9717414296294907 recorded at step 225000

- Stop `diacritics_19-07-19_14_09_16`
Because loss of train are not reducing anymore
```yaml
------------------      Evaluation      ------------------
INFO:root:Number of batchs: 157
INFO:root:Current best score is 0.9738843542930108 at /source/main/train/output//saved_models/Model/1.2/360000.pt
INFO:root:Current best score: 0.9738843542930108 recorded at step 360000
Loss ~= 0.10
```
