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
