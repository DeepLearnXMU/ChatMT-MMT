code_dir=thumt-step1
work_dir=$PWD
data_dir=/path/to/ende
vocab_dir=$data_dir
export PYTHONPATH=$work_dir/$code_dir:$PYTHONPATH
model_name=en2de_model_base_share
python $work_dir/$code_dir/thumt/bin/trainer.py \
  --model transformer \
  --output $work_dir/models/$model_name \
  --input $data_dir/en2de.en.nodup.norm.tok.clean.bpe $data_dir/en2de.de.nodup.norm.tok.clean.bpe \
  --vocabulary $vocab_dir/ende.bpe32k.vocab4.txt $vocab_dir/ende.bpe32k.vocab4.txt \
  --validation $test_data_dir/dev.tok.bpe.32000.en \
  --references $test_data_dir/dev.tok.de \
  --parameters=device_list=[0,1,2,3],update_cycle=2,eval_steps=20000000,train_steps=200000,batch_size=4096,max_length=128,constant_batch_size=False,residual_dropout=0.1,attention_dropout=0.1,relu_dropout=0.1,hidden_size=512,filter_size=2048,num_heads=8,max_relative_dis=16,shared_source_target_embedding=True,save_checkpoint_steps=5000,keep_checkpoint_max=100

chmod 777 -R ${work_dir}/models/$model_name
