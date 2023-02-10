code_dir=thumt-step2
work_dir=$PWD
work_dir=/pathto/THUMT-mono
data_dir=/pathto/ende
test_data_dir=/pathto/big_order_en_de_ctx3
train_data_dir=$data_dir/ende4mono/big_order_en_de_ctx3/
#sp=$1
dia=$1
sp=$2
idx=$3
export PYTHONPATH=$work_dir/$code_dir:$PYTHONPATH
python $work_dir/$code_dir/thumt/bin/trainer.py \
  --model transformer \
  --output $work_dir/models/model_base_ende_stage2_sp${sp}_dia${dia}_mono \
  --input $data_dir/en2de.en.nodup.norm.tok.clean.bpe $data_dir/en2de.de.nodup.norm.tok.clean.bpe \
  --vocabulary $data_dir/ende.bpe32k.vocab4.txt $data_dir/ende.bpe32k.vocab4.txt \
  --validation $test_data_dir/dev.tok.bpe.32000.en \
  --references $test_data_dir/dev.tok.de \
  --dialog_en $train_data_dir/dialog.en.tok.bpe \
  --dialog_ctx_en $train_data_dir/dialog_ctx.en.tok.bpe  \
  --dialog_en_label $train_data_dir/self-dialogs.json_en_dia_labels.txt \
  --speaker_en $train_data_dir/speaker.en.tok.bpe \
  --speaker_ctx_en $train_data_dir/speaker_ctx.en.tok.bpe \
  --speaker_en_label $train_data_dir/self-dialogs.json_en_sp_labels.txt \
  --dialog_de $train_data_dir/dialog.de.tok.bpe \
  --dialog_ctx_de $train_data_dir/dialog_ctx.de.tok.bpe  \
  --dialog_de_label $train_data_dir/self-dialogs.json_de_dia_labels.txt \
  --speaker_de $train_data_dir/speaker.de.tok.bpe \
  --speaker_ctx_de $train_data_dir/speaker_ctx.de.tok.bpe \
  --speaker_de_label $train_data_dir/self-dialogs.json_de_sp_labels.txt \
  --parameters=device_list=[0,1,2,3],update_cycle=1,eval_steps=50,train_steps=276000,batch_size=2048,max_length=128,constant_batch_size=False,residual_dropout=0.1,attention_dropout=0.1,relu_dropout=0.1,shared_source_target_embedding=True,hidden_size=512,filter_size=2048,num_heads=8,max_relative_dis=16,use_speaker=True,use_dialogue=True,sp_alpha=${sp},dia_alpha=${dia}
