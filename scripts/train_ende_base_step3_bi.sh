code_dir=thumt-step3
work_dir=$PWD
work_dir=/pathto/chatnmt/THUMT-mono
data_dir=/pathto/chatnmt/ende
train_data_dir=$data_dir/ende4mono/big_order_en_de_ctx3
test_data_dir=/pathto/chatnmt/ende/big_order_en_de_ctx3
dia=0.8
sp=0.1
sp2=$1
dia2=$2
kl_annealing_steps=$3
kl_annealing_steps2=$kl_annealing_steps
#idx=$3
bi=.bi
export PYTHONPATH=$work_dir/$code_dir:$PYTHONPATH
python $work_dir/$code_dir/thumt/bin/trainer.py \
  --model transformer \
  --output $work_dir/models/model_base_ende_stage3_sp${sp2}_dia${dia2}_kl${kl_annealing_steps}_bi \
  --input $train_data_dir/train.tok.bpe.32000.en${bi} $train_data_dir/train.tok.bpe.32000.de${bi} \
  --vocabulary $data_dir/ende.bpe32k.vocab4.txt $data_dir/ende.bpe32k.vocab4.txt \
  --validation $test_data_dir/dev.tok.bpe.32000.en \
  --references $test_data_dir/dev.tok.de \
  --dev_dialog_src_context $test_data_dir/dev_ctx.tok.bpe.32000.en \
  --train_dialog_en $train_data_dir/train_dia.tok.bpe.32000.en${bi} \
  --train_dialog_ctx_en $train_data_dir/train_ctx.tok.bpe.32000.en${bi} \
  --train_dialog_en_label $train_data_dir/train_en_dia_label.txt${bi} \
  --train_speaker_ctx_en $train_data_dir/train_enper_ctx.tok.bpe.32000.en${bi} \
  --train_speaker_en_label $train_data_dir/train_enper_label.txt${bi} \
  --train_dialog_de $train_data_dir/train_dia.tok.bpe.32000.de${bi} \
  --train_dialog_ctx_de $train_data_dir/train_ctx.tok.bpe.32000.de${bi} \
  --train_dialog_de_label $train_data_dir/train_de_dia_label.txt${bi} \
  --train_speaker_ctx_de $train_data_dir/train_deper_ctx.tok.bpe.32000.de${bi} \
  --train_speaker_de_label $train_data_dir/train_deper_label.txt${bi} \
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
  --parameters=device_list=[0,1,2,3],update_cycle=1,eval_steps=50,train_steps=280000,start_steps=275103,batch_size=2048,max_length=128,constant_batch_size=False,residual_dropout=0.1,attention_dropout=0.1,relu_dropout=0.1,shared_source_target_embedding=True,hidden_size=512,filter_size=2048,num_heads=8,max_relative_dis=16,use_speaker=True,use_dialogue=True,sp_alpha=${sp},dia_alpha=${dia},kl_annealing_steps=${kl_annealing_steps},kl_annealing_steps2=${kl_annealing_steps2},sp_alpha2=${sp2},dia_alpha2=${dia2}
