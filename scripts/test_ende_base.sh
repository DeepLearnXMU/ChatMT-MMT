code_dir=thumt
code_dir=thumt-step3
work_dir=/pathto/chatnmt/THUMT-mono
train_dir=$work_dir/data
vocab_data_dir=/pathto/chatnmt/ende
data_dir=/pathto/chatnmt/ende/big_order_en_de_ctx3
export PYTHONPATH=$work_dir/$code_dir:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
data_name=test
checkpoint_dir=models/model_base_ende_stage3_sp0.7_dia0.5_kl5000/eval
Step="275454"
for idx in $Step
do
    echo model_checkpoint_path: \"model.ckpt-$idx\" > $work_dir/$checkpoint_dir/checkpoint
    cat $work_dir/$checkpoint_dir/checkpoint
    echo decoding with checkpoint-$idx
    python $work_dir/$code_dir/thumt/bin/translator.py \
        --models transformer \
        --checkpoints $work_dir/$checkpoint_dir \
        --input $data_dir/"$data_name".tok.bpe.32000.en \
        --output $data_dir/"$data_name".out.de.$idx \
        --vocabulary $vocab_data_dir/ende.bpe32k.vocab4.txt $vocab_data_dir/ende.bpe32k.vocab4.txt \
        --dev_dialog_src_context $data_dir/test_ctx.tok.bpe.32000.en \
        --parameters=decode_batch_size=64
    echo evaluating with checkpoint-$idx
#    cd $train_dir
    sed -r "s/(@@ )|(@@ ?$)//g" $data_dir/"$data_name".out.de.$idx > $data_dir/${data_name}.out.de.delbpe.$idx
    $data_dir/multi-bleu.perl $data_dir/"$data_name".tok.de < $data_dir/${data_name}.out.de.delbpe.$idx
    #cd $work_dir
    echo finished of checkpoint-$idx
done
