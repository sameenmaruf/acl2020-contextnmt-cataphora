For pre-training the sentence-level transformer: (Here the data files should be in format source ||| target and --model-path provides path to folder where model files will be save)

./build_gpu/transformer-train --dynet_mem 15500 --minibatch-size 1500 --max-seq-len 80 --treport 100000 --dreport 500000 -t $trainfname -d $devfname --model-path $modelfname \
--sgd-trainer 4 --lr-eta 0.0001 -e 35 --patience 15 --use-label-smoothing --encoder-emb-dropout-p 0.1 --encoder-sublayer-dropout-p 0.1 --decoder-emb-dropout-p 0.1 \
--decoder-sublayer-dropout-p 0.1 --attention-dropout-p 0.1 --ff-dropout-p 0.1 --ff-activation-type 1 --nlayers 4 --num-units 512 --num-heads 8

For computing encoder-based representations: (Here input_doc is in format docID ||| source, input_type denotes the portion of data i.e. 0 for train, 1 for dev)

./build_gpu/transformer-computerep --dynet-mem 15500 --model-path $modelfname --input_doc $fname --input_type 0

For training the document-level model with HAN encoder: (Here the data files should be in format docID ||| source ||| target, --model-file is to give the model a name of your choice,
--context-type 1 means use the next sentence as context, 2 for previous, 3 for previous+next, 4 for two next sentences, 5 for two previous sentences, 6 for two previous and two next sentences)

./build_gpu/transformer-context --dynet_mem 15500 --minibatch-size 1000 --dtreport 150 --ddreport 750 --update-steps 5 --train_doc $trainfname --devel_doc $devfname \
--model-path $modelfname --model-file $modelname --context-type 1 --use-new-dropout --encoder-emb-dropout-p 0.2 --encoder-sublayer-dropout-p 0.2 --decoder-emb-dropout-p 0.2 \
--decoder-sublayer-dropout-p 0.2 --attention-dropout-p 0.2 --ff-dropout-p 0.2 --sgd-trainer 4 --lr-eta 0.0001 -e 25 --patience 15 

Decoding the sentence-level Transformer: (Here the test file contains only sourse sentences, only greedy decoding has been implemented)

./build_gpu/transformer-decode --dynet-mem 15500 --model-path $modelfname --beam 1 -T $testfname 

Decoding the document-level model with HAN encoder using single sentence as context: (The test file should be in the format: current_src ||| next_src if context-type is set to 1 (or 0 if decoding w/o context) 
or prev_src ||| current_src if context-type set to 2, for decoding the representations are computed on the fly)

./build_gpu/transformer-single-context-decode --dynet-mem 15500 --model-path $modelfname --model-file $modelname --beam 1 -T $testfname --context-type 1

Decoding the document-level model with HAN encoder using upto two sentences as context: (The test file should be in the format: docID ||| source, for decoding the representations are computed on the fly)

./build_gpu/transformer-context-decode --dynet-mem 15500 --model-path $modelfname --model-file $modelname --beam 1 -T $testfname --context-type 1
