# Setup
DATATYPE=msrvtt
N_GPU=1
N_THREAD=8

# PATH to files
DATA_PATH=../dataset/MSRVTT/MSRVTT_data.json
CKPT_ROOT=../ckpts
INIT_MODEL_PATH=../weight/univl.pretrained.bin
FEATURES_PATH=<CLIP FEATURE FOR Theta_2> # Change into the path to the features you extracted from CLIP4Clip
DATA_GEOMETRIC_PATH=<GRAPH FEATURE for Theta_1> # Change into the path to the graph-based features (can be grid or object-based features)
NODE_FEATURES=geometric # please only use geometric for now
# Params
LEARNING_RATE=(3e-4)

for lr in "${LEARNING_RATE[@]}"
do
  python -m torch.distributed.launch --nproc_per_node=${N_GPU} \
  ../main_task_caption_GNN.py --do_train --num_thread_reader=${N_THREAD} \
  --epochs=50 --batch_size=1024 --n_display=50 --gradient_accumulation_steps 1 \
  --data_path ${DATA_PATH} --features_path ${FEATURES_PATH} \
  --output_dir ${CKPT_ROOT}/${DATATYPE}_lr${lr}_gnn \
  --bert_model bert-base-uncased --do_lower_case \
  --lr ${lr} --max_words 48 --max_frames 20 --batch_size_val 128 \
  --visual_num_hidden_layers 2 --decoder_num_hidden_layers 2 \
  --datatype ${DATATYPE} --init_model ${INIT_MODEL_PATH} \
  --data_geometric_path ${DATA_GEOMETRIC_PATH} \
  --node_features ${NODE_FEATURES} --node_feat_dim 512 --d_model 512 --video_dim 512 --edge_dim 1024 \
  --tradeoff_theta_2 4 --tradeoff_distill 1 --gnn_model_type transformer \
done
