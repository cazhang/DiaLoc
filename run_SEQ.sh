CUDA_VISIBLE_DEVICES=0 python train_SEQ.py \
--config ./configs/seq_config.yaml \
--data_dir ./led_data/way_splits/ \
--image_dir ./led_data/floorplans/ \
--embedding_dir ./led_data/word_embeddings/ \
--connect_dir ./led_data/connectivity/ \
--geodistance_file ./led_data/geodistance_nodes.json \
--discount 0.0 \
--aux_w 1.0
