CUDA_VISIBLE_DEVICES=0 python train_LED.py \
--config ./configs/led_config.yaml \
--data_dir ./led_data/way_splits/ \
--image_dir ./led_data/floorplans/ \
--embedding_dir ./led_data/word_embeddings/ \
--connect_dir ./led_data/connectivity/ \
--geodistance_file ./led_data/geodistance_nodes.json
