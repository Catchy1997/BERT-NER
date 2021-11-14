CURRENT_DIR=`pwd`

# # python -m torch.distributed.launch infer_query.py \
# python infer_query.py \
#   --best_model=prev_trained_model/query_v1/best_model_query.model \
#   --num_labels=7 \
#   --base_path=prev_trained_model/bert-base-chinese \
#   --seed=42 \
#   --output_dir=outputs/infer \
#   --overwrite_output_dir=False \
#   --markup=bios \
#   --data_dir=datasets/query_infer \
#   --max_seq_length=128 \
#   --str_name=caption_pro \
#   --batch_size=192 \
#   --input_dir=datasets/query_infer/test.json \
#   --local_rank=-1

# python infer_trans.py \
#   --pred_path=outputs/infer/test-0.json \
#   --text_path=datasets/query/test-0.json \
#   --pred_str_name=ocr_process \
#   --true_str_name=ocr_result \
#   --save_path=outputs/infer/test-0-ocr.json

python infer_trans.py \
  --pred_path=outputs/query_output/bert/test_prediction.json \
  --text_path=datasets/query/test.json \
  --pred_str_name=caption_pro \
  --true_str_name=caption \
  --save_path=/home/zhanglin12/BERT-NER/datasets/query/data/general_submit.json

# python infer_online.py \
#   --best_model=best_model.model \
#   --num_labels=7 \
#   --base_path=prev_trained_model/bert-base-chinese \
#   --seed=42 \
#   --output_dir=outputs/infer \
#   --overwrite_output_dir=False \
#   --markup=bios \
#   --data_dir=datasets/query \
#   --max_seq_length=128 \
#   --str_name=cover_ocr_process \
#   --batch_size=192 \
#   --input_dir=datasets/query/test-50.json \
#   --local_rank=-1