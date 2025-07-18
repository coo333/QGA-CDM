# 

## Configuring Running Environment
> pip install -r requirements.txt

## Running QGA-CDM
- Start training and testing QGA-CDM:
> python run.py --train_file data/math1_train_0.8_0.2.csv --valid_file data/math1_valid_0.8_0.2.csv --test_file data/math1_test_0.8_0.2.csv --Q_matrix data/math1_Q_matrix.npy --save_path ./result/ID-CDM-Math1 --n_user 4209 --n_item 20 --n_know 11 --user_dim 32 --item_dim 32 --batch_size 32 --lr 0.0005 --epoch 5 --device cpu
>python run.py --train_file data/math1_train_0.8_0.2.csv --valid_file data/math1_val id_0.8_0.2.csv --test_file data/math1_test_0.8_0.2.csv --Q_matrix data/math1_Q_matrix.npy --save_path ./result/ID-CDM-Math1 --n_user 4209 --n_item 20 --n_k
now 11 --user_dim 32 --item_dim 32 --batch_size 32 --lr 0.0001 --epoch 5 --device cpu --q_aug single --lambda_q 0.01
>python run.py --train_file data/a2017_train_0.8_0.2.csv --valid_file data/a2017_valid_0.8_0.2.csv --test_file data/a2017_test_0.8_0.2.csv --Q_matrix data/a2017_Q_matrix.npy --save_path ./result/ID-CDM-Math1 --n_user 1678 --n_item 2210 --n_know 101 --user_dim 32 -
-item_dim 32 --batch_size 32 --lr 0.0001 --epoch 10 --device cpu
> 
> python run.py --train_file data/junyi_train_0.8_0.2.csv --valid_file data/junyi_valid_0.8_0.2.csv --test_fil
e data/junyi_test_0.8_0.2.csv --Q_matrix data/junyi_Q_matrix.npy --save_path ./junyiresult --n_user 10000 --n_item 734 --n_know 734 --user_dim 32 --item_di
m 32 --batch_size 32 --lr 0.00009 --epoch 30 --device cpu --q_aug single --lambda_q 0.008
> 
> python ga_qsearch.py  --train_file data/junyi_train_0.8_0.2.csv  --valid_file data/junyi_valid_0.8_0.2.csv  
--test_file data/junyi_test_0.8_0.2.csv  --save_path ./junyiresult/Top20QGA_Qsearch_fixed  --n_user 10000  --n_item 734  --n_know 734  --user_dim 32  --ite
m_dim 32  --batch_size 32  --lr 0.00009  --ga_small_epoch 5  --ga_pop_size 100  --ga_ngen 15  --ga_cxpb 0.5  --ga_mutpb 0.2  --device cuda --init_q_file da
ta/junyi_Q_matrix.npy
