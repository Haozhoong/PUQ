# for PUQ and PUQ w/o G
python train_twostep.py -c ./config/PUQ_recon.yaml
python pixels_gen.py -c ./config/PUQ_recon.yaml -t model_uq -d t1 -ut evar -s 100
python train_regress.py -c ./config/PUQ_fit.yaml

python pixels_gen.py -c ./config/PUQ_recon.yaml -t model -d t1 -ut evar -s 100
python train_regress.py -c ./config/PUQ_wo_fit.yaml

# MANTIS
python train_onestep.py -c ./config/MANTIS.yaml
# Dopamine
python train_onestep.py -c ./config/Dopamine.yaml
# Deept1
python train_twostep.py -c ./config/Deept1_recon.yaml
python train_twostep.py -c ./config/Deept1_fit.yaml