#!/bin/bash

# for seed in 1 11 21 31 41
# do
#    echo "Running with seed: $seed"
#    python main.py --config=a2r_smac --env-config=sc2 with name="no_vae_tta_instrinc_smpe" seed=$seed evaluate=True use_tta=True disable_belief=False checkpoint_path="/home/waq/robust_marl/results/a2r_smpe/models/add_belief_intrinstic_seed13_4m_vs_3m_2026-03-02 16:19:41.956021/"
# done

# for seed in 1 11 21 31 41
# do
#    echo "Running with seed: $seed   "
#    python main.py --config=a2r_smac --env-config=sc2 with name="b_smpe_bsm" seed=$seed evaluate=True use_tta=False  disable_belief=False checkpoint_path="/home/waq/robust_marl/results/belief_smpe/models/smpe_seed9_4m_vs_3m_2026-02-08 21:27:56.130732/"
# done


# for seed in 1 3 4 5 7 9 13 15 16 18 19 
# do 
#    echo "Running with seed: $seed   "
#    python main.py --config=a2r_smac --env-config=sc2 with name="SABER" evaluate=True use_tta=True disable_belief=False  checkpoint_path="/home/waq/robust_marl/results/a2r_smpe/models/add_belief_intrinstic_seed13_7m_vs_6m_2026-03-03 12:07:51.528046/" seed=$seed
# done

for seed in  1 2 6 7 9 11 12 15 17 19 21
do 
   python main.py --config=a2r_smac --env-config=sc2 with name="SABER" evaluate=True use_tta=True disable_belief=False use_belief_in_filter=True   checkpoint_path="/home/waq/robust_marl/results/a2r_smpe/models/add_belief_intrinstic_seed13_7m_vs_6m_2026-03-03 12:07:51.528046/" seed=$seed
done