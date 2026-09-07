(mamba) root@5fe3ffc024a9:/workspace/nav_data/mamba/camrl/CrowdNav/crowd_nav# python test.py --gpu --episodes 100 --mamba_bias     --bias_alpha 0.30     --bias_stall_alpha 0.60     --bias_speed_floor 0.65     --bias_min_clear 0.22     --bias_min_ttc 1.6     --bias_progress_eps 0.04     --bias_late_ratio 0.55
[INFO] ========================================
[INFO] TEST.PY - Mamba Policy Testing
[INFO] ========================================
[INFO] Config: T=12, dt=0.25, v_max=1.0 (MATCHED TO TRAINING)
[INFO] Feature: TTC-Sorting enabled (与contracts.py训练时一致)
Using device: cuda

======================================================================
Test Case [0]: baseline_circle | 5 humans
======================================================================
[INFO] Loading Mamba policy: mamba
[INFO] Forced SARL-style prediction (use_sarl_predict=True)
[DEBUG] Policy Class: <class 'crowd_nav.policy.mamba_rl.MambaRLPolicy'>
100%|█████████████████████████████████████████| 100/100 [03:23<00:00,  2.04s/it, S=100/100, C=0/100]

Results for baseline_circle:
  SUCCESS:   100/100 (100.0%)
  COLLISION: 0/100 (0.0%)
  TIMEOUT:   0/100 (0.0%)
  TIME TAKEN (s): 12.61
  DISC. FREQ: 0.46
  DISC. DIST (m): 1.33

======================================================================
Test Case [1]: baseline_square | 10 humans
======================================================================
[INFO] Loading Mamba policy: mamba
[INFO] Forced SARL-style prediction (use_sarl_predict=True)
[DEBUG] Policy Class: <class 'crowd_nav.policy.mamba_rl.MambaRLPolicy'>
100%|██████████████████████████████████████████| 100/100 [04:42<00:00,  2.83s/it, S=79/100, C=3/100]

Results for baseline_square:
  SUCCESS:   79/100 (79.0%)
  COLLISION: 3/100 (3.0%)
  TIMEOUT:   18/100 (18.0%)
  TIME TAKEN (s): 15.51
  DISC. FREQ: 2.29
  DISC. DIST (m): 0.92

======================================================================
Test Case [2]: dense_circle | 10 humans
======================================================================
[INFO] Loading Mamba policy: mamba
[INFO] Forced SARL-style prediction (use_sarl_predict=True)
[DEBUG] Policy Class: <class 'crowd_nav.policy.mamba_rl.MambaRLPolicy'>
100%|██████████████████████████████████████████| 100/100 [04:31<00:00,  2.71s/it, S=97/100, C=0/100]

Results for dense_circle:
  SUCCESS:   97/100 (97.0%)
  COLLISION: 0/100 (0.0%)
  TIMEOUT:   3/100 (3.0%)
  TIME TAKEN (s): 16.02
  DISC. FREQ: 0.82
  DISC. DIST (m): 1.07

======================================================================
Test Case [3]: dense_square | 20 humans
======================================================================
[INFO] Loading Mamba policy: mamba
[INFO] Forced SARL-style prediction (use_sarl_predict=True)
[DEBUG] Policy Class: <class 'crowd_nav.policy.mamba_rl.MambaRLPolicy'>
100%|█████████████████████████████████████████| 100/100 [06:13<00:00,  3.73s/it, S=54/100, C=11/100]

Results for dense_square:
  SUCCESS:   54/100 (54.0%)
  COLLISION: 11/100 (11.0%)
  TIMEOUT:   35/100 (35.0%)
  TIME TAKEN (s): 18.48
  DISC. FREQ: 10.88
  DISC. DIST (m): 0.56

======================================================================
Test Case [4]: large_circle | 12 humans
======================================================================
[INFO] Loading Mamba policy: mamba
[INFO] Forced SARL-style prediction (use_sarl_predict=True)
[DEBUG] Policy Class: <class 'crowd_nav.policy.mamba_rl.MambaRLPolicy'>
100%|██████████████████████████████████████████| 100/100 [05:50<00:00,  3.51s/it, S=63/100, C=0/100]

Results for large_circle:
  SUCCESS:   63/100 (63.0%)
  COLLISION: 0/100 (0.0%)
  TIMEOUT:   37/100 (37.0%)
  TIME TAKEN (s): 18.75
  DISC. FREQ: 2.66
  DISC. DIST (m): 1.13

======================================================================
Test Case [5]: large_square | 20 humans
======================================================================
[INFO] Loading Mamba policy: mamba
[INFO] Forced SARL-style prediction (use_sarl_predict=True)
[DEBUG] Policy Class: <class 'crowd_nav.policy.mamba_rl.MambaRLPolicy'>
100%|█████████████████████████████████████████| 100/100 [05:18<00:00,  3.18s/it, S=70/100, C=12/100]

Results for large_square:
  SUCCESS:   70/100 (70.0%)
  COLLISION: 12/100 (12.0%)
  TIMEOUT:   18/100 (18.0%)
  TIME TAKEN (s): 17.02
  DISC. FREQ: 6.68
  DISC. DIST (m): 0.74
(mamba) root@5fe3ffc024a9:/workspace/nav_data/mamba/camrl/CrowdNav/crowd_nav# 


(mamba) root@5fe3ffc024a9:/workspace/nav_data/mamba/camrl/CrowdNav/crowd_nav# python test2.py --policy sarl --gpu --episodes 100
[INFO] ========================================
[INFO] TEST2.PY - SARL Baseline Testing
[INFO] ========================================
[INFO] Config: T=12, dt=0.25, v_max=1.0 (MATCHED TO TRAINING)
[INFO] Feature: TTC-Sorting enabled (与contracts.py训练时一致)
Using device: cuda

======================================================================
Test Case [0]: baseline_circle | 5 humans
======================================================================
[INFO] Loading SARL policy

[DEBUG] Policy Class: <class 'crowd_nav.policy.sarl.SARL'>
100%|██████████████████████████████████████████| 100/100 [26:59<00:00, 16.20s/it, S=99/100, C=1/100]

Results for baseline_circle:
  SUCCESS:   99/100 (99.0%)
  COLLISION: 1/100 (1.0%)
  TIMEOUT:   0/100 (0.0%)
  TIME TAKEN (s): 10.56
  DISC. FREQ: 0.75
  DISC. DIST (m): 1.17

======================================================================
Test Case [1]: baseline_square | 10 humans
======================================================================
[INFO] Loading SARL policy

[DEBUG] Policy Class: <class 'crowd_nav.policy.sarl.SARL'>
100%|█████████████████████████████████████████| 100/100 [17:46<00:00, 10.66s/it, S=88/100, C=11/100]

Results for baseline_square:
  SUCCESS:   88/100 (88.0%)
  COLLISION: 11/100 (11.0%)
  TIMEOUT:   1/100 (1.0%)
  TIME TAKEN (s): 9.47
  DISC. FREQ: 5.77
  DISC. DIST (m): 0.70

======================================================================
Test Case [2]: dense_circle | 10 humans
======================================================================
[INFO] Loading SARL policy

[DEBUG] Policy Class: <class 'crowd_nav.policy.sarl.SARL'>
100%|█████████████████████████████████████████| 100/100 [18:36<00:00, 11.16s/it, S=69/100, C=31/100]

Results for dense_circle:
  SUCCESS:   69/100 (69.0%)
  COLLISION: 31/100 (31.0%)
  TIMEOUT:   0/100 (0.0%)
  TIME TAKEN (s): 11.94
  DISC. FREQ: 4.85
  DISC. DIST (m): 0.69

======================================================================
Test Case [3]: dense_square | 20 humans
======================================================================
[INFO] Loading SARL policy

[DEBUG] Policy Class: <class 'crowd_nav.policy.sarl.SARL'>
100%|█████████████████████████████████████████| 100/100 [19:00<00:00, 11.40s/it, S=38/100, C=62/100]

Results for dense_square:
  SUCCESS:   38/100 (38.0%)
  COLLISION: 62/100 (62.0%)
  TIMEOUT:   0/100 (0.0%)
  TIME TAKEN (s): 11.46
  DISC. FREQ: 12.08
  DISC. DIST (m): 0.38

======================================================================
Test Case [4]: large_circle | 12 humans
======================================================================
[INFO] Loading SARL policy

[DEBUG] Policy Class: <class 'crowd_nav.policy.sarl.SARL'>
100%|█████████████████████████████████████████| 100/100 [30:40<00:00, 18.41s/it, S=82/100, C=18/100]

Results for large_circle:
  SUCCESS:   82/100 (82.0%)
  COLLISION: 18/100 (18.0%)
  TIMEOUT:   0/100 (0.0%)
  TIME TAKEN (s): 17.29
  DISC. FREQ: 5.82
  DISC. DIST (m): 1.09

======================================================================
Test Case [5]: large_square | 20 humans
======================================================================
[INFO] Loading SARL policy

[DEBUG] Policy Class: <class 'crowd_nav.policy.sarl.SARL'>
100%|█████████████████████████████████████████| 100/100 [22:12<00:00, 13.33s/it, S=59/100, C=41/100]

Results for large_square:
  SUCCESS:   59/100 (59.0%)
  COLLISION: 41/100 (41.0%)
  TIMEOUT:   0/100 (0.0%)
  TIME TAKEN (s): 9.59
  DISC. FREQ: 9.57
  DISC. DIST (m): 0.53
(mamba) root@5fe3ffc024a9:/workspace/nav_data/mamba/camrl/CrowdNav/crowd_nav# 


(mamba) root@5fe3ffc024a9:/workspace/nav_data/mamba/camrl/CrowdNav/crowd_nav# python test3.py --gpu --episodes 100
[INFO] ========================================
[INFO] TEST3.PY - LSTM Baseline Testing
[INFO] ========================================
[INFO] Config: T=12, dt=0.25, v_max=1.0 (MATCHED TO TRAINING)
[INFO] Feature: TTC-Sorting enabled (与contracts.py训练时一致)
Using device: cuda

======================================================================
Test Case [0]: baseline_circle | 5 humans
======================================================================
[INFO] Loading LSTM policy
[DEBUG] Policy Class: <class 'crowd_nav.policy.lstm_rl.LstmRL'>
100%|██████████████████████████████████████████| 100/100 [24:34<00:00, 14.75s/it, S=95/100, C=4/100]

Results for baseline_circle:
  SUCCESS:   95/100 (95.0%)
  COLLISION: 4/100 (4.0%)
  TIMEOUT:   1/100 (1.0%)
  TIME TAKEN (s): 11.22
  DISC. FREQ: 0.53
  DISC. DIST (m): 1.35

======================================================================
Test Case [1]: baseline_square | 10 humans
======================================================================
[INFO] Loading LSTM policy
[DEBUG] Policy Class: <class 'crowd_nav.policy.lstm_rl.LstmRL'>
100%|██████████████████████████████████████████| 100/100 [23:11<00:00, 13.91s/it, S=82/100, C=9/100]

Results for baseline_square:
  SUCCESS:   82/100 (82.0%)
  COLLISION: 9/100 (9.0%)
  TIMEOUT:   9/100 (9.0%)
  TIME TAKEN (s): 12.42
  DISC. FREQ: 2.54
  DISC. DIST (m): 1.14

======================================================================
Test Case [2]: dense_circle | 10 humans
======================================================================
[INFO] Loading LSTM policy
[DEBUG] Policy Class: <class 'crowd_nav.policy.lstm_rl.LstmRL'>
100%|█████████████████████████████████████████| 100/100 [20:38<00:00, 12.38s/it, S=89/100, C=11/100]

Results for dense_circle:
  SUCCESS:   89/100 (89.0%)
  COLLISION: 11/100 (11.0%)
  TIMEOUT:   0/100 (0.0%)
  TIME TAKEN (s): 13.64
  DISC. FREQ: 2.21
  DISC. DIST (m): 0.97

======================================================================
Test Case [3]: dense_square | 20 humans
======================================================================
[INFO] Loading LSTM policy
[DEBUG] Policy Class: <class 'crowd_nav.policy.lstm_rl.LstmRL'>
100%|█████████████████████████████████████████| 100/100 [36:39<00:00, 22.00s/it, S=65/100, C=23/100]

Results for dense_square:
  SUCCESS:   65/100 (65.0%)
  COLLISION: 23/100 (23.0%)
  TIMEOUT:   12/100 (12.0%)
  TIME TAKEN (s): 15.34
  DISC. FREQ: 4.72
  DISC. DIST (m): 0.82

======================================================================
Test Case [4]: large_circle | 12 humans
======================================================================
[INFO] Loading LSTM policy
[DEBUG] Policy Class: <class 'crowd_nav.policy.lstm_rl.LstmRL'>
100%|██████████████████████████████████████████| 100/100 [32:53<00:00, 19.73s/it, S=74/100, C=8/100]

Results for large_circle:
  SUCCESS:   74/100 (74.0%)
  COLLISION: 8/100 (8.0%)
  TIMEOUT:   18/100 (18.0%)
  TIME TAKEN (s): 20.03
  DISC. FREQ: 1.55
  DISC. DIST (m): 1.87

======================================================================
Test Case [5]: large_square | 20 humans
======================================================================
[INFO] Loading LSTM policy
[DEBUG] Policy Class: <class 'crowd_nav.policy.lstm_rl.LstmRL'>
100%|█████████████████████████████████████████| 100/100 [16:39<00:00,  9.99s/it, S=51/100, C=31/100]

Results for large_square:
  SUCCESS:   51/100 (51.0%)
  COLLISION: 31/100 (31.0%)
  TIMEOUT:   18/100 (18.0%)
  TIME TAKEN (s): 14.97
  DISC. FREQ: 3.18
  DISC. DIST (m): 1.19
(mamba) root@5fe3ffc024a9:/workspace/nav_data/mamba/camrl/CrowdNav/crowd_nav# 


(mamba) root@5fe3ffc024a9:/workspace/nav_data/mamba/camrl/CrowdNav/crowd_nav# python test4.py --gpu --episodes 100
[INFO] ========================================
[INFO] TEST4.PY - CADRL Baseline Testing
[INFO] ========================================
[INFO] Config: T=12, dt=0.25, v_max=1.0 (MATCHED TO TRAINING)
[INFO] Feature: TTC-Sorting enabled (与contracts.py训练时一致)
Using device: cuda

======================================================================
Test Case [0]: baseline_circle | 5 humans
======================================================================
[INFO] Loading CADRL policy
[DEBUG] Policy Class: <class 'crowd_nav.policy.cadrl.CADRL'>
100%|█████████████████████████████████████████| 100/100 [24:45<00:00, 14.85s/it, S=66/100, C=34/100]

Results for baseline_circle:
  SUCCESS:   66/100 (66.0%)
  COLLISION: 34/100 (34.0%)
  TIMEOUT:   0/100 (0.0%)
  TIME TAKEN (s): 11.19
  DISC. FREQ: 14.13
  DISC. DIST (m): 0.80

======================================================================
Test Case [1]: baseline_square | 10 humans
======================================================================
[INFO] Loading CADRL policy
[DEBUG] Policy Class: <class 'crowd_nav.policy.cadrl.CADRL'>
100%|█████████████████████████████████████████| 100/100 [22:13<00:00, 13.34s/it, S=64/100, C=35/100]

Results for baseline_square:
  SUCCESS:   64/100 (64.0%)
  COLLISION: 35/100 (35.0%)
  TIMEOUT:   1/100 (1.0%)
  TIME TAKEN (s): 10.66
  DISC. FREQ: 13.61
  DISC. DIST (m): 0.60

======================================================================
Test Case [2]: dense_circle | 10 humans
======================================================================
[INFO] Loading CADRL policy
[DEBUG] Policy Class: <class 'crowd_nav.policy.cadrl.CADRL'>
100%|█████████████████████████████████████████| 100/100 [16:57<00:00, 10.17s/it, S=22/100, C=78/100]

Results for dense_circle:
  SUCCESS:   22/100 (22.0%)
  COLLISION: 78/100 (78.0%)
  TIMEOUT:   0/100 (0.0%)
  TIME TAKEN (s): 13.00
  DISC. FREQ: 15.66
  DISC. DIST (m): 0.39

======================================================================
Test Case [3]: dense_square | 20 humans
======================================================================
[INFO] Loading CADRL policy
[DEBUG] Policy Class: <class 'crowd_nav.policy.cadrl.CADRL'>
100%|█████████████████████████████████████████| 100/100 [29:01<00:00, 17.41s/it, S=20/100, C=80/100]

Results for dense_square:
  SUCCESS:   20/100 (20.0%)
  COLLISION: 80/100 (80.0%)
  TIMEOUT:   0/100 (0.0%)
  TIME TAKEN (s): 11.64
  DISC. FREQ: 13.02
  DISC. DIST (m): 0.32

======================================================================
Test Case [4]: large_circle | 12 humans
======================================================================
[INFO] Loading CADRL policy
[DEBUG] Policy Class: <class 'crowd_nav.policy.cadrl.CADRL'>
100%|█████████████████████████████████████████| 100/100 [30:27<00:00, 18.28s/it, S=22/100, C=73/100]

Results for large_circle:
  SUCCESS:   22/100 (22.0%)
  COLLISION: 73/100 (73.0%)
  TIMEOUT:   5/100 (5.0%)
  TIME TAKEN (s): 17.05
  DISC. FREQ: 18.09
  DISC. DIST (m): 0.63

======================================================================
Test Case [5]: large_square | 20 humans
======================================================================
[INFO] Loading CADRL policy
[DEBUG] Policy Class: <class 'crowd_nav.policy.cadrl.CADRL'>
100%|█████████████████████████████████████████| 100/100 [36:36<00:00, 21.97s/it, S=61/100, C=36/100]

Results for large_square:
  SUCCESS:   61/100 (61.0%)
  COLLISION: 36/100 (36.0%)
  TIMEOUT:   3/100 (3.0%)
  TIME TAKEN (s): 10.89
  DISC. FREQ: 13.04
  DISC. DIST (m): 0.56
(mamba) root@5fe3ffc024a9:/workspace/nav_data/mamba/camrl/CrowdNav/crowd_nav# 


(mamba) root@5fe3ffc024a9:/workspace/nav_data/mamba/camrl/CrowdNav/crowd_nav# python test5.py --gpu --episodes 100
[INFO] ========================================
[INFO] TEST5.PY - ORCA Baseline Testing
[INFO] ========================================
[INFO] Config: T=12, dt=0.25, v_max=1.0 (MATCHED TO TRAINING)
[INFO] Feature: Classical collision avoidance (no learning)
Using device: cuda

======================================================================
Test Case [0]: baseline_circle | 5 humans
======================================================================
[INFO] Loading ORCA policy
[DEBUG] Policy Class: <class 'crowd_nav.policy.orca_wrapper.ORCA_WRAPPER'>
100%|█████████████████████████████████████████| 100/100 [00:01<00:00, 71.21it/s, S=56/100, C=44/100]

Results for baseline_circle:
  SUCCESS:   56/100 (56.0%)
  COLLISION: 44/100 (44.0%)
  TIMEOUT:   0/100 (0.0%)
  TIME TAKEN (s): 11.06
  DISC. FREQ: 9.73
  DISC. DIST (m): 0.86

======================================================================
Test Case [1]: baseline_square | 10 humans
======================================================================
[INFO] Loading ORCA policy
[DEBUG] Policy Class: <class 'crowd_nav.policy.orca_wrapper.ORCA_WRAPPER'>
100%|█████████████████████████████████████████| 100/100 [00:03<00:00, 30.51it/s, S=56/100, C=44/100]

Results for baseline_square:
  SUCCESS:   56/100 (56.0%)
  COLLISION: 44/100 (44.0%)
  TIMEOUT:   0/100 (0.0%)
  TIME TAKEN (s): 10.05
  DISC. FREQ: 6.90
  DISC. DIST (m): 0.74

======================================================================
Test Case [2]: dense_circle | 10 humans
======================================================================
[INFO] Loading ORCA policy
[DEBUG] Policy Class: <class 'crowd_nav.policy.orca_wrapper.ORCA_WRAPPER'>
100%|█████████████████████████████████████████| 100/100 [00:03<00:00, 26.96it/s, S=25/100, C=75/100]

Results for dense_circle:
  SUCCESS:   25/100 (25.0%)
  COLLISION: 75/100 (75.0%)
  TIMEOUT:   0/100 (0.0%)
  TIME TAKEN (s): 12.94
  DISC. FREQ: 10.78
  DISC. DIST (m): 0.53

======================================================================
Test Case [3]: dense_square | 20 humans
======================================================================
[INFO] Loading ORCA policy
[DEBUG] Policy Class: <class 'crowd_nav.policy.orca_wrapper.ORCA_WRAPPER'>
100%|█████████████████████████████████████████| 100/100 [00:09<00:00, 10.90it/s, S=17/100, C=83/100]

Results for dense_square:
  SUCCESS:   17/100 (17.0%)
  COLLISION: 83/100 (83.0%)
  TIMEOUT:   0/100 (0.0%)
  TIME TAKEN (s): 13.79
  DISC. FREQ: 8.48
  DISC. DIST (m): 0.43

======================================================================
Test Case [4]: large_circle | 12 humans
======================================================================
[INFO] Loading ORCA policy
[DEBUG] Policy Class: <class 'crowd_nav.policy.orca_wrapper.ORCA_WRAPPER'>
100%|█████████████████████████████████████████| 100/100 [00:07<00:00, 13.69it/s, S=32/100, C=68/100]

Results for large_circle:
  SUCCESS:   32/100 (32.0%)
  COLLISION: 68/100 (68.0%)
  TIMEOUT:   0/100 (0.0%)
  TIME TAKEN (s): 16.59
  DISC. FREQ: 11.10
  DISC. DIST (m): 0.77

======================================================================
Test Case [5]: large_square | 20 humans
======================================================================
[INFO] Loading ORCA policy
[DEBUG] Policy Class: <class 'crowd_nav.policy.orca_wrapper.ORCA_WRAPPER'>
100%|█████████████████████████████████████████| 100/100 [00:11<00:00,  8.98it/s, S=38/100, C=61/100]

Results for large_square:
  SUCCESS:   38/100 (38.0%)
  COLLISION: 61/100 (61.0%)
  TIMEOUT:   1/100 (1.0%)
  TIME TAKEN (s): 10.57
  DISC. FREQ: 8.50
  DISC. DIST (m): 0.57
(mamba) root@5fe3ffc024a9:/workspace/nav_data/mamba/camrl/CrowdNav/crowd_nav# 



(mamba) root@5fe3ffc024a9:/workspace/nav_data/mamba/camrl/CrowdNav/crowd_nav# python test6.py --gpu --episodes 100
[INFO] ========================================
[INFO] TEST6.PY - PPO Baseline Testing
[INFO] ========================================
[INFO] Config: T=12, dt=0.25, v_max=1.0 (MATCHED TO TRAINING)
[INFO] Feature: TTC-Sorting enabled (与contracts.py训练时一致)
Using device: cuda

======================================================================
Test Case [0]: baseline_circle | 5 humans
======================================================================
[INFO] Detected d_state=64 from checkpoint
[INFO] Loading PPO policy
[DEBUG] Policy Class: <class 'crowd_nav.policy.mamba_rl_ppo.MambaRL'>
100%|██████████████████████████████████████████| 100/100 [00:20<00:00,  4.77it/s, S=68/100, C=3/100]

Results for baseline_circle:
  SUCCESS:   68/100 (68.0%)
  COLLISION: 3/100 (3.0%)
  TIMEOUT:   29/100 (29.0%)
  TIME TAKEN (s): 13.31
  DISC. FREQ: 0.15
  DISC. DIST (m): 1.68

======================================================================
Test Case [1]: baseline_square | 10 humans
======================================================================
[INFO] Detected d_state=64 from checkpoint
[INFO] Loading PPO policy
[DEBUG] Policy Class: <class 'crowd_nav.policy.mamba_rl_ppo.MambaRL'>
100%|█████████████████████████████████████████| 100/100 [00:35<00:00,  2.84it/s, S=30/100, C=19/100]

Results for baseline_square:
  SUCCESS:   30/100 (30.0%)
  COLLISION: 19/100 (19.0%)
  TIMEOUT:   51/100 (51.0%)
  TIME TAKEN (s): 15.18
  DISC. FREQ: 1.01
  DISC. DIST (m): 1.33

======================================================================
Test Case [2]: dense_circle | 10 humans
======================================================================
[INFO] Detected d_state=64 from checkpoint
[INFO] Loading PPO policy
[DEBUG] Policy Class: <class 'crowd_nav.policy.mamba_rl_ppo.MambaRL'>
100%|█████████████████████████████████████████| 100/100 [00:37<00:00,  2.68it/s, S=41/100, C=17/100]

Results for dense_circle:
  SUCCESS:   41/100 (41.0%)
  COLLISION: 17/100 (17.0%)
  TIMEOUT:   42/100 (42.0%)
  TIME TAKEN (s): 16.32
  DISC. FREQ: 0.63
  DISC. DIST (m): 1.32

======================================================================
Test Case [3]: dense_square | 20 humans
======================================================================
[INFO] Detected d_state=64 from checkpoint
[INFO] Loading PPO policy
[DEBUG] Policy Class: <class 'crowd_nav.policy.mamba_rl_ppo.MambaRL'>
100%|██████████████████████████████████████████| 100/100 [00:56<00:00,  1.77it/s, S=4/100, C=32/100]

Results for dense_square:
  SUCCESS:   4/100 (4.0%)
  COLLISION: 32/100 (32.0%)
  TIMEOUT:   64/100 (64.0%)
  TIME TAKEN (s): 17.56
  DISC. FREQ: 1.37
  DISC. DIST (m): 1.06

======================================================================
Test Case [4]: large_circle | 12 humans
======================================================================
[INFO] Detected d_state=64 from checkpoint
[INFO] Loading PPO policy
[DEBUG] Policy Class: <class 'crowd_nav.policy.mamba_rl_ppo.MambaRL'>
100%|██████████████████████████████████████████| 100/100 [00:46<00:00,  2.17it/s, S=50/100, C=3/100]

Results for large_circle:
  SUCCESS:   50/100 (50.0%)
  COLLISION: 3/100 (3.0%)
  TIMEOUT:   47/100 (47.0%)
  TIME TAKEN (s): 19.12
  DISC. FREQ: 0.21
  DISC. DIST (m): 1.45

======================================================================
Test Case [5]: large_square | 20 humans
======================================================================
[INFO] Detected d_state=64 from checkpoint
[INFO] Loading PPO policy
[DEBUG] Policy Class: <class 'crowd_nav.policy.mamba_rl_ppo.MambaRL'>
100%|█████████████████████████████████████████| 100/100 [00:42<00:00,  2.36it/s, S=21/100, C=47/100]

Results for large_square:
  SUCCESS:   21/100 (21.0%)
  COLLISION: 47/100 (47.0%)
  TIMEOUT:   32/100 (32.0%)
  TIME TAKEN (s): 16.38
  DISC. FREQ: 1.56
  DISC. DIST (m): 1.06
(mamba) root@5fe3ffc024a9:/workspace/nav_data/mamba/camrl/CrowdNav/crowd_nav# 


(mamba) root@5fe3ffc024a9:/workspace/nav_data/mamba/camrl/CrowdNav/crowd_nav# python test7.py --episodes 100
[INFO] ========================================
[INFO] TEST7.PY - DSRNN Baseline Testing
[INFO] ========================================
[INFO] Config: T=12, dt=0.25, v_max=1.0 (MATCHED TO TRAINING)
[INFO] Feature: TTC-Sorting enabled (与contracts.py训练时一致)
Using device: cpu

======================================================================
Test Case [0]: baseline_circle | 5 humans
======================================================================
[INFO] Loading DSRNN policy
[DEBUG] Loading weights from runs/mamba_vl/27776.pt
[DEBUG] State dict has 45 keys
[DEBUG] Loaded weights via policy.load_state_dict()
[DEBUG] Policy Class: <class 'policy.dsrnn_policy.DSRNNPolicy'>
100%|█████████████████████████████████████████| 100/100 [00:08<00:00, 11.49it/s, S=76/100, C=13/100]

Results for baseline_circle:
  SUCCESS:   76/100 (76.0%)
  COLLISION: 13/100 (13.0%)
  TIMEOUT:   11/100 (11.0%)
  TIME TAKEN (s): 12.97
  DISC. FREQ: 1.24
  DISC. DIST (m): 1.43

======================================================================
Test Case [1]: baseline_square | 10 humans
======================================================================
[INFO] Loading DSRNN policy
[DEBUG] Loading weights from runs/mamba_vl/27776.pt
[DEBUG] State dict has 45 keys
[DEBUG] Loaded weights via policy.load_state_dict()
[DEBUG] Policy Class: <class 'policy.dsrnn_policy.DSRNNPolicy'>
100%|█████████████████████████████████████████| 100/100 [00:08<00:00, 12.03it/s, S=37/100, C=57/100]

Results for baseline_square:
  SUCCESS:   37/100 (37.0%)
  COLLISION: 57/100 (57.0%)
  TIMEOUT:   6/100 (6.0%)
  TIME TAKEN (s): 14.85
  DISC. FREQ: 2.27
  DISC. DIST (m): 0.89

======================================================================
Test Case [2]: dense_circle | 10 humans
======================================================================
[INFO] Loading DSRNN policy
[DEBUG] Loading weights from runs/mamba_vl/27776.pt
[DEBUG] State dict has 45 keys
[DEBUG] Loaded weights via policy.load_state_dict()
[DEBUG] Policy Class: <class 'policy.dsrnn_policy.DSRNNPolicy'>
100%|█████████████████████████████████████████| 100/100 [00:07<00:00, 13.30it/s, S=25/100, C=70/100]

Results for dense_circle:
  SUCCESS:   25/100 (25.0%)
  COLLISION: 70/100 (70.0%)
  TIMEOUT:   5/100 (5.0%)
  TIME TAKEN (s): 15.10
  DISC. FREQ: 2.92
  DISC. DIST (m): 0.87

======================================================================
Test Case [3]: dense_square | 20 humans
======================================================================
[INFO] Loading DSRNN policy
[DEBUG] Loading weights from runs/mamba_vl/27776.pt
[DEBUG] State dict has 45 keys
[DEBUG] Loaded weights via policy.load_state_dict()
[DEBUG] Policy Class: <class 'policy.dsrnn_policy.DSRNNPolicy'>
100%|██████████████████████████████████████████| 100/100 [00:11<00:00,  8.43it/s, S=8/100, C=89/100]

Results for dense_square:
  SUCCESS:   8/100 (8.0%)
  COLLISION: 89/100 (89.0%)
  TIMEOUT:   3/100 (3.0%)
  TIME TAKEN (s): 17.69
  DISC. FREQ: 2.72
  DISC. DIST (m): 0.63

======================================================================
Test Case [4]: large_circle | 12 humans
======================================================================
[INFO] Loading DSRNN policy
[DEBUG] Loading weights from runs/mamba_vl/27776.pt
[DEBUG] State dict has 45 keys
[DEBUG] Loaded weights via policy.load_state_dict()
[DEBUG] Policy Class: <class 'policy.dsrnn_policy.DSRNNPolicy'>
100%|█████████████████████████████████████████| 100/100 [00:17<00:00,  5.81it/s, S=37/100, C=47/100]

Results for large_circle:
  SUCCESS:   37/100 (37.0%)
  COLLISION: 47/100 (47.0%)
  TIMEOUT:   16/100 (16.0%)
  TIME TAKEN (s): 19.24
  DISC. FREQ: 2.71
  DISC. DIST (m): 1.18

======================================================================
Test Case [5]: large_square | 20 humans
======================================================================
[INFO] Loading DSRNN policy
[DEBUG] Loading weights from runs/mamba_vl/27776.pt
[DEBUG] State dict has 45 keys
[DEBUG] Loaded weights via policy.load_state_dict()
[DEBUG] Policy Class: <class 'policy.dsrnn_policy.DSRNNPolicy'>
100%|█████████████████████████████████████████| 100/100 [00:16<00:00,  6.20it/s, S=13/100, C=81/100]

Results for large_square:
  SUCCESS:   13/100 (13.0%)
  COLLISION: 81/100 (81.0%)
  TIMEOUT:   6/100 (6.0%)
  TIME TAKEN (s): 12.44
  DISC. FREQ: 2.69
  DISC. DIST (m): 0.80
(mamba) root@5fe3ffc024a9:/workspace/nav_data/mamba/camrl/CrowdNav/crowd_nav# 


