# Integrating the policy components

The methods use PyTorch. The experiments used the panoramic `Lookahead_3DFF` setup described in the [g3D-LF repository](https://github.com/MrZihan/g3D-LF). The published baseline was evaluated from its `ckpt.iter25000.pth` checkpoint on Habitat. Scene data and upstream dependencies can be obtained by following the upstream setup and asset instructions.

## GRPO

For each instruction and start, the training host runs a group of stochastic rollouts. `trajectory_reward` uses the rollout's final geodesic distance, initial geodesic distance, and accumulated path length. The training reward uses a 1.5 m success cutoff. R2R-CE evaluation uses the benchmark's 3 m threshold. `group_relative_advantages` takes a `[groups, rollouts]` tensor of completed rewards. Broadcast each advantage to that rollout's valid decision steps before passing the sampled action log probabilities to `grpo_loss`. The old policy supplies the sampling probabilities and a frozen copy supplies the KL reference. The historical GRPO run updated the global navigation encoder and action readout.

## Reliability fusion

For each decision, extract the original global scores, the indices of the two candidates selected for lookahead, the lookahead scores for those candidates, and the imagined panoramic features. `reliability_features` packs these into the gate's 12 inputs. Candidate scores use STOP at index 0. Apply the resulting sigmoid weight with `fuse_logits`. Unselected actions retain their global scores and invalid actions remain masked.

The ranking label is based on agreement with a teacher using real observations at those candidate positions. The teacher is used during training. The runtime gate uses only the current policy's scores and imagined features. Offline tuning uses the signed-return objective in `offline_loss_terms` with a fixed initial gate as a reference.

## Evaluation provenance

The benchmark comparison uses the full R2R-CE `val_unseen` split (613 routes and 1,839 instructions). The original g3D-LF and the final GRPO-plus-fusion policy were evaluated in separate runs. The final policy uses the GRPO iter500 navigation checkpoint and the all-states offline gate selected at epoch 8. Its full report combines 300 development episodes with 1,539 confirmation episodes. All three examples shown in the README come from that confirmation subset. The demo videos are separate recording runs. Both policies start from the same episode and seed, and the video shows the RGB camera recorded during each rollout. The final distances shown with the videos come from those recording runs.
