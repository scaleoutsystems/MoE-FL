# MoE-FL

## Todo Experiments
1. Run convnext moe experiment for imagenet-100 (inc expert weighted)
2. Run context-segmented experiments moe imagenet-100 (inc expert weighted)
3. Run BPR ablation
4. Run label heterogeniety experiment on imagenet-100
5. Run Imagenet-1k experiments

## Ideas for aggregation in FL
1. Penalize contribution from high entropy clients? Not good specialization (this may not work at all)

## Ideas for communication efficient MoEs
0. Since we are aggregating are all experts even needed (think of it like client participation)
1. Do not distribute all experts to last layer clients
2. Distribute router and check utilization before sending experts (seems stupid in a real FL setting)
3. Use some metric (kl div, gini impurity, coverage at threshold, shannon entropy) to estimate needed experts
4. Cluster data distributions and add experts per cluster
5. Multi armed bandit
6. Fisher information diagonal

## TODO code specified
1. Utils
    1. FL utils
        - add SCAFFOLD
2. Modules
   1. convnext_moe
        - add option to choose routing
        - make it possible to turn off BPR
   2. vit_moe
        - add option to choose routing
        - make it possible to turn off BPR