# Implementation, Cluster Workflow and Debugging

**Contents**

1. [Implementation framework](#1-implementation-framework)
2. [Cluster workflow and reproducibility](#2-cluster-workflow-and-reproducibility)
3. [Code patches and debugging](#3-code-patches-and-debugging)

---

## 1. Implementation framework

This part connects the mathematical method with the software pipeline,
emphasizing the repository organization, the configuration mechanism and the
execution entry point.

### 1.1 Repository organization

The relevant code organization is:

```
g-adaptivity/
  src/
    run_pipeline.py
    run_GNN.py
    utils_train.py
    utils_eval.py
    utils_eval_rollout.py
    pde_solvers.py
    params.py
    models/
      mesh_adaptor_model.py
      direct_optimisation.py
  classical_meshing/
    ma_mesh_2d.py
  configs/
    base_config.yaml
    repro_poisson_square_gadapt.yaml
    repro_poisson_poly_gadapt.yaml
    repro_nonconvex_l_gadapt.yaml
    repro_nonconvex_h_gadapt.yaml
    repro_burgers_10s_gadapt_mix.yaml
    repro_burgers_roll_gadapt_mix.yaml
    repro_ns_cyl_gadapt.yaml
    ... (um2ng/um2n variants of the above)
```

The entry point is `run_pipeline.py`. It reads the configuration, builds or
loads data, trains or loads the model, evaluates it, logs results and optionally
generates diagnostic plots. The `repro_*` configuration files encode the
paper-exact protocol of the final experiments; each has variants for the learned
models compared in the paper (G-Adapt, UM2N-G, UM2N).

### 1.2 Configuration system

Experiments are controlled by YAML files. The code first reads default
parameters from `configs/base_config.yaml` and then overwrites them with the
experiment-specific file passed on the command line:

```bash
python3 src/run_pipeline.py --exp_config configs/<experiment>.yaml
```

This design is useful because only the parameters that differ from the base
configuration need to be included in an experiment file.

The configuration is organized into five main sections:

| Section  | Meaning |
|----------|---------|
| `run`    | High-level experiment settings, such as PDE type, model, device and logging flags. |
| `data`   | Dataset generation, mesh geometry, mesh dimensions, monitor function and train/test sizes. |
| `pde`    | PDE-specific parameters, especially for Burgers and Navier–Stokes experiments. |
| `exp`    | Training parameters, such as epochs, batch size, learning rate and loss type. |
| `model`  | Neural-network and mesh-deformation architecture parameters. |

### 1.3 Important configuration keys

The most important keys for the replication are:

| Key | Role in the experiment |
|-----|------------------------|
| `pde_type` | Selects the PDE: `Poisson`, `Burgers` or `NavierStokes`. |
| `data_type` | Selects the dataset family: `randg_mix` (random Gaussians, mixed mesh resolutions), `randg` (random Gaussians, fixed mesh) or `RBF` (Navier–Stokes). |
| `mesh_geometry` | Selects the computational domain: `rectangle`, `polygon_010`, `L-shape`, `H-shape` or `cylinder_100_025`. |
| `mesh_dims_train` / `mesh_dims_test` | Mesh resolutions used for training and evaluation. |
| `monitor_type` | Selects the monitor used in classical mesh generation; `monitor_hessian` throughout. |
| `loss_type` | Selects the training loss: `pde_loss_regularised` for G-Adapt and UM2N-G, `UM2N_loss` for the original UM2N ablation. |
| `loss_regulariser_weight` | Weight λ of the mesh regularization term (1.0 in the final campaign). |
| `gnn_num_layers` | Number of graph diffusion/deformation blocks (4). |
| `grand_diffusion` | Activates the diffusion-based Diffformer deformer. |
| `grand_diffusion_steps` / `grand_step_size` | Explicit Euler rollout of the diffusion dynamics (32 steps, step size 0.1). |

### 1.4 Caveat: paper parameters vs. YAML keys

The mapping between the parameters described in the paper and the YAML keys
exposed by the repository is **not always one-to-one**, and some keys present in
`base_config.yaml` are not the ones actually consumed by the code.

In particular, the deformer implementation in
`src/models/mesh_adaptor_model.py` reads `grand_diffusion` (with default value
`true` set in `src/params.py`), together with `grand_diffusion_steps` and
`grand_step_size`. The similarly named key `GNN_diffusion` that appears in the
base configuration is **legacy and is not read by the model code**.

Therefore, to ensure correct execution during production runs, the `repro_*`
configurations of the final experiments explicitly set `grand_diffusion` and its
rollout keys. A representative production configuration (the paper-exact Poisson
square G-Adapt file) is:

```yaml
run:
  pde_type: "Poisson"
  data_type: "randg_mix"
  model: "MeshAdaptor"

data:
  mesh_geometry: "rectangle"
  mesh_dims: [15, 15]
  num_train: 100
  num_test: 100
  num_gauss_range: [2, 3]
  anis_gauss: false
  mesh_dims_train: [[15, 15], [20, 20]]
  mesh_dims_test: [[12, 12], [13, 13], [14, 14], [15, 15], [16, 16],
                   [17, 17], [18, 18], [19, 19], [20, 20], [21, 21],
                   [22, 22], [23, 23]]
  monitor_type: "monitor_hessian"
  monitor_alpha: 5
  mon_reg: 0.1
  mon_power: 0.2
  mesh_file_type: "bin"
  eval_quad_points: 101
  HO_degree: 4

exp:
  epochs: 300
  batch_size: 1
  lr: 0.001
  loss_type: "pde_loss_regularised"
  loss_regulariser_weight: 1.0
  loss_fn: "l1"

model:
  new_model_monitor_type: "UM2N"
  grand_diffusion: true
  grand_diffusion_steps: 32
  grand_step_size: 0.1
  gnn_num_layers: 4
```

The other `repro_*` files differ only in geometry, dataset family, PDE block
and (for UM2N-G/UM2N) deformer and loss keys.

---

## 2. Cluster workflow and reproducibility

This part records the computational environment used to make the experiments
reproducible on the PoliMi DMAT Student cluster, separating persistent storage, batch
execution and runtime configuration.

### 2.1 Execution model

All production runs were executed through **PBS batch scripts**. The login node is used only for access, Git operations, job submission,
monitoring and file transfer. Numerical computations and training runs must be
executed on compute nodes.

A typical workflow is:

```bash
ssh <user>@10.78.18.100
. /etc/profile.d/pbs.sh
cd /work/$USER/NAPDE_project/jobs
qsub <pbs_script>.pbs
qstat -u $(whoami)
```

The exact username and project paths differ between cluster accounts.

### 2.2 Project directory layout

The adopted persistent project layout is:

```
/work/$USER/NAPDE_project/
  repos/
    NAPDE-PACS-project/
      g-adaptivity/
  envs/
    firedrake/
  jobs/
  logs/
  data/
  results/
```

The `/work` filesystem stores project files, environments and final results.
For I/O-intensive jobs, the run scripts may additionally copy the repository and
data to the node-local `/scratch_local` directory, execute there, and copy the
results back to `/work`. This avoids unnecessary stress on the shared network
filesystem.

### 2.3 Environment activation and execution strategy

Prior to launching the pipeline, the Firedrake virtual environment is activated
and its environment variables are exported through `firedrake-configure`. A
small set of environment variables is used to ensure that the run is
deterministic and batch-friendly:

- thread counts for OpenMP, MKL, OpenBLAS and NumExpr are set to one to avoid
  oversubscription;
- Weights & Biases logging is redirected to local cache directories;
- the Matplotlib backend is forced to `Agg` for non-interactive plotting;
- Python output is unbuffered.


### 2.4 Hardware and software of the final campaign

All 73 production runs were executed on the CPU queue of the `mate.polimi.it`
cluster, on three homogeneous nodes (`cpu01`, `cpu04`, `cpu05`). 
Each job requested 4 cores and a 24 h
walltime, but the code runs essentially single-threaded because the BLAS/OpenMP
thread counts are pinned to one.

The software stack is identical for all runs:

- Python 3.12.3
- PyTorch 2.11.0 (CPU build)
- PyTorch-Geometric 2.4.0
- Firedrake 2026.4.0
- `movement` 0.3
- petsc4py 3.25.0

Two facts about this environment matter for interpreting the timing results.
First, the nodes are **shared** (`sharing = default_shared` in PBS): jobs of
other users can run on the same node, so absolute times depend on the node load
at execution time. Second, effective job walltimes were dominated by training
(which performs a Firedrake adjoint solve at every optimization step): typically
3–9 h for the Poisson experiments, up to 13–17 h for Burgers and the non-convex
geometries, and 15–23 h for the 1000-epoch UM2N runs. The relocation times
reported in the results refer only to inference at evaluation time.


---
## 3. Code patches and debugging

Three types of changes are distinguished here: implemented compatibility
patches, cluster workflow adaptations, and diagnostic modifications used during
debugging. This distinction matters because not every temporary debugging change
should be described as final production code.

### 3.1 Implemented compatibility patch: PyVista cell quality

The evaluation code computes mesh quality metrics through PyVista. Recent
PyVista versions changed the cell-quality API. The original code expected:

```python
pv_mesh.compute_cell_quality(quality_measure=metric)
```

To ensure compatibility with the cluster environment while maintaining support
for older versions, this was replaced with a compatibility block that
dynamically detects whether to use the legacy method or the newer
`cell_quality()` interface. This modification does not alter the mathematical
method; it only allows the evaluation code to compute quantities such as minimum
angle, aspect ratio and scaled Jacobian under the installed PyVista version.

### 3.2 Implemented reproducibility patch: persistent output saving

A second implemented patch concerns `src/run_pipeline.py`. The original workflow
was inadequate for non-interactive cluster jobs, because results, summary tables
and plots needed to be saved in a predictable directory. The patched pipeline
writes output to the path specified by the environment variable `RESULTS_DIR`.
The saved files include:

- `results_raw.csv`: sample-wise FEM errors and error reductions;
- `times_raw.csv`: mesh generation or model inference times;
- `mesh_metrics_raw.csv`: mesh-quality indicators;
- `rollout_raw.csv` (Burgers rollout only): per-block rollout errors and times;
- `config_used.yaml`: the exact configuration consumed by the run;
- summary CSV files obtained from descriptive statistics, and PNG plots.

This patch is essential for reproducibility because a PBS job may be unattended
rather than interactive, and thus it may terminate without a session in which
intermediate results can be inspected manually. All the result tables in the
report are built automatically from these per-run CSV files by an aggregation
script, so that every number can be traced back to a job ID.

### 3.3 Diagnostic debugging: hidden PDE-loss failures and loss magnitude
The most instructive bug was invisible from the training curve: early runs sometimes worsened the FEM error on the test set, even though the logged loss (~10⁻⁶) decreased smoothly. Plotting the validation error alongside it revealed the contradiction, as it stayed noisy with no downward trend.

The cause was in how `run_GNN.py` combines the PDE loss and the regularizer:

```python
try:
    pseudo_loss = coarse_solver.loss(...) \
        + opt['loss_regulariser_weight'] * equidistribution_loss(x_phys, data)
except:
    Warning("PDE loss failed, using only regulariser for this datapoint")
    pseudo_loss = opt['loss_regulariser_weight'] * equidistribution_loss(x_phys, data)
```

The bare `except` swallowed every exception, and `Warning(...)` builds a warning object without ever raising or logging it, so a failure left no trace. Whenever the `try` block failed, the objective quietly dropped from `L_PDE + λ·L_reg` to `λ·L_reg`: the network was only smoothing the mesh, learning nothing about the physics. Because that term is small and well-behaved, the curve looked fine.

Replacing the bare handler with a proper `except Exception as e:` that prints the error, and checking `requires_grad` on both loss terms before the backward pass, exposed the real problem: a Firedrake/PyTorch incompatibility (`'Cofunction' object has no attribute 'vector'`) that fired on nearly every datapoint. Fixing the Firedrake object so `coarse_solver.loss(...)` returns a differentiable tensor let the full loss train as intended.

A related subtlety: the equidistribution loss used `torch.nn.MSELoss()` (a mean) instead of the sum used in the paper, which pushes the logged value down to ~10⁻⁶. The magnitude of the logged loss is therefore uninformative unless you know whether it is a sum or a mean and whether the two loss terms are logged separately.

Both episodes make the same point: a smooth, tiny loss curve is not on its own evidence that the right objective is being optimized. The practical safeguards are to log the PDE and regularization losses separately, confirm the PDE loss actually contributes gradients to the network, avoid catch-all exception handlers that silently drop it, and check the sign convention of the error-reduction metric.


### 3.4 Summary of patch status

| Item | Purpose | Status |
|------|---------|--------|
| PyVista API patch | Make mesh-quality evaluation compatible with the installed PyVista version. | Implemented |
| Output-saving patch | Save raw CSV files, summary tables and PNG plots under `RESULTS_DIR`. | Implemented |
| Gradient-clipping flag | Optional `--grad_clip_norm` for the stability study. | Implemented (opt-in) |
| Rollout evaluation | Reconstruction of the Burgers rollout protocol, absent from the released code. | Implemented |
| Scratch workflow | Avoid heavy I/O on shared filesystems and preserve final outputs. | Workflow adaptation |
| `--nv` Apptainer flag | Make GPUs visible inside the container when GPU nodes are used. | Conditional workflow adaptation |
| Exception exposure | Reveal hidden Firedrake/PDE-loss failures during training. | Diagnostic |
| `requires_grad` prints | Check whether losses remain connected to the computational graph. | Diagnostic |

