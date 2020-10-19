# Quantized Spiking Neural Networks

This repository contains the models and training scripts used in the papers: ["Quantizing Spiking Neural Networks with Integers"](https://dl.acm.org/doi/abs/10.1145/3407197.3407203) (ICONS 2020) and ["Memory Organization for Energy-Efficient Learning and Inference in Digital Neuromorphic Accelerators"](https://ieeexplore.ieee.org/document/9180443) (ISCAS 2020).

## Requiremnts

- Python
- PyTorch
- torchvision
- NumPy
- pickle
- argparse


## Quantized SNNs for Spatio-Temporal Patterns

All relevant code for the experiments from the ISCAS paper is contained in spytorch_precise.py, quantization.py and spytorch_util.py. To run the experiments execute:

```
python spytorch_precise.py
```

You can specify desired setting either as command-line arguments or within spytorch_precise.py.

Optional arguments:

| Argument               |  Description                                             | 
|:-----------------------|:---------------------------------------------------------|
| --input INPUT          | Input pickle file (default: ./data/input_700_250_25.pkl) |
| --target TARGET        | Target pattern pickle (default: ./data/smile95.pkl)      |
| --global_wb GLOBAL_WB  | Weight bitwidth (default: 2)                             |
| --global_ab GLOBAL_AB  | Membrane potential, synapse state bitwidth (default: 8)  |
| --global_gb GLOBAL_GB  | Gradient bitwidth (default: 8)                           |
| --global_eb GLOBAL_EB  | Error bitwidth (default: 8)                              |
| --global_rb GLOBAL_RB  | Gradient RNG bitwidth (default: 16)                      |
| --time_step TIME_STEP  | Simulation time step size (default: 0.001)               |
| --nb_steps NB_STEPS    | Simulation steps (default: 250)                          |
| --nb_epochs NB_EPOCHS  | Simulation steps (default: 10000)                        |
| --tau_mem TAU_MEM      | Time constant for membrane potential (default: 0.01)     |
| --tau_syn TAU_SYN      | Time constant for synapse (default: 0.005)               |
| --tau_vr TAU_VR        | Time constant for Van Rossum distance (default: 0.005)   |
| --alpha ALPHA          | Time constant for synapse (default: 0.75)                |
| --beta BETA            | Time constant for Van Rossum distance (default: 0.875)   |
| --nb_inputs NB_INPUTS  | Spatial input dimensions (default: 700)                  |
| --nb_hidden NB_HIDDEN  | Spatial hidden dimensions (default: 400)                 |
| --nb_outputs NB_OUTPUTS| Spatial output dimensions (default: 250)                 |


## Quantized SNNs for Gesture Detection with Local Learning



