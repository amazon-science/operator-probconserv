## Operator-ProbConserv: OOD Uncertainty Quantification (UQ) for Neural Operators

This repository contains the code for the paper "Using Uncertainty Quantification to Characterize and Improve Out-of-Domain Learning for PDEs" by S Chandra Mouli, Danielle C. Maddix, Shima Alizadeh, Gaurav Gupta, Yuyang Wang, Andrew Stuart, Michael W. Mahoney

## Setup
Install dependencies by running
```
conda create -n env python=3.9
conda activate env
pip install -r requirements.txt
```

Run `python -u experiment_ood_params.py --help` for possible options.

Example to train DiverseNO model on 1-d heat equation task:
```
python -u experiment_ood_params.py --model=DiverseFNO2d --dataset=HeatEquation_1D --seed=0 --dataset_params=1,5,0,0 --train_ood_dataset_params=1,5,0,0 --n_samples=200 --tplot=0.5 --m.n_models=10 --m.reg_type=weights_l2 --m.reg_strength=1 --epochs=1000
```

To evaluate the trained model on different OOD parameters, use `--ood_dataset_params` and `--no_train` options.
```
python -u experiment_ood_params.py --model=DiverseFNO2d --dataset=HeatEquation_1D --seed=0 --dataset_params=1,5,0,0 --train_ood_dataset_params=1,5,0,0 --n_samples=200 --tplot=0.5 --m.n_models=10 --m.reg_type=weights_l2 --m.reg_strength=1 --epochs=1000 --ood_dataset_params=5,6,0,0 --no_train
```

Models: EnsembleFNO2d, BayesianFNO2d, MCDropoutFNO2d, OutputVarFNO2d, DiverseFNO2d
Datasets: HeatEquation_1D, PME_1D, StefanPME_1D, LinearAdvection_1D.

## Sources
This repo contains modified versions of the code found in the following repos:

https://github.com/zongyi-li/fourier_neural_operator: For implementation of the Fourier Neural Operator (FNO) (MIT license)
https://github.com/amazon-science/probconserv: For implementation of ProbConserv (Apache 2.0 license)

## Citation
If you use this code, or our work, please cite: 

@article{mouli2024_ood_uq_no,
    title={Using Uncertainty Quantification to Characterize and Improve Out-of-Domain Learning for PDEs},
    author={Mouli, S.C., Maddix, D.C., Alizadeh, S., Gupta, G., Wang, Y., Stuart, A., Mahoney, M.W.},
    year={2024}
}

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.

