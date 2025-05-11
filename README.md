## Unified conformaitonal search algorithm

Conformational search method for cyclic molecules based on inverse kinematics Monte-Carlo and Bayessian optimization:

## References

1. Inverse kinematics: [paper](https://pubs.acs.org/doi/abs/10.1021/acs.jcim.3c02040), [docs & code](https://knvvv.gitlab.io/ringo/index.html)
2. Bayessian optimization: paper is under review, [code](https://github.com/IvanBespalov64/bo_confsearch/)

## Usage

1. Clone repo
```
git clone https://github.com/IvanBespalov64/bo_confsearch.git
```

2. Install dependecies
```
conda create -c envname python=3.10 rdkit pyyaml netoworkx icecream
pip install trieste vf3py ringo-ik
conda activate envname
```

3. Customize the config file `config.yaml`

4. Run the sampling script:

```
python runner.py
```

Also, [XTB](https://github.com/grimme-lab/xtb) have to be installed and available in $PATH.
