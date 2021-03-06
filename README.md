# An Information-Theoretic Justification for Model Pruning
PyTorch Implementation of the SuRP algorithm by the authors of the AISTATS 2022 paper "An Information-Theoretic Justification for Model Pruning". 

> [An Information-Theoretic Justification for Model Pruning](https://arxiv.org/pdf/2102.08329.pdf) <br/>
>[Berivan Isik](https://sites.google.com/view/berivanisik), [Tsachy Weissman](https://web.stanford.edu/~tsachy/), [Albert No](http://albertno.hongik.ac.kr/) <br/>
> International Conference on Artificial Intelligence and Statistics (AISTATS), 2022. <br/>

## 1) Train the baseline model:
To train the baseline model to be compressed, set `trainer=Classifier`. To try this for ResNet-20, run:

```
python3 main.py --trainer=Classifier --config=cifar_resnet20/config.yaml
```

To test the baseline model, run:

```
python3 main.py --trainer=Classifier --config=cifar_resnet20/config.yaml --test
```

## 2) One-shot (non-iterative) reconstruction with SuRP:
To compress the baseline model with SuRP non-iteratively, change the experiment id `exp_id` of the target model and target sparsity ratio `sparsity: [sparsity of the input model, target sparsity]` in the `recon.yaml` file accordingly. Then, run:

```
python3 main.py --trainer=Reconstruction --config=cifar_resnet20/recon.yaml
```

## 3) Iterative reconstruction with SuRP:
To compress the baseline model with SuRP iteratively, apply SuRP several times following a sparsity schedule. Each time, modify `exp_id` and `sparsity: [sparsity of the input model, target sparsity]`, accordingly. To retrain the sparse models before applying SuRP again, set `retrain: True`. And run:

```
python3 main.py --trainer=ReconFromFile --config=cifar_resnet20/recon.yaml
```

## References
If you find this work useful in your research, please consider citing our paper:
```
@InProceedings{pmlr-v151-isik22a,
  title = 	 { An Information-Theoretic Justification for Model Pruning },
  author =       {Isik, Berivan and Weissman, Tsachy and No, Albert},
  booktitle = 	 {Proceedings of The 25th International Conference on Artificial Intelligence and Statistics},
  pages = 	 {3821--3846},
  year = 	 {2022},
  volume = 	 {151},
  series = 	 {Proceedings of Machine Learning Research},
  publisher =    {PMLR},
  url = 	 {https://proceedings.mlr.press/v151/isik22a.html}
}
```
