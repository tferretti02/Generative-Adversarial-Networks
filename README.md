# Generative-Adversarial-Networks

Ceci est le code d'un GAN basique à entrainer sur la base de données des digits de MNIST.

Sur une carte graphique CUDA relativement basique, il faut compter environ 10 minutes pour 10 epochs (1875 steps/epoch).

Pour installer les packages nécessaires:

```bash
$ pip install -r requirements.txt
```

Pour lancer le script:

```bash
$ python GAN.py
```

Pour lancer le tensorboard:

```bash
$ tensorboard --logdir=runs
```
