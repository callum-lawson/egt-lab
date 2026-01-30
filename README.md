# 🧬 egt-lab


**Experimental exploration of evolutionary game theory and unsupervised environment design** 🤖🌱🧪

---

This is a collection of questions and experiments I'm exploring to help me understand how ideas from ecology and evolution might connect with robustness in reinforcement learning, and especially to unsupervised environment design (UED; see in particular [Emergent Complexity and Zero-shot Transfer via Unsupervised Environment Design](https://arxiv.org/abs/2012.02096)). If you come across anything you'd be interested in chatting about, I'd love to chat! You can message me at `callumrlawson [at] gmail [dot] com`.

## Repository structure 📂

- `lab-notes/`  
  📝 One-page write-ups of questions I'm curious about and referencing research papers / mini experiments that shed some light on them and their implications for evolution-based UED techniques. 
  
  - [**The non-saturating loss function in GANs**](lab-notes/2025-12-17_gan-nonsaturating-loss/notes.md) – Why the non-saturating GAN loss improves convergence and what this might teach us about UED training dynamics.
  
  - [**Constraints in GANs**](lab-notes/2025-12-21_gan-collapse/notes.md) – How constraints on representation and convergence affect GAN training, with experiments on latent dimensionality and learning rates.
  
  - [**GANs vs UED**](lab-notes/2026-01-22_gans-vs-ued/notes.md) – Exploring the analogy between GANs, UED, and EGT.

- `experiments/`  
  ⚗️ Minimal code used to answer specific questions from the notes.

- `foundations/`  
  🏗️ Background work rebuilding core deep learning machinery (JAX, Flax, Optax, Orbax, backprop from scratch).
