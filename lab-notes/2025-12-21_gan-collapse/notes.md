# Constraints in GANs

## Question

How do constraints affect training dynamics in GANs?

## Idea

In ecological systems, diversity arises from *constraints*: trade-offs in fitness between different environments mean that there can be no one "Darwinian Demon" species that's optimal at everything, so species end up branching into specialists. 

Here we look at how constraints affect GAN training, hoping to provide some clues on how constraints might affect UED training and diversity generation in evolutionary game theory algorithms. 

## Problems: where GAN theory fails in practice

GANs, like UED, are grounded in a minimax game in theory, but training them requires approximations which may violate the assumptions of the minimax game in practice. Three important cases are:

**1. Non-zero sum games**

For a game to be zero-sum (and thus a minimax game), both agents have to be fighting over the exact same loss quantity (just with a different sign). 

If the two agents are trying to maximise different objectives, there's no guarantee they'll always push against each other in the same way, and the game is not truly minimax (see for example [The non-saturating loss function in GANs](../2025-12-17_gan-nonsaturating-loss/notes.md), also discussed in Goodfellow's NeurIPS tutorial).

There are however non-zero-sum games where the minimax *equilibria* are still the same - if the point(s) at which the two agents (minimiser and maximiser) *counterbalance* each other at the same points. This is the case for the non-saturating GAN loss.

**2. Representation constraints**

Even if there exists a mixed strategy that can perfectly generate the real data distribution in theory, there might not be a neural network that can represent this distribution in practice. This constraint arises from the fact that in the "infinite tabular" case of the original minimax decision theory, we can blend any set of solutions in whatever way we need to to concoct the unbeatable strategy, whereas in the applied deep learning case, the neural network itself it limited in both its "pure strategy" palette that it has available and the ability to do the blending, which it has to do as best it can internally. In mathematical jargon, this is known as non-convexity, and it can prevent us from obtaining the minimax equilibrium in either of two ways:

1. The minimax equilibrium cannot be *represented*: for example, we might find one set of parameters that generates a Gaussian distribution with mean $ \mu_1 $, and another set that generates one with mean  $ \mu_2 $, but not be able to linearly interpolate between those two parameter sets to produce a bimodal Gaussian with a 50-50 mix between the two. This can constrain the generator in two ways:

Key point: even if a neural network class is expressive, its strategy set is generally **not convex**.

If we take two generator parameter settings $\theta_1, \theta_2$, the corresponding generated distributions are
$$
p_{g_{\theta_1}}, \quad p_{g_{\theta_2}}.
$$

Their mixture distribution is
$$
\frac{1}{2} p_{g_{\theta_1}} + \frac{1}{2} p_{g_{\theta_2}}.
$$

In general, there does **not** exist a single parameter vector $\theta$ such that
$$
p_{g_\theta}
=
\frac{1}{2} p_{g_{\theta_1}} + \frac{1}{2} p_{g_{\theta_2}},
$$
unless the model class is explicitly designed to represent mixtures (e.g. via an explicit mixture latent or gating mechanism).

The same issue applies to discriminators: convex combinations of realizable functions are typically not realizable within the same parameterized class.

As a result, the classical game-theoretic assumption that players can freely play mixed strategies **within the same strategy class** no longer holds.

- Ability to *represent* the optimal adversarial distribution

**3. Convergence constraints**

1. The minimax equilibrium cannot be *reached*: for example, even if there is a set of parameters that can generate the bimodal distribution, *successive changes* in those parameters might not be able to reach the equilibrium in consecutive steps. This problem can be caused not just from an insufficiently flexible network, but also from aspects of the learning algorithm, so as learning rates that are too high and lead to the optimisation cycling endlessly around the equilibrium, never slowing down enough to land on it (Red Queen dynamics)

1. updates are made in function space so the objective can be non-convex

Even if there is a set of parameters that can generate the equilibrium distribution, *successive changes* in those parameters might not be able to reach the equilibrium in consecutive steps. This problem can be caused not just from an insufficiently flexible network, but also from aspects of the learning algorithm—such as learning rates that are too high and lead to the optimization cycling endlessly around the equilibrium, never slowing down enough to land on it (Red Queen dynamics).

The same applies to the discrimnator.

- Convergence problems: even if such strategies can be represented, the training dynamics might not be able to take us there.

## Experiments

We test these two types of constraints:

**Experiment 1: Representation limits** - Vary the latent vector dimensionality (64, 4, 1) to see how model capacity affects convergence to a bimodal Gaussian distribution.

**Experiment 2: Convergence limits** - Vary the learning rate (low, med, high) for 1D latent vectors to see how optimization dynamics affect convergence.

### Experiment 1: representation limits

- *Representation limits*: Limit the size of the GAN neural network and see if it affects an ability to converge to a known GAN (generate some synthetic data from e.g. a mixed Gaussian / bimodal distribution)

- We can vary the *latent vector* - raw material for generation, similar to dimensionality of an ecological system - or the *generator network architecture* - ability to turn that raw material into training-useful "level" differences

We vary the *latent vector* dimensionality—the raw material for generation, similar to dimensionality in an ecological system—to test whether representation capacity affects the generator's ability to match a known target distribution (a mixture of two Gaussians).

#### Figures (vary latent dimensionality, fixed learning rate)

Training dynamics for each latent dimensionality (lr=$10^{-3}$):

**Dim $z = 64$**

![Training dynamics (lr=3, dim=64)](figures/train_lr3_dim64.png)

**Dim $z = 4$**

![Training dynamics (lr=3, dim=4)](figures/train_lr3_dim4.png)

**Dim $z = 1$**

![Training dynamics (lr=3, dim=1)](figures/train_lr3_dim1.png)

Sample histograms for the same runs:

**Dim $z = 64$**

![Sample histogram (lr=3, dim=64)](figures/hist_lr3_dim64.png)

**Dim $z = 4$**

![Sample histogram (lr=3, dim=4)](figures/hist_lr3_dim4.png)

**Dim $z = 1$**

![Sample histogram (lr=3, dim=1)](figures/hist_lr3_dim1.png)

### Experiment 2: convergence limits

We vary the learning rate to test how optimization dynamics affect the generator's ability to reach equilibrium when representation capacity is severely limited (1D latent).

#### Figures (vary learning rate for 1D latent vectors)

Training dynamics for each learning rate (dim $z = 1$):

**Low lr=$10^{-4}$**

![Training dynamics (lr=2, dim=1)](figures/train_lr2_dim1.png)

**Med lr=$10^{-3}$**

![Training dynamics (lr=3, dim=1)](figures/train_lr3_dim1.png)

**High lr=$10^{-2}$**

![Training dynamics (lr=4, dim=1)](figures/train_lr4_dim1.output.png)

Sample histograms for the same runs:

**Low lr=$10^{-4}$**

![Sample histogram (lr=2, dim=1)](figures/hist_lr2_dim1.png)

**Med lr=$10^{-3}$**

![Sample histogram (lr=3, dim=1)](figures/hist_lr3_dim1.png)

**High lr=$10^{-2}$**

![Sample histogram (lr=4, dim=1)](figures/hist_lr4_dim1.png)

## Conclusions

Both experiments were designed to show how constraints on model structures and training algorithms lead to GANs "missing" the minimax equilibrium points predicted by the theory. This did happen, but not in the way I expected!

Both experiments were designed to show how constraints on model structures and training algorithms lead to GANs "missing" the minimax equilibrium points predicted by theory. The results were surprising!

**Representation limits:** 

- Representation limits: representation should *match the dimensionality of the system*. In other words, model size is not only a question of small model size constraining the system; making a model too *large* may also be a problem! 

The generator performed *better* with fewer latent dimensions (1D > 4D > 64D). This is counterintuitive—bigger models aren't always better.

A latent generator with more dimensions just has more flexibility in the randomness - A larger z makes it easier for the generator to produce a wide variety of outputs — but unless it can ensure all those outputs are high-quality, that variety increases the probability of hitting regions the discriminator rejects. The easiest way to reduce that risk is to ignore most of z and output a conservative, average-looking sample.

With a higher-dimensional latent space, the generator has more flexibility to produce diverse outputs, but this increases the risk of generating samples that fall outside the "safe basin" where the discriminator accepts them as real. The generator becomes risk-averse: rather than exploring the full space, it collapses to a conservative mode that reliably fools the discriminator. In other words, a larger latent space paradoxically leads to *less* diversity through mode collapse.

The discriminator defines a "safe basin" where samples look real. If G(z) spreads out, some mass falls outside the basin → gets punished.

The discriminator only sees one sample at a time - so it can't be like "all these IDs are the same" 

Generator becomes "risk averse" - penalty for wrong answers leads to mode collapse to stay in the safe region 

"with a bigger latent, the generator often has more incentive to "play it safe" by reducing output variance" - 

**Convergence limits:** 

Learning rate effects appear early in training when neither model is competent yet. The dynamics look qualitatively different from near-equilibrium behavior.

It might not be *absolute* learning rates that matter - a given change in learning rate might mean something different to the discriminator than the generator - e.g. because the discriminator finds it easy to find the separation manifold early whereas the generator needs to experiment wildly to have any chance of stumbling on a good solution

What matters may be *relative* learning rates between generator and discriminator, not absolute values. The discriminator can easily find the separation manifold early in training, while the generator must explore wildly to stumble on good solutions. Even with very small learning rates, games can cycle indefinitely because the gradient field has a strong rotational component (unlike convex optimization where gradients point downhill).

## Links to ecology

McNamara's convexity assumption might not hold in the sense that even with perfect mixing (at no cost), we might be constrained to a few phenotypes with particular distributions (e.g. normal distributions).

## Overall take-home

- Effects in the transient dynamics early in training, where neither model (and especially the generator) is good, can look very different from effects later in training when we're near the equilibrium.

- It may be worth examining two regimes in our EGT experiments: far from equilibrium and close to a (known) equilibrium.

- It may be *relative* learning rates that's the main determinant, not overall learning rate.

- Lower-dimensionality latent vectors (uniform random draws) mean fewer dials, fewer styles of levels.

## Follow-ups

- Do we get around the convexity issue by having separate policies and encoding the mix, i.e. $p(\text{select})$ explicitly?
- This is like a SoftMax over the QD cells?

## Notes

The idea of plotting the dynamics in phase space was inspired by [Lilian Weng](https://lilianweng.github.io/posts/2017-08-20-gan/)'s use of a Lotka-Volterra-equivalent model to illustrate the "Red Queen" dynamics of GAN training.

