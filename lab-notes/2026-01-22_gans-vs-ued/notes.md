# GANs vs UED

## Question

To what extent are GANs analogous to UED?

## Background: the GAN-RL analogy

Why might GAN training dynamics be a useful analogy to RL? Ian Goodfellow's 2016 NeurIPS tutorial explains how we can think of the generator as an RL agent who chooses an action - e.g. which image to generate - and receives a scalar reward from the "value function" of the discriminator. As in RL, the generator agent has no examples of which actions are best - it simply has to guess-and-check against the reward model. But unlike in standard RL or RLHF, the reward model itself is non-stationary: it adversarially adapts to the agent to minimise the agent's reward, in a way that feels similar to the adversarially-generated environments of UED. 

## Comparing GANs and UED

However, in this analogy, GANs are different to UED in five ways:

1. In GANs, the adversary (discriminator) is changing the *rewards* for a given action (no state), whereas in UED, the adversary is changing the *environments* (distributions of input states and the transition dynamics between them). These might be comparable in partially-observable environments, where the same state can lead to different rewards (e.g. depending on what's "behind the curtain" in a partially-observed maze).

1. In GANs, one player is an RL agent and the other is a supervised learning model, and has to distinguish between labelled examples of true and fake things; In UED, both agents are learning through trial-and-error (RL). This is what makes UED powerful because the generator is *unsupervised*, so it's unconstrained by human intuitions about what a good environment (or reward function) should look like. 

1. In GANs, the agent gets access not only to the reward (the judgement of the reward model), but also the *gradient* of the reward model (Goodfellow NeurIPS tutorial). This feels similar to the action-value function in RL because it can see how much more reward it would have gotten from selecting a slightly different action (e.g. generating a slightly different image). 

1. In GANs, the state (the latent vector) is random and not dependent on previous actions - i.e. it is closer to a bandit problem than to the true temporal-sequence RL problem of UED. However, the dimensionality of the latent vector might be an interesting way to think about how the dimensionality of the environment might affect training dynamics in UED. 

1. In GANs, the two players' objectives are strictly adversarial—the discriminator always wants to minimise what the generator wants to maximise, so they always push against each other. In UED, particularly with learnability-based methods such as [Rutherford et al. 2024](https://arxiv.org/abs/2408.15099), the objectives aren't always adversarial - in some cases the learner and the teacher's objectives can push in the same direction (e.g. both towards higher completion rates in low-completion levels).

## Links to evolutionary biology

We can extend each of the above GAN-UED analogies to evolutionary game theory (EGT) [WIP]:

4. In EGT, the dimensionality of the latent vector of the generator agent might link to the effect of dimensionality of phenotype spaces in maintaining diversity through frequency-dependent selection (Doebeli and Ispolatov 2010 Science).