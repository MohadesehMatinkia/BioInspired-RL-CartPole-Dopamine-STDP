# 🧠 Bio-Inspired Reinforcement Learning  
## Dopamine-Modulated Synapses, Membrane Dynamics & Sleep-Based Consolidation

*A computational neuroscience–inspired CartPole agent built without neural networks, gradients, or backpropagation — only biologically plausible learning rules.*

---

## 🌱 Motivation

This project explores a fundamental question in computational neuroscience:

**Can meaningful behavior emerge from simple biological learning rules alone?**

Instead of artificial neural networks, the agent relies on mechanisms inspired by the brain:

- Membrane-potential–based action selection  
- Dopamine-like reward prediction errors  
- Local Hebbian/TD synaptic updates  
- Sleep-based synaptic downscaling for stability  

The goal is not algorithmic performance, but **interpretability** and **biological realism** in reinforcement learning.

---

## 🧬 Biological Inspiration

### **1. Membrane-Potential Action Selection**

The agent uses a small **4×2 synaptic weight matrix** to map sensory inputs to motor outputs.

Membrane potential for each action:

$$
V_a = s \cdot W_{:,a}
$$

Action is chosen through a winner-take-all mechanism:

$$
a = \arg\max(V_a)
$$

This mirrors competitive action selection in basal ganglia circuits.

---

### **2. Dopamine as Temporal-Difference Error**

Learning is driven by a dopamine-like scalar signal:

$$
\delta = r + \gamma \max(V') - V_a
$$

Synaptic updates occur only on the active input-to-action pathway:

$$
W_{i,a} \leftarrow W_{i,a} + \alpha \, \delta \, s_i
$$

This combination of **local synaptic eligibility** + **global dopamine modulation** reflects key biological credit-assignment principles.

---

### **3. Sleep-Based Synaptic Pruning**

Every 100 episodes, the agent enters a simulated “sleep” stage.

Weak synapses are pruned:

$$
|W_{i,j}| < \epsilon \Rightarrow W_{i,j} = 0
$$

Inspired by the **Synaptic Homeostasis Hypothesis (SHY)**, this prevents weight explosion, reduces noise, and supports long-term stability.

---

## 🚀 Training Overview

- Environment: **Gymnasium CartPole-v1**  
- Learning: dopamine-modulated TD update  
- Action selection: membrane potentials + argmax  
- Regularization: synaptic pruning during sleep  
- Stability: weight clipping between –5 and +5  

This system is:
- Lightweight  
- Fully interpretable  
- CPU-friendly (no GPU required)  
- Neuroscience-inspired rather than algorithm-driven  

---

## 📊 Results

### **Behavioral Performance**

The learning curve shows:
- Raw episode returns (grey)  
- Smoothed reward trajectory (blue)  
- Success threshold (green dashed)  
- Sleep cycles (purple vertical lines)  

Behavior reflects noisy but adaptive biological learning, rather than clean optimization.

### **Synaptic Connectivity “Brain Map”**

The final 4×2 weight matrix reveals how each sensory input influences the two actions.

<img width="1400" height="600" alt="Figure_4" src="https://github.com/user-attachments/assets/2e9624f7-7a7e-457f-b9ee-82db44bc9660" />

