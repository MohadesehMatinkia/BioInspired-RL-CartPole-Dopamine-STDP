# 🧠 Bio-Inspired Reinforcement Learning  
## Dopamine-Modulated Synapses, Membrane Dynamics & Sleep-Driven Consolidation  
*A research-notebook style experiment inspired by computational neuroscience.*

---

## 🌱 Motivation

This project explores a central question in computational neuroscience:

**Can an agent learn meaningful behavior using only biologically plausible mechanisms—without gradients, backpropagation, or deep networks?**

Instead of artificial neural networks, this implementation follows principles drawn from real neural systems:

- Action selection through **membrane potentials**  
- Learning via **dopamine-modulated prediction errors**  
- Local synaptic plasticity inspired by **Hebbian/TD rules**  
- Night-time consolidation through **sleep-based synaptic pruning**  

The result is a lightweight yet interpretable agent capable of balancing the CartPole environment using minimal but biologically grounded computations.

---

## 🧬 Biological Inspiration

### **1. Membrane-Potential Action Selection**

Each action corresponds to a motor neuron receiving weighted sensory inputs:

\[
V_{a} = s \cdot W_{:,a}
\]

Where:  
- \( s \) — 4-dimensional state (position, velocity, angle, angular velocity)  
- \( W \) — synaptic weight matrix (4×2)  
- \( V_a \) — membrane potential for action \(a\)

The chosen action:

\[
a = \arg\max(V_a)
\]

This approximates a **winner-take-all** mechanism similar to basal ganglia circuits.

---

### **2. Dopamine as Temporal-Difference Error**

Learning is driven by a dopamine-like prediction signal:

\[
\delta = r + \gamma \max(V') - V_a
\]

Weights update only for synapses linked to the chosen action:

\[
W_{i,a} \leftarrow W_{i,a} + \alpha \, \delta \, s_i
\]

This is a simplified form of:

- Dopamine-modulated Hebbian learning  
- Temporal Difference (TD) learning  

It captures the interaction between **local synaptic eligibility** and **global reward feedback**, widely observed in biological systems.

---

### **3. Sleep-Based Synaptic Consolidation**

Inspired by Tononi & Cirelli’s *Synaptic Homeostasis Hypothesis (SHY)*:

Every 100 episodes, the agent “sleeps”, pruning weak synapses:

\[
|W_{i,j}| < \epsilon \Rightarrow W_{i,j} = 0
\]

This maintains network stability, energy efficiency, and prevents uncontrolled synaptic growth.

---

## 🚀 Training Procedure

- **Environment:** Gymnasium CartPole-v1  
- **Policy:** membrane potentials + winner-take-all  
- **Learning:** dopamine-modulated TD error  
- **Regularization:** sleep-driven synaptic pruning  
- **Stability:** weight clipping (-5 to +5)  

The system is fully interpretable and computationally lightweight—no GPU required.

---

## 📊 Results

### **1. Behavioral Performance**

- Grey: raw trial rewards  
- Blue: smoothed learning curve  
- Green dashed: success threshold  
- Purple vertical lines: sleep cycles  

Learning remains noisy yet adaptive, similar to biological learning rather than engineered optimization.

### **2. Synaptic Connectivity Map**

A heatmap of the final 4×2 synaptic weight matrix:

- **Inputs:** Position, Velocity, Angle, Angular Velocity  
- **Actions:** Push Left, Push Right  

This matrix represents the agent’s “brain”.
