# 🧠 Bio-Inspired Reinforcement Learning:  
## Dopamine-Modulated Synapses, Membrane Dynamics, and Sleep-Driven Consolidation  
*A research-notebook style implementation inspired by computational neuroscience principles.*

---

## 🌱 Motivation

This project explores a question that sits at the intersection of **neuroscience** and **reinforcement learning**:

> *How far can an agent learn using only biologically plausible mechanisms—without gradients,  
  without backprop, and without deep networks?*

Instead of conventional RL architectures, the agent relies on simplified neural processes:
- Membrane-potential–based action selection  
- Dopamine-like prediction errors  
- Synaptic plasticity rules reminiscent of Hebbian/TD learning  
- Sleep-driven synaptic downscaling  

The goal is not to achieve high performance but to illustrate how **neural principles** can give rise to adaptive behavior in a lightweight, interpretable system.

---

## 🧬 Biological Foundations

### **1. Membrane Potential as Computation**
Each action corresponds to a motor neuron receiving weighted sensory inputs:

\[
V_{a} = s \cdot W_{:,a}
\]

Where:  
- \( s \) is the 4-dimensional sensory state  
- \( W \) is the synaptic weight matrix  
- The chosen action is:  
\[
a = \arg\max(V_a)
\]

This resembles a **winner-take-all selection circuit**, common in basal ganglia networks.

---

### **2. Dopamine as Prediction Error**

Learning is driven by a dopamine-like signal computed as:

\[
\delta = r + \gamma \max(V') - V_a
\]

This scalar signal modulates synaptic updates:

\[
W_{i,a} \leftarrow W_{i,a} + \alpha \, \delta \, s_i
\]

This is similar to:
- **Dopamine-modulated Hebbian learning**  
- **Temporal Difference (TD) learning** in RL  

It reflects how real neural circuits combine local synaptic eligibility with global reward feedback.

---

### **3. Sleep-Induced Synaptic Consolidation**

Inspired by the **Synaptic Homeostasis Hypothesis (Tononi & Cirelli)**:

- After every 100 episodes, the agent enters a *“sleep phase.”*  
- Weak synapses are pruned:

\[
|W_{i,j}| < \epsilon \Rightarrow W_{i,j} = 0
\]

This prevents runaway growth and encourages a sparse, energy-efficient connectome.

---

## 🚀 Training Procedure

- Environment: **Gymnasium — CartPole v1**  
- Policy: winner-take-all membrane potential  
- Learning: dopamine-modulated TD rule  
- Regularization: sleep-driven pruning  
- Safety: weight clipping to keep synaptic dynamics stable  

This leads to a model that is:
- Lightweight  
- Fully interpretable  
- Biologically grounded  
- Easy to experiment with  

---

## 📊 Results

The project produces two main visualizations:

### **1. Behavioral Learning Curve**
Shows:
- Raw trial rewards  
- Smoothed performance  
- Success threshold  
- Sleep cycles (vertical dotted lines)  

The dynamics resemble noisy biological learning rather than clean algorithmic optimization.

### **2. Synaptic Connectivity Map**
A heatmap of the final 4×2 sensory-to-motor weight matrix:

- Inputs: Position, Velocity, Angle, Angular Velocity  
- Outputs: Push Left, Push Right  

This matrix acts as the agent’s *“brain.”*

<img width="1400" height="600" alt="Figure_4" src="https://github.com/user-attachments/assets/7f4c5075-fec9-4c94-912e-e074a0f2eed0" />
 

```markdown
![Behavior and Brain](results/behavior_and_brain.png)
# BioInspired-RL-CartPole-Dopamine-STDP
