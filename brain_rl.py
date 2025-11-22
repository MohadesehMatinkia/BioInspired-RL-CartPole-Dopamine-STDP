import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

# --- Configuration ---
NUM_EPISODES = 600        # Total number of episodes for training
LEARNING_RATE = 0.02      # Synaptic plasticity rate
INPUT_NEURONS = 4         # Sensory inputs: Position, Velocity, Angle, Angular Velocity
OUTPUT_NEURONS = 2        # Motor outputs: Push Left, Push Right
SLEEP_INTERVAL = 100      # Interval for sleep/consolidation phase (every N episodes)

class SpikingBioAgent:
    def __init__(self, n_input, n_output):
        """
        Initializes the bio-inspired agent with a random synaptic weight matrix.
        """
        # Initialize weights randomly (mimicking an undeveloped brain)
        self.weights = np.random.rand(n_input, n_output) * 0.1
        # Keep a copy of initial weights for comparison if needed
        self.initial_weights = self.weights.copy() 
        
    def get_action(self, state):
        """
        Determines action based on sensory input using an Integrate-and-Fire mechanism.
        """
        # Calculate membrane potential for output neurons
        membrane_potential = np.dot(state, self.weights)
        
        # Neural Noise: Adds exploration (stochasticity) to the decision making
        if np.random.random() < 0.05:
            return np.random.choice([0, 1])
            
        # Winner-Take-All: The neuron with the highest potential fires (spikes)
        return np.argmax(membrane_potential)

    def train(self, state, action, reward, next_state):
        """
        Dopamine-Modulated STDP Learning Rule with Homeostatic Constraints.
        This function updates synaptic weights based on a reward signal (proxy for dopamine).
        """
        # Calculate the Prediction Error (Dopamine Signal/TD Error)
        # We estimate the value of the next state to guide the learning process.
        target = reward + 0.99 * np.max(np.dot(next_state, self.weights))
        current = self.weights[state > 0, action].sum()
        dopamine_signal = target - current
        
        # Synaptic Plasticity Update:
        # Update weights only for the chosen action and active input features (Eligibility Trace).
        # This mimics Hebbian learning modulated by a global reward signal.
        self.weights[:, action] += LEARNING_RATE * dopamine_signal * state
        
        # --- THE PRO FIX: Homeostatic Regulation (Prevent Explosion) ---
        # Clip weights to prevent them from growing indefinitely (Exploding Gradients).
        # In biology, synapses have physical limits on their strength.
        np.clip(self.weights, -5.0, 5.0, out=self.weights)
        
    def sleep_and_consolidate(self):
        """
        Simulates sleep phase. Prunes weak synaptic connections to reduce noise
        and consolidate memory (Synaptic Homeostasis Hypothesis).
        """
        pruning_threshold = 0.05
        
        # Count non-zero synapses before pruning
        before_count = np.count_nonzero(self.weights)
        
        # Set weak weights to zero
        self.weights[np.abs(self.weights) < pruning_threshold] = 0
        
        # Count non-zero synapses after pruning
        after_count = np.count_nonzero(self.weights)
        
        return before_count - after_count

def run_bio_rl():
    # Create the CartPole environment
    env = gym.make('CartPole-v1')
    agent = SpikingBioAgent(INPUT_NEURONS, OUTPUT_NEURONS)
    
    reward_history = []
    avg_rewards = []

    print(">>> Starting Pro-Level Bio-RL Training...")
    print("-" * 50)

    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # 1. Sensory Processing & Decision
            action = agent.get_action(state)
            
            # 2. Action Execution
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Apply penalty if the pole drops (fail state)
            if done and total_reward < 499:
                reward = -10
            
            # 3. Synaptic Plasticity (Learning)
            agent.train(state, action, reward, next_state)
            
            state = next_state
            total_reward += 1
            
        # Record metrics
        reward_history.append(total_reward)
        avg_rew = np.mean(reward_history[-50:])
        avg_rewards.append(avg_rew)
        
        # Log progress
        if episode % 50 == 0:
             print(f"Episode: {episode:4d} | Score: {total_reward:4.0f} | Avg Score: {avg_rew:4.1f}")

        # Sleep & Pruning Phase
        if episode % SLEEP_INTERVAL == 0 and episode > 0:
            pruned = agent.sleep_and_consolidate()
            print(f"Episode {episode}: [SLEEP] Consolidated Memory. Pruned {pruned} synapses.")

    env.close()
    return reward_history, avg_rewards, agent

def plot_results_and_brain(rewards, avg_rewards, agent):
    """
    Visualizes behavioral performance and the internal synaptic structure (Brain Map).
    """
    # Create a figure with two subplots
    fig = plt.figure(figsize=(14, 6))
    
    # --- Plot 1: Behavioral Performance ---
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(rewards, alpha=0.3, color='gray', label='Raw Reward')
    ax1.plot(avg_rewards, color='#007acc', linewidth=2.5, label='Learning Curve')
    ax1.axhline(y=195, color='green', linestyle='--', label='Success Threshold')
    
    # Mark Sleep Phases
    for i in range(0, NUM_EPISODES, SLEEP_INTERVAL):
        if i > 0:
            ax1.axvline(x=i, color='purple', linestyle=':', alpha=0.3)
            
    ax1.set_title('Behavioral Performance', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Balance Time')
    ax1.legend()
    ax1.grid(True, alpha=0.2)

    # --- Plot 2: Connectome (Brain Map) ---
    # This heatmap visualizes the learned synaptic weights
    ax2 = fig.add_subplot(1, 2, 2)
    
    # Display weights as a heatmap
    im = ax2.imshow(agent.weights, cmap='viridis', aspect='auto')
    
    # Customize the plot
    ax2.set_title('Synaptic Connectivity Map (The "Brain")', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Motor Neurons (Actions)')
    ax2.set_ylabel('Sensory Neurons (Inputs)')
    
    # Set ticks for interpretability
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(['Push Left', 'Push Right'])
    ax2.set_yticks([0, 1, 2, 3])
    ax2.set_yticklabels(['Position', 'Velocity', 'Angle', 'Ang. Vel.'])
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label('Synaptic Strength')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    raw, smooth, trained_agent = run_bio_rl()
    plot_results_and_brain(raw, smooth, trained_agent)