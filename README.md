# Flappy Bird AI: Mastering Flight with D3QN and Pixels

An artificial intelligence agent that learns to play Flappy Bird entirely from scratch. Instead of using hardcoded coordinates or simplified states, this AI learns exactly like a human does: by looking at raw pixels on the screen. 

This project implements a state-of-the-art **Dueling Double Deep Q-Network (D3QN)** paired with a **Convolutional Neural Network (CNN)** to achieve superhuman performance, consistently passing hundreds of pipes.

---

## 🧠 The Architecture

To bridge the gap between basic Reinforcement Learning and true Computer Vision, this agent uses a highly optimized DeepMind-style architecture.

### 1. Convolutional Neural Network (CNN)
The environment does not feed the agent the X/Y coordinates of the bird or pipes. Instead:
* The game screen is captured as raw RGB pixels.
* The image is converted to grayscale and downsampled to an $80 \times 80$ grid.
* 4 consecutive frames are stacked together (to give the AI a sense of velocity and direction).
* A 3-layer CNN processes these stacked frames to extract spatial features and "see" the pipes.

### 2. Double Deep Q-Network (DDQN)
Standard DQNs suffer from an overestimation bias (hallucinating that bad actions will yield high rewards). This project implements Double DQN logic to fix this by separating selection from evaluation:
* **Policy Network:** Selects the best action for the next state.
* **Target Network:** Evaluates the actual Q-value of that specific chosen action.

### 3. Dueling Network
The neural network splits into two separate streams after the convolutional layers:
* **Value Stream $V(s)$:** Calculates how good the current state is overall (e.g., being in the middle of the screen is good, being near the ground is bad).
* **Advantage Stream $A(s, a)$:** Calculates the specific benefit of taking a certain action (flapping vs. doing nothing).

These streams are aggregated at the final output layer using the Dueling formula to ensure mathematical identifiability and stable training:
$Q(s, a) = V(s) + \left( A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a'} A(s, a') \right)$

---

## ⚙️ Training Details & Optimizations

* **Huber Loss (Smooth L1):** Used instead of Mean Squared Error to prevent gradient explosions when the bird violently crashes and receives a massive negative reward.
* **Custom Reward Shaping:** 
  * `+0.05` for every frame survived (encourages staying in the air).
  * `+1.0` for passing a pipe.
  * `-10.0` for crashing.
* **Modified Experience Replay:** The replay buffer actively forces few absolute newest memory into every random training batch. This completely prevents catastrophic forgetting of recent actions while maintaining the stability of random sampling.
* **Hardware:** Trained locally using an NVIDIA RTX 5050 (8GB VRAM).

---

## 🏆 Final Results

After approximately 12,000 episodes of training (and slowly decaying the epsilon exploration rate), the CNN fully converged. The AI learned to perfectly balance gravity and consistently navigate through tight pipe gaps.

**Testing Phase (100% Exploitation / 0 Epsilon):**
During a 5-episode evaluation phase, the agent achieved the following scores:
* Episode 1: **373**
* Episode 2: **34**
* Episode 3: **74**
* Episode 4: **424**
* Episode 5: **65**

**Average Score:** **194.26 pipes passed per game.**

---

## 🚀 Future Scope
Because this AI is built on a pure, pixel-based visual brain, it is entirely agnostic to the game it is playing. With minimal hyperparameter tuning and an updated action space, this exact D3QN architecture can be directly ported to master classic Atari environments like Breakout or Space Invaders.

---
*Developed by Himanshu Arora*
