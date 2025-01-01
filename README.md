# Decision Transformers in Rust

A fast, extensible implementation of Decision Transformers in Rust using dfdx. Based on the paper [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345).

## Overview

This crate provides a framework for implementing Decision Transformers in Rust. Decision Transformers frame reinforcement learning as a sequence prediction problem, allowing for more efficient learning from demonstration data. This implementation is built on top of dfdx for efficient tensor operations and automatic differentiation.

## Installation

```bash
cargo add decision-transformer-dfdx
```

## Quick Start

To implement a Decision Transformer for your own environment:

1. Define your configuration by implementing `DTModelConfig`
2. Define your environment's state representation
3. Define your action space (as an enum or similar)
4. Implement the `DTState` trait for your environment (required)
5. Implement `GetOfflineData` if you want to train from demonstrations
6. Implement `HumanEvaluatable` if you want to visualize/evaluate the environment
7. Create a training loop:
   - Initialize model and optimizer
   - Collect training data (offline or online)
   - Train model
   - Evaluate performance

For a complete working example, check out the [Snake Game Implementation](#snake-game-example).

## Core Concepts

### Traits

The framework is built around three main traits:

1. **DTState**: The core trait that defines your environment. Required for basic functionality.
```rust
pub trait DTState<E: Dtype, D: Device<E>, Config: DTModelConfig> {
    type Action: Clone;
    const STATE_SIZE: usize;    // Total number of floats needed to represent the state
    const ACTION_SIZE: usize;   // Total number of possible actions

    // Required methods
    fn new_random<R: rand::Rng + ?Sized>(rng: &mut R) -> Self;
    fn apply_action(&mut self, action: Self::Action);
    fn get_reward(&self, action: Self::Action) -> f32;
    fn to_tensor(&self) -> Tensor<(Const<{ Self::STATE_SIZE }>,), E, D>;
    fn action_to_index(action: &Self::Action) -> usize;
    fn index_to_action(action: usize) -> Self::Action;

    // Provided method
    fn action_to_tensor(action: &Self::Action) -> Tensor<(Const<{ Self::ACTION_SIZE }>,), E, D>;
    fn build_model() -> DTModelWrapper<E, D, Config, Self>;
}
```

2. **GetOfflineData**: For training from demonstration data.
```rust
pub trait GetOfflineData<E: Dtype, D: Device<E>, Config: DTModelConfig>: DTState<E, D, Config> {
    // Required method
    fn play_one_game<R: rand::Rng + ?Sized>(rng: &mut R) -> (Vec<Self>, Vec<Self::Action>);

    // Provided method
    fn get_batch<const B: usize, R: rand::Rng + ?Sized>(
        rng: &mut R,
        cap_from_game: Option<usize>
    ) -> (BatchedInput<B, { Self::STATE_SIZE }, { Self::ACTION_SIZE }, E, D, Config>, [Self::Action; B]);
}
```

3. **HumanEvaluatable**: For environments that can be visualized or evaluated by humans.
```rust
pub trait HumanEvaluatable<E: Dtype, D: Device<E>, Config: DTModelConfig>: DTState<E, D, Config> {
    // All methods required
    fn print(&self);                               // Print the current state
    fn print_action(action: &Self::Action);        // Print a given action
    fn is_still_playing(&self) -> bool;            // Check if episode is ongoing
}
```

### Configuration

The `DTModelConfig` trait allows you to configure the transformer architecture:

```rust
pub trait DTModelConfig {
    const NUM_ATTENTION_HEADS: usize;  // Number of attention heads
    const HIDDEN_SIZE: usize;          // Size of hidden layers
    const MLP_INNER: usize;            // Size of inner MLP layer (typically 4*HIDDEN_SIZE)
    const SEQ_LEN: usize;              // Length of sequence to consider
    const MAX_EPISODES_IN_GAME: usize; // Maximum episodes in a game
    const NUM_LAYERS: usize;           // Number of transformer layers
}
```

## Training Approaches

### Offline Learning

Train your model using pre-collected demonstration data:

```rust
let mut model = MyEnvironment::build_model();
let mut optimizer = Adam::new(&model.0, config);

// Get a batch of demonstration data
let (batch, actions) = MyEnvironment::get_batch::<1024, _>(&mut rng, Some(256));

// Train on the batch
let loss = model.train_on_batch(batch.clone(), actions, &mut optimizer);
```

### Online Learning

Train your model through self-play:

```rust
let temp = 0.5;  // Temperature for exploration
let desired_reward = 5.0;  // Target reward

// Train through self-play
let loss = model.online_learn::<100, _>(
    temp, 
    desired_reward, 
    &mut optimizer, 
    &mut rng,
    Some(256)  // Optional cap on episodes per game
);
```

## Snake Game Example

The repository includes a complete example implementing a snake game environment using the Decision Transformer framework. This serves as a reference implementation showing how to:

- Define your environment state and actions
- Implement the required traits
- Configure the model
- Train using both offline and online approaches
- Evaluate the trained model

Check out the [snake game implementation](https://github.com/JYudelson1/snake_dt) for a complete working example.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.