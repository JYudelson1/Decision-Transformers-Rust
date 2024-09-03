// use std::mem::MaybeUninit;

// use dfdx::prelude::*;

// use crate::conversions::{action_to_tensor, state_to_tensor};
// use crate::dt_model::{Input, RewardsToGo, TimeSteps};

// pub fn get_samples<
//     const MAX_EPISODES_IN_SEQ: usize,
//     const S: usize,
//     const A: usize,
//     F,
//     E: Dtype,
//     D: Device<E> + dfdx::tensor::ZerosTensor<usize>,
// >(
//     f: F,
// ) -> Vec<Input<MAX_EPISODES_IN_SEQ, S, A, E, D>> {
//     let dev: D = Default::default();
//     let mut rtg_tensor: RewardsToGo<MAX_EPISODES_IN_SEQ, E, D> = dev.zeros();
//     let mut timesteps: TimeSteps<MAX_EPISODES_IN_SEQ, D> = dev.zeros();

//     // Get rewards
//     let mut backwards_rewards = vec![];
//     let mut rewards_so_far = 0.0;
//     assert_eq!(actions.len(), states.len());
//     for i in 0..actions.len() {
//         let reward = get_reward(&states[i], actions[i].clone());
//         rewards_so_far += reward;
//         backwards_rewards.push(rewards_so_far);
//     }
//     backwards_rewards.reverse();
//     for i in 0..actions.len() {
//         println!("i: {i}, reward: {}", backwards_rewards[i]);
//         rtg_tensor[[i, 0]] = backwards_rewards[i];
//     }
//     println!("{rtg_tensor:?}");

//     let mut all_states = zeros_in_shape::<S>(&dev);
//     let mut all_actions = zeros_in_shape::<A>(&dev);

//     // Update each state and action that exists
//     for i in 0..actions.len() {
//         let action_tensor = action_to_tensor(&actions[i]);
//         let state_tensor = state_to_tensor(&states[i]);
//         all_states[i] = state_tensor;
//         all_actions[i] = action_tensor;

//         // Also, timesteps
//         timesteps[[i]] = i;
//     }

//     (
//         all_states.stack(),
//         all_actions.stack(),
//         rtg_tensor,
//         timesteps,
//     )
// }
// // TAKE a vec of actions and of states
// // At each time t, find the reward
// // Then add that reward to each previous reward
