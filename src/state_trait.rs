use dfdx::prelude::*;
use stack::StackKernel;

use crate::{
    dt_model::{BatchedInput, Input},
    trait_helpers::{batch_inputs, game_to_inputs},
};

pub trait DTState<E: Dtype + From<f32>, D: Device<E>>: Clone {
    type Action: Clone;
    const STATE_SIZE: usize;
    const ACTION_SIZE: usize;
    const MAX_EPISODES_IN_SEQ: usize;

    fn apply_action(&mut self, action: Self::Action);

    fn get_reward(&self, action: Self::Action) -> f32;

    fn to_tensor(&self) -> Tensor<(Const<{ Self::STATE_SIZE }>,), E, D>;

    fn action_to_tensor(action: &Self::Action) -> Tensor<(Const<{ Self::ACTION_SIZE }>,), E, D>;
}

pub trait GetOfflineData<
    E: Dtype + From<f32>,
    D: Device<E> + dfdx::tensor::ZerosTensor<usize> + StackKernel<usize>,
>: DTState<E, D>
{
    /// Required method
    fn play_one_game<R: rand::Rng + ?Sized>(rng: &mut R) -> (Vec<Self>, Vec<Self::Action>);

    /// Provided method
    fn get_batch<const B: usize, R: rand::Rng + ?Sized>(
        rng: &mut R,
    ) -> (BatchedInput<
        { Self::MAX_EPISODES_IN_SEQ },
        B,
        { Self::STATE_SIZE },
        { Self::ACTION_SIZE },
        E,
        D,
        NoneTape,
    >, [Self::Action; B]){
        let dev: D = Default::default();
        let mut batch: [Input<
            { Self::MAX_EPISODES_IN_SEQ },
            { Self::STATE_SIZE },
            { Self::ACTION_SIZE },
            E,
            D,
            NoneTape,
        >; B] = std::array::from_fn(|_| (dev.zeros(), dev.zeros(), dev.zeros(), dev.zeros()));

        let mut num_examples = 0;
        let mut true_actions: [Option<Self::Action>; B] = std::array::from_fn(|_| None);
        let mut num_actions = 0;

        while num_examples < B {
            // Play one game
            let (states, actions) = Self::play_one_game(rng);

            // Update true actions (for training)
            for action in actions.iter() {
                true_actions[num_actions] = Some(action.clone());
                num_actions += 1;
            }

            // Turn into tensor inputs
            let mut inputs = game_to_inputs(states, actions, &dev);

            // Throw away inputs above size B
            let len = inputs.len();
            inputs.truncate(B - num_examples);

            // Add the examples to the batch
            for (i, input) in inputs.into_iter().enumerate() {
                let batch_i = num_examples + i;
                batch[batch_i] = input;
            }

            // Mark down the number added
            num_examples += len;
        }

        let true_actions = true_actions.map(|inner| inner.unwrap());

        let batched_inputs = batch_inputs(batch, &dev);

        (batched_inputs, true_actions)
    }
}
