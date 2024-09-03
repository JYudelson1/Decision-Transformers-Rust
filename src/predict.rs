use crate::{dt_model::Input, trait_helpers::get_rewards_to_go, DTModelWrapper, DTState};

use dfdx::prelude::*;
use rand::{distributions::WeightedIndex, thread_rng};
use rand_distr::Distribution;
use stack::StackKernel;

impl<
        E: Dtype
            + From<f32>
            + num_traits::Float
            + rand_distr::uniform::SampleUniform
            + for<'a> std::ops::AddAssign<&'a E>,
        D: Device<E> + DeviceBuildExt + dfdx::tensor::ZerosTensor<usize> + StackKernel<usize>,
        Game: DTState<E, D>,
    > DTModelWrapper<E, D, Game>
where
    [(); Game::MAX_EPISODES_IN_GAME]: Sized,
    [(); Game::EPISODES_IN_SEQ]: Sized,
    [(); Game::ACTION_SIZE]: Sized,
    [(); Game::STATE_SIZE]: Sized,
{
    pub fn make_move(
        &self,
        state_history: &Vec<Game>,
        action_history: &Vec<Game::Action>,
        temperature: E,
        desired_total_reward: f32,
    ) -> Game::Action
    where
        [(); 3 * { Game::EPISODES_IN_SEQ }]: Sized,
    {
        assert_eq!(state_history.len(), action_history.len() + 1);
        let dev: D = Default::default();

        // Update the rewards-to-go to total to the desired reward
        let state_without_last = Vec::from(&state_history[0..state_history.len() - 1]);
        let mut rewards_so_far = get_rewards_to_go(&state_without_last, &action_history);

        let total_reward = if rewards_so_far.len() > 0 {
            rewards_so_far[0]
        } else {
            0.0
        };

        if total_reward > desired_total_reward {
            panic!("Attempting to reach a reward of {desired_total_reward}, but reward is already {total_reward}");
        } else {
            let reward_diff = desired_total_reward - total_reward;
            for reward in rewards_so_far.iter_mut() {
                *reward += reward_diff;
            }
        }

        // build input
        let mut actions_in_seq: [Tensor<(Const<{ Game::ACTION_SIZE }>,), E, D>;
            Game::EPISODES_IN_SEQ] = std::array::from_fn(|_| dev.zeros());
        let mut states_in_seq: [Tensor<(Const<{ Game::STATE_SIZE }>,), E, D>;
            Game::EPISODES_IN_SEQ] = std::array::from_fn(|_| dev.zeros());
        let mut rtg_in_seq: [Tensor<(Const<1>,), E, D>; Game::EPISODES_IN_SEQ] =
            std::array::from_fn(|_| dev.zeros());
        let mut timesteps_in_seq: [Tensor<(), usize, D>; Game::EPISODES_IN_SEQ] =
            std::array::from_fn(|_| dev.zeros());

        // Build actions (last entry should be empty)
        let mut amt_to_use = { Game::EPISODES_IN_SEQ } - 1;
        if action_history.len() < amt_to_use {
            amt_to_use = action_history.len();
        }
        let actions_start = action_history.len() - amt_to_use;
        for (actions_index, seq_index) in
            (({ Game::EPISODES_IN_SEQ } - amt_to_use - 1)..Game::EPISODES_IN_SEQ - 1).enumerate()
        {
            let action = &action_history[actions_start + actions_index];
            actions_in_seq[seq_index] = Game::action_to_tensor(action);
        }

        // Build states
        let mut amt_to_use = Game::EPISODES_IN_SEQ;
        if state_history.len() < amt_to_use {
            amt_to_use = state_history.len();
        }
        let states_start = state_history.len() - amt_to_use;
        for (states_index, seq_index) in
            (({ Game::EPISODES_IN_SEQ } - amt_to_use)..Game::EPISODES_IN_SEQ).enumerate()
        {
            let state = &state_history[states_start + states_index];
            states_in_seq[seq_index] = state.to_tensor();
        }

        // Build rewards (last entry should be empty)
        assert_eq!(rewards_so_far.len(), action_history.len());
        let mut amt_to_use = { Game::EPISODES_IN_SEQ } - 1;
        if action_history.len() < amt_to_use {
            amt_to_use = action_history.len();
        }
        let rewards_start = action_history.len() - amt_to_use;
        for (rewards_index, seq_index) in
            (({ Game::EPISODES_IN_SEQ } - amt_to_use - 1)..Game::EPISODES_IN_SEQ - 1).enumerate()
        {
            let reward = rewards_so_far[rewards_index + rewards_start];
            rtg_in_seq[seq_index] = dev.tensor([reward.into()]);
        }

        // Build timesteps
        let mut amt_to_use = Game::EPISODES_IN_SEQ;
        if state_history.len() < amt_to_use {
            amt_to_use = state_history.len();
        }
        let time_start = state_history.len() - amt_to_use;
        for (time, seq_index) in
            (({ Game::EPISODES_IN_SEQ } - amt_to_use)..Game::EPISODES_IN_SEQ).enumerate()
        {
            timesteps_in_seq[seq_index] = dev.tensor(time_start + time)
        }

        let input: Input<
            { Game::EPISODES_IN_SEQ },
            { Game::STATE_SIZE },
            { Game::ACTION_SIZE },
            E,
            D,
            NoneTape,
        > = (
            states_in_seq.stack(),
            actions_in_seq.stack(),
            rtg_in_seq.stack(),
            timesteps_in_seq.stack(),
        );

        // Forward the input
        let out: Tensor<
            (
                Const<{ Game::EPISODES_IN_SEQ }>,
                Const<{ Game::ACTION_SIZE }>,
            ),
            E,
            D,
        > = self.0.forward(input);

        // Select the last segment
        let logits: Tensor<(Const<{ Game::ACTION_SIZE }>,), E, D> =
            out.select(dev.tensor(Game::EPISODES_IN_SEQ - 1)) / temperature;
        let probs: Tensor<(Const<{ Game::ACTION_SIZE }>,), E, D> = logits.softmax();

        let mut options = vec![];

        for (i, prob) in probs.as_vec().iter().enumerate() {
            options.push((Game::index_to_action(i), *prob))
        }
        let dist = WeightedIndex::new(options.iter().map(|item| item.1)).unwrap();

        options[dist.sample(&mut thread_rng())].0.clone()
    }
}
