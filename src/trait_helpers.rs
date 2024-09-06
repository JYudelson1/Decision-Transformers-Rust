use std::{array, path::Path};

use crate::{
    state_trait::HumanEvaluatable,
    utils::{stack_usize, stack_usize_batched},
    DTModel, DTModelConfig,
};
use dfdx::{prelude::*, tensor::safetensors::SafeDtype};
use num_traits::Float;
use rand::seq::IteratorRandom;
use rand_distr::uniform::SampleUniform;

use crate::{
    dt_model::{BatchedInput, Input},
    DTState,
    //sts::get_samples,
};

pub fn batch_inputs<
    const B: usize,
    const S: usize,
    const A: usize,
    E: Dtype,
    D: Device<E> + dfdx::tensor::ZerosTensor<usize> + CopySlice<usize>,
    Config: DTModelConfig + 'static,
>(
    inputs: [Input<S, A, E, D, Config, NoneTape>; B],
    device: &D,
) -> BatchedInput<B, S, A, E, D, Config, NoneTape>
where
    [(); Config::MAX_EPISODES_IN_GAME]: Sized,
    [(); Config::SEQ_LEN]: Sized,
    [(); 3 * Config::SEQ_LEN]: Sized,
    [(); Config::HIDDEN_SIZE]: Sized,
    [(); Config::MLP_INNER]: Sized,
    [(); Config::NUM_ATTENTION_HEADS]: Sized,
{
    let mut states = array::from_fn(|_| device.zeros());
    let mut actions = array::from_fn(|_| device.zeros());
    let mut rewards = array::from_fn(|_| device.zeros());
    let mut times = array::from_fn(|_| device.zeros());

    for (i, (state, action, reward, timestep)) in inputs.into_iter().enumerate() {
        states[i] = state;
        actions[i] = action;
        rewards[i] = reward;
        times[i] = timestep;
    }

    (
        states.stack(),
        actions.stack(),
        rewards.stack(),
        stack_usize_batched(times, device),
    )
}

pub fn game_to_inputs<
    E: Dtype + From<f32> + num_traits::Float + rand_distr::uniform::SampleUniform,
    D: Device<E> + dfdx::tensor::ZerosTensor<usize>,
    Config: DTModelConfig + 'static,
    Game: DTState<E, D, Config>,
>(
    states: Vec<Game>,
    actions: Vec<Game::Action>,
    dev: &D,
) -> Vec<Input<{ Game::STATE_SIZE }, { Game::ACTION_SIZE }, E, D, Config, NoneTape>>
where
    [(); Config::MAX_EPISODES_IN_GAME]: Sized,
    [(); Config::SEQ_LEN]: Sized,
    [(); 3 * Config::SEQ_LEN]: Sized,
    [(); Config::HIDDEN_SIZE]: Sized,
    [(); Config::MLP_INNER]: Sized,
    [(); Config::NUM_ATTENTION_HEADS]: Sized,
{
    let mut inputs = vec![];

    let rewards_to_go = get_rewards_to_go(&states, &actions);

    let mut actions_in_seq: [Tensor<(Const<{ Game::ACTION_SIZE }>,), E, D>; Config::SEQ_LEN] =
        std::array::from_fn(|_| dev.zeros());
    let mut states_in_seq: [Tensor<(Const<{ Game::STATE_SIZE }>,), E, D>; Config::SEQ_LEN] =
        std::array::from_fn(|_| dev.zeros());
    let mut rtg_in_seq: [Tensor<(Const<1>,), E, D>; Config::SEQ_LEN] =
        std::array::from_fn(|_| dev.zeros());
    let mut timesteps_in_seq: [Tensor<(), usize, D>; Config::SEQ_LEN] =
        std::array::from_fn(|_| dev.zeros());

    for (i, (state, action)) in states.into_iter().zip(actions.into_iter()).enumerate() {
        // Update actions
        let new_action = Game::action_to_tensor(&action);
        next_sequence(&mut actions_in_seq, new_action);

        // Update states
        let new_state = state.to_tensor();
        next_sequence(&mut states_in_seq, new_state);

        // Update rtg
        let new_reward: E = rewards_to_go[i].into();
        next_sequence(&mut rtg_in_seq, dev.tensor([new_reward]));

        // Update timesteps
        next_sequence(&mut timesteps_in_seq, dev.tensor(i + 1));

        let input = (
            states_in_seq.clone().stack(),
            masked_next(&actions_in_seq, dev).stack(),
            rtg_in_seq.clone().stack(),
            stack_usize(timesteps_in_seq.clone(), &dev),
        );
        inputs.push(input)
    }

    inputs
}

fn next_sequence<Config: DTModelConfig + 'static, T>(
    seq: &mut [T; Config::SEQ_LEN],
    new_last_element: T,
) {
    seq.rotate_left(1);
    seq[seq.len() - 1] = new_last_element;
}

pub fn get_rewards_to_go<
    E: Dtype + From<f32> + num_traits::Float + rand_distr::uniform::SampleUniform,
    D: Device<E>,
    Config: DTModelConfig + 'static,
    Game: DTState<E, D, Config>,
>(
    states: &Vec<Game>,
    actions: &Vec<Game::Action>,
) -> Vec<f32> {
    let mut backwards_rewards = vec![];
    let mut rewards_so_far = 0.0;
    assert_eq!(actions.len(), states.len());
    for i in 0..actions.len() {
        let reward = Game::get_reward(&states[i], actions[i].clone());
        rewards_so_far += reward;
        backwards_rewards.push(rewards_so_far);
    }
    backwards_rewards.reverse();
    backwards_rewards
}

fn masked_next<
    E: Dtype + rand_distr::uniform::SampleUniform,
    D: Device<E>,
    Config: DTModelConfig + 'static,
    S: ConstShape,
>(
    seq: &[Tensor<S, E, D>; Config::SEQ_LEN],
    dev: &D,
) -> [Tensor<S, E, D>; Config::SEQ_LEN]{
    let mut new_seq = seq.clone();
    new_seq[new_seq.len() - 1] = dev.zeros();
    new_seq
}

pub struct DTModelWrapper<
    E: Dtype + From<f32> + Float + SampleUniform,
    D: Device<E>,
    Config: DTModelConfig + 'static,
    Game: DTState<E, D, Config>,
>(pub DTModel<Config, { Game::STATE_SIZE }, { Game::ACTION_SIZE }, E, D>)
where
    [(); Config::MAX_EPISODES_IN_GAME]: Sized,
    [(); Game::ACTION_SIZE]: Sized,
    [(); Game::STATE_SIZE]: Sized,
    [(); 3 * Config::SEQ_LEN]: Sized,
    [(); Config::SEQ_LEN]: Sized,
    [(); Config::HIDDEN_SIZE]: Sized,
    [(); Config::MLP_INNER]: Sized,
    [(); Config::NUM_LAYERS]: Sized,
    [(); Config::NUM_ATTENTION_HEADS]: Sized;

impl<
        E: Dtype
            + From<f32>
            + num_traits::Float
            + rand_distr::uniform::SampleUniform
            + for<'a> std::ops::AddAssign<&'a E>,
        D: Device<E> + DeviceBuildExt + dfdx::tensor::ZerosTensor<usize> + CopySlice<usize>,
        Config: DTModelConfig + 'static,
        Game: DTState<E, D, Config> + HumanEvaluatable<E, D, Config>,
    > DTModelWrapper<E, D, Config, Game>
where
    [(); Config::MAX_EPISODES_IN_GAME]: Sized,
    [(); Config::SEQ_LEN]: Sized,
    [(); Game::ACTION_SIZE]: Sized,
    [(); Game::STATE_SIZE]: Sized,
    [(); 3 * { Config::SEQ_LEN }]: Sized,
    [(); Config::MLP_INNER]: Sized,
    [(); Config::HIDDEN_SIZE]: Sized,
    [(); Config::NUM_ATTENTION_HEADS]: Sized,
    [(); Config::NUM_LAYERS]: Sized,
    [(); Config::HIDDEN_SIZE / Config::NUM_ATTENTION_HEADS]: Sized,
{
    pub fn evaluate(&self, mut starting_state: Game, temp: E, desired_reward: f32) {
        let mut state_history = vec![starting_state.clone()];
        let mut action_history = vec![];

        starting_state.print();

        while starting_state.is_still_playing() {
            let action = self.make_move(&state_history, &action_history, temp, desired_reward);
            action_history.push(action.clone());

            Game::print_action(&action);

            starting_state.apply_action(action);
            state_history.push(starting_state.clone());

            starting_state.print()
        }
    }

    fn play_one_game<R: rand::Rng + ?Sized>(
        &self,
        temp: E,
        desired_reward: f32,
        rng: &mut R,
    ) -> (Vec<Game>, Vec<Game::Action>) {
        let mut states = vec![];
        let mut actions = vec![];

        let mut state = Game::new_random(rng);
        while state.is_still_playing() {
            states.push(state.clone());
            let action = self.make_move(&states, &actions, temp, desired_reward);
            actions.push(action.clone());
            state.apply_action(action);
        }
        (states, actions)
    }

    pub fn online_learn<
        const B: usize,
        R: rand::Rng + ?Sized,
        O: Optimizer<DTModel<Config, { Game::STATE_SIZE }, { Game::ACTION_SIZE }, E, D>, D, E>,
    >(
        &mut self,
        temp: E,
        desired_reward: f32,
        optimizer: &mut O,
        dev: &D,
        rng: &mut R,
        cap_from_game: Option<usize>
    ) -> E
    where
        [(); Config::MAX_EPISODES_IN_GAME]: Sized,
        [(); Config::SEQ_LEN]: Sized,
        [(); 3 * Config::SEQ_LEN]: Sized,
        [(); Config::HIDDEN_SIZE]: Sized,
        [(); Config::MLP_INNER]: Sized,
        [(); Config::NUM_LAYERS]: Sized,
        [(); Config::NUM_ATTENTION_HEADS]: Sized,
        [(); Config::HIDDEN_SIZE / Config::NUM_ATTENTION_HEADS]: Sized,
    {
        let (batch, actual) =
            get_batch_from_fn(rng, |rng| self.play_one_game(temp, desired_reward, rng), cap_from_game);

        self.train_on_batch::<B, O>(batch, actual, optimizer, dev)
    }

    pub fn save<P: AsRef<Path>>(&self, path: P)
    where
        E: SafeDtype,
    {
        self.0.save_safetensors(path).unwrap()
    }

    pub fn load<P: AsRef<Path>>(&mut self, path: P)
    where
        E: SafeDtype,
    {
        self.0.load_safetensors(path).unwrap();
    }
}

pub fn get_batch_from_fn<
    const B: usize,
    R: rand::Rng + ?Sized,
    F,
    E: Dtype + From<f32> + Float + SampleUniform,
    D: Device<E> + dfdx::tensor::ZerosTensor<usize> + CopySlice<usize>,
    Config: DTModelConfig + 'static,
    Game: DTState<E, D, Config>,
>(
    rng: &mut R,
    player_fn: F,
    cap_from_game: Option<usize>
) -> (
    BatchedInput<B, { Game::STATE_SIZE }, { Game::ACTION_SIZE }, E, D, Config, NoneTape>,
    [Game::Action; B],
)
where
    F: Fn(&mut R) -> (Vec<Game>, Vec<Game::Action>),
    [(); Config::MAX_EPISODES_IN_GAME]: Sized,
    [(); Config::SEQ_LEN]: Sized,
    [(); 3 * Config::SEQ_LEN]: Sized,
    [(); Config::HIDDEN_SIZE]: Sized,
    [(); Config::MLP_INNER]: Sized,
    [(); Config::NUM_ATTENTION_HEADS]: Sized,
{
    let dev: D = Default::default();
    let mut batch: [Input<{ Game::STATE_SIZE }, { Game::ACTION_SIZE }, E, D, Config, NoneTape>; B] =
        std::array::from_fn(|_| (dev.zeros(), dev.zeros(), dev.zeros(), dev.zeros()));

    let mut num_examples = 0;
    let mut true_actions: [Option<Game::Action>; B] = std::array::from_fn(|_| None);
    let mut num_actions = 0;

    while num_examples < B {
        // Play one game
        let (states, actions) = player_fn(rng);
        let (mut states, mut actions) = if let Some(cap) = cap_from_game {
            let mut indices = (0..states.len()).choose_multiple(rng, cap);
            indices.sort();
            let mut indices_2 = indices.clone();
            let states: Vec<Game> = states.into_iter().enumerate().filter(|(i, _)| {
                if !indices.is_empty() && indices[0] == *i {
                    indices.remove(0);
                    true
                } else {
                    false
                }
            }).map(|(_, element)| element).collect();

            let actions: Vec<Game::Action> = actions.into_iter().enumerate().filter(|(i, _)| {
                if !indices_2.is_empty() && indices_2[0] == *i {
                    indices_2.remove(0);
                    true
                } else {
                    false
                }
            }).map(|(_, element)| element).collect();

            (states, actions)
        } else {
            (states, actions)
        };

        // Update true actions (for training)
        for action in actions.iter() {
            if num_actions == B {
                break;
            }
            true_actions[num_actions] = Some(action.clone());
            num_actions += 1;
        }

        // Throw away inputs above size B
        actions.truncate(B - num_examples);
        states.truncate(B - num_examples);
        // Turn into tensor inputs
        
        let inputs = game_to_inputs(states, actions, &dev);

        
        let len = inputs.len();

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
