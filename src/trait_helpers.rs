use std::array;

use crate::{
    state_trait::HumanEvaluatable, trait_helpers::stack::StackKernel, DTModel, GetOfflineData,
};
use dfdx::{optim::Adam, prelude::*};
use num_traits::Float;
use rand_distr::uniform::SampleUniform;

use crate::{
    dt_model::{BatchedInput, Input},
    DTState,
    //sts::get_samples,
};

pub fn batch_inputs<
    const EPISODES_IN_SEQ: usize,
    const B: usize,
    const S: usize,
    const A: usize,
    E: Dtype,
    D: Device<E> + StackKernel<usize> + dfdx::tensor::ZerosTensor<usize>,
>(
    inputs: [Input<EPISODES_IN_SEQ, S, A, E, D, NoneTape>; B],
    device: &D,
) -> BatchedInput<EPISODES_IN_SEQ, B, S, A, E, D, NoneTape> {
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
        times.stack(),
    )
}

pub fn game_to_inputs<E: Dtype + From<f32>+ num_traits::Float + rand_distr::uniform::SampleUniform, D: Device<E> + dfdx::tensor::ZerosTensor<usize> + StackKernel<usize>, Game: DTState<E, D>>(
    states: Vec<Game>,
    actions: Vec<Game::Action>,
    dev: &D
) -> Vec<
    Input<
        { Game::EPISODES_IN_SEQ },
        { Game::STATE_SIZE },
        { Game::ACTION_SIZE },
        E,
        D,
        NoneTape,
    >,
>{
    let mut inputs = vec![];

    let rewards_to_go = get_rewards_to_go(&states, &actions);

    let mut actions_in_seq: [Tensor<(Const<{ Game::ACTION_SIZE }>,), E, D>; Game::EPISODES_IN_SEQ] =
        std::array::from_fn(|_| dev.zeros());
    let mut states_in_seq: [Tensor<(Const<{ Game::STATE_SIZE }>,), E, D>; Game::EPISODES_IN_SEQ] =
        std::array::from_fn(|_| dev.zeros());
    let mut rtg_in_seq: [Tensor<(Const<1>,), E, D>; Game::EPISODES_IN_SEQ] =
        std::array::from_fn(|_| dev.zeros());
    let mut timesteps_in_seq: [Tensor<(), usize, D>; Game::EPISODES_IN_SEQ] =
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
            masked_next(&rtg_in_seq, dev).stack(),
            timesteps_in_seq.clone().stack(),
        );
        inputs.push(input)
    }

    inputs
}

fn next_sequence<
    E: Dtype + From<f32> + num_traits::Float + rand_distr::uniform::SampleUniform,
    D: Device<E>,
    Game: DTState<E, D>,
    T,
>(
    seq: &mut [T; Game::EPISODES_IN_SEQ],
    new_last_element: T,
) {
    seq.rotate_left(1);
    seq[seq.len() - 1] = new_last_element;
}

pub fn get_rewards_to_go<E: Dtype + From<f32>+ num_traits::Float + rand_distr::uniform::SampleUniform , D: Device<E>, Game: DTState<E, D>>(
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

fn masked_next<E: Dtype + From<f32>+ num_traits::Float + rand_distr::uniform::SampleUniform, D: Device<E>, Game: DTState<E, D>, S: ConstShape>(
    seq: &[Tensor<S, E, D>; Game::EPISODES_IN_SEQ],
    dev: &D,
) -> [Tensor<S, E, D>; Game::EPISODES_IN_SEQ]{
    let mut new_seq = seq.clone();
    new_seq[new_seq.len() - 1] = dev.zeros();
    new_seq
}

pub struct DTModelWrapper<
    E: Dtype + From<f32> + Float + SampleUniform,
    D: Device<E>,
    Game: DTState<E, D>,
>(pub DTModel<{ Game::MAX_EPISODES_IN_GAME }, { Game::STATE_SIZE }, { Game::ACTION_SIZE }, E, D>)
where
    [(); Game::MAX_EPISODES_IN_GAME]: Sized,
    [(); Game::ACTION_SIZE]: Sized,
    [(); Game::STATE_SIZE]: Sized;

impl<
        E: Dtype
            + From<f32>
            + num_traits::Float
            + rand_distr::uniform::SampleUniform
            + for<'a> std::ops::AddAssign<&'a E>,
        D: Device<E> + DeviceBuildExt + dfdx::tensor::ZerosTensor<usize> + StackKernel<usize>,
        Game: DTState<E, D> + HumanEvaluatable<E, D>,
    > DTModelWrapper<E, D, Game>
where
    [(); Game::MAX_EPISODES_IN_GAME]: Sized,
    [(); Game::EPISODES_IN_SEQ]: Sized,
    [(); Game::ACTION_SIZE]: Sized,
    [(); Game::STATE_SIZE]: Sized,
    [(); 3 * { Game::EPISODES_IN_SEQ }]: Sized,
{
    pub fn evaluate(&self, mut starting_state: Game) {
        let mut state_history = vec![starting_state.clone()];
        let mut action_history = vec![];

        starting_state.print();

        while starting_state.is_still_playing() {
            let action = self.make_move(&state_history, &action_history, 1.0.into(), 5.0);
            action_history.push(action.clone());

            Game::print_action(&action);

            starting_state.apply_action(action);
            state_history.push(starting_state.clone());

            starting_state.print()
        }
    }

    fn play_one_game<R: rand::Rng + ?Sized>(&self, temp: E, desired_reward: f32, rng: &mut R) -> (Vec<Game>, Vec<Game::Action>) {
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

    pub fn online_learn<const B: usize, R: rand::Rng + ?Sized>(&mut self, temp: E, desired_reward: f32, optimizer: &mut Adam<
        DTModel<{ Game::MAX_EPISODES_IN_GAME }, { Game::STATE_SIZE }, { Game::ACTION_SIZE }, E, D>,
        E,
        D,
    >,
    dev: &D,
    rng: &mut R) -> E{
        let (batch, actual) = get_batch_from_fn(rng, |rng| self.play_one_game(temp, desired_reward, rng));

        self.train_on_batch::<B>(batch, actual, optimizer, dev)
    }
}

pub fn get_batch_from_fn<
    const B: usize,
    R: rand::Rng + ?Sized,
    F,
    E: Dtype + From<f32> + Float + SampleUniform,
    D: Device<E> + dfdx::tensor::ZerosTensor<usize> + StackKernel<usize>,
    Game: DTState<E, D>,
>(
    rng: &mut R,
    player_fn: F,
) -> (
    BatchedInput<
        { Game::EPISODES_IN_SEQ },
        B,
        { Game::STATE_SIZE },
        { Game::ACTION_SIZE },
        E,
        D,
        NoneTape,
    >,
    [Game::Action; B],
)
where
    F: Fn(&mut R) -> (Vec<Game>, Vec<Game::Action>),
{
    let dev: D = Default::default();
    let mut batch: [Input<
        { Game::EPISODES_IN_SEQ },
        { Game::STATE_SIZE },
        { Game::ACTION_SIZE },
        E,
        D,
        NoneTape,
    >; B] = std::array::from_fn(|_| (dev.zeros(), dev.zeros(), dev.zeros(), dev.zeros()));

    let mut num_examples = 0;
    let mut true_actions: [Option<Game::Action>; B] = std::array::from_fn(|_| None);
    let mut num_actions = 0;

    while num_examples < B {
        // Play one game
        let (states, actions) = player_fn(rng);

        // Update true actions (for training)
        for action in actions.iter() {
            if num_actions == B {
                break;
            }
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
