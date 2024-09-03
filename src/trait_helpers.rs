use std::array;

use dfdx::prelude::*;

use crate::{
    dt_model::{BatchedInput, Input},
    DTState,
    //sts::get_samples,
};

pub fn batch_inputs<
    const MAX_EPISODES_IN_SEQ: usize,
    const B: usize,
    const S: usize,
    const A: usize,
    E: Dtype,
    D: Device<E>,
>(
    inputs: [Input<MAX_EPISODES_IN_SEQ, S, A, E, D, NoneTape>; B],
    device: &D,
) -> BatchedInput<MAX_EPISODES_IN_SEQ, B, S, A, E, D, NoneTape> {
    let dev: Cpu = Default::default();
    let mut states = array::from_fn(|_| dev.zeros());
    let mut actions = array::from_fn(|_| dev.zeros());
    let mut rewards = array::from_fn(|_| dev.zeros());
    let mut times = array::from_fn(|_| dev.zeros());

    for (i, (state, action, reward, timestep)) in inputs.into_iter().enumerate() {
        states[i] = state.to_device(&dev);
        actions[i] = action.to_device(&dev);
        rewards[i] = reward.to_device(&dev);
        times[i] = timestep.to_device(&dev);
    }

    (
        states.stack().to_device(device),
        actions.stack().to_device(device),
        rewards.stack().to_device(device),
        times.stack().to_device(device),
    )
}

pub fn game_to_inputs<E: Dtype, D: Device<E>, Game: DTState<E, D>>(
    states: Vec<Game>,
    actions: Vec<Game::Action>,
    dev: &D
) -> Vec<
    Input<
        { Game::MAX_EPISODES_IN_SEQ },
        { Game::STATE_SIZE },
        { Game::ACTION_SIZE },
        E,
        D,
        NoneTape,
    >,
>{
    let mut inputs = vec![];

    todo!();
    inputs
}

pub fn next_sequence<
    const LEN: usize,
    E: Dtype,
    D: Device<E>,
    Game: DTState<E, D>,
    T: Tape<E, D>,
>(
    mut seq: [Tensor<(Const<{ Game::MAX_EPISODES_IN_SEQ }>, Const<LEN>), E, D, T>; {
        Game::MAX_EPISODES_IN_SEQ
    }],
    new_last_element: Tensor<(Const<{ Game::MAX_EPISODES_IN_SEQ }>, Const<LEN>), E, D, T>,
) -> [Tensor<(Const<{ Game::MAX_EPISODES_IN_SEQ }>, Const<LEN>), E, D, T>; {
       Game::MAX_EPISODES_IN_SEQ
   }] {
    seq.rotate_right(1);
    seq[seq.len() - 1] = new_last_element;
    seq
}
