use dfdx::prelude::*;

use crate::{dt_model::Input, DTModelConfig, DTState};

pub(crate) fn stack_usize<E: Dtype, D: Device<E>, Config: DTModelConfig>(
    tensors: [Tensor<(), usize, D>; Config::SEQ_LEN],
    dev: &D,
) -> Tensor<(Const<{ Config::SEQ_LEN }>,), usize, D> {
    let mut data: [usize; Config::SEQ_LEN] = [0; Config::SEQ_LEN];

    for i in 0..Config::SEQ_LEN {
        data[i] = tensors[i].as_vec()[0];
    }

    dev.tensor(data)
}

pub(crate) fn stack_usize_batched<
    E: Dtype,
    D: Device<E> + CopySlice<usize>,
    Config: DTModelConfig,
    const B: usize,
>(
    tensors: [Tensor<(Const<{ Config::SEQ_LEN }>,), usize, D>; B],
    dev: &D,
) -> Tensor<(Const<B>, Const<{ Config::SEQ_LEN }>), usize, D>{
    let mut data: [[usize; Config::SEQ_LEN]; B] = [[0; Config::SEQ_LEN]; B];

    for i in 0..B {
        tensors[i].copy_into(&mut data[i]);
    }

    dev.tensor(data)
    //dev.tensor(tensors)
}

#[allow(dead_code)]
pub(crate) fn print_input<
    const S: usize,
    const A: usize,
    E: Dtype,
    D: Device<E>,
    Config: DTModelConfig,
>(
    inp: &Input<S, A, E, D, Config>,
) where
    [(); Config::SEQ_LEN]: Sized,
{
    println!("States:");
    print_seq(&inp.0);
    println!("Actions:");
    print_seq(&inp.1);
    println!("Rewards:");
    print_seq(&inp.2);
    println!("Timesteps:");
    println!("{:?}", &inp.3.as_vec());
}

#[allow(dead_code)]
fn print_seq<E: Dtype, D: Device<E>, Config: DTModelConfig, const M: usize>(
    seq: &Tensor<(Const<{ Config::SEQ_LEN }>, Const<M>), E, D>,
) {
    let dev: D = Default::default();
    for i in 0..Config::SEQ_LEN {
        println!("{i}: {:?}", seq.clone().select(dev.tensor(i)).as_vec());
    }
}

#[allow(dead_code)]
pub(crate) fn cumulative_rewards<
    E: Dtype + From<f32> + num_traits::Float + rand_distr::uniform::SampleUniform,
    D: Device<E>,
    Config: DTModelConfig + 'static,
    S: DTState<E, D, Config>,
>(
    states: Vec<S>,
    actions: Vec<S::Action>,
) -> f32 {
    assert_eq!(states.len(), actions.len());

    let mut reward = 0.0;

    for i in 0..states.len() {
        reward += states[i].get_reward(actions[i].clone())
    }

    reward
}
