use dfdx::prelude::*;
use stack::StackKernel;

use crate::{dt_model::BatchedInput, trait_helpers::get_batch_from_fn, DTModel, DTModelWrapper};

pub trait DTState<
    E: Dtype + From<f32> + num_traits::Float + rand_distr::uniform::SampleUniform,
    D: Device<E>,
>: Clone
{
    type Action: Clone;
    const STATE_SIZE: usize;
    const ACTION_SIZE: usize;
    const EPISODES_IN_SEQ: usize;
    const MAX_EPISODES_IN_GAME: usize;

    /// Required method
    fn new_random<R: rand::Rng + ?Sized>(rng: &mut R) -> Self;

    /// Required method
    fn apply_action(&mut self, action: Self::Action);

    /// Required method
    fn get_reward(&self, action: Self::Action) -> f32;

    /// Required method
    fn to_tensor(&self) -> Tensor<(Const<{ Self::STATE_SIZE }>,), E, D>;

    /// Required method
    fn action_to_index(action: &Self::Action) -> usize;

    /// Required method
    fn index_to_action(action: usize) -> Self::Action;

    /// Provided method
    fn action_to_tensor(action: &Self::Action) -> Tensor<(Const<{ Self::ACTION_SIZE }>,), E, D> {
        let dev: Cpu = Default::default();
        let mut t = dev.zeros();
        t[[Self::action_to_index(action)]] = 1.0_f32.into();
        t.to_device(&D::default())
    }

    /// Provided method
    fn build_model() -> DTModelWrapper<E, D, Self>
    where [(); Self::MAX_EPISODES_IN_GAME]: Sized,
    [(); Self::ACTION_SIZE]: Sized,
    [(); Self::STATE_SIZE]: Sized,
    {
        let dev: D = Default::default();
        let mut model = DTModel::<
            { Self::MAX_EPISODES_IN_GAME },
            { Self::STATE_SIZE },
            { Self::ACTION_SIZE },
            E,
            D,
        >::build(&dev);
        model.reset_params();
        DTModelWrapper(model)
    }
}

pub trait GetOfflineData<
    E: Dtype + From<f32> + num_traits::Float + rand_distr::uniform::SampleUniform,
    D: Device<E> + ZerosTensor<usize> + StackKernel<usize>,
>: DTState<E, D>
{
    /// Required method
    fn play_one_game<R: rand::Rng + ?Sized>(rng: &mut R) -> (Vec<Self>, Vec<Self::Action>);

    /// Provided method
    fn get_batch<const B: usize, R: rand::Rng + ?Sized>(
        rng: &mut R,
    ) -> (BatchedInput<
        { Self::EPISODES_IN_SEQ },
        B,
        { Self::STATE_SIZE },
        { Self::ACTION_SIZE },
        E,
        D,
        NoneTape,
    >, [Self::Action; B]){
        get_batch_from_fn(rng, Self::play_one_game)
    }
}

pub trait HumanEvaluatable<
    E: Dtype + From<f32> + num_traits::Float + rand_distr::uniform::SampleUniform,
    D: Device<E>,
>: DTState<E, D>
{
    fn print(&self);
    fn print_action(action: &Self::Action);
    fn is_still_playing(&self) -> bool;
}
