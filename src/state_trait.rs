use dfdx::prelude::*;

use crate::{
    dt_model::BatchedInput, trait_helpers::get_batch_from_fn, DTModel, DTModelConfig,
    DTModelWrapper,
};

pub trait DTState<
    E: Dtype + From<f32> + num_traits::Float + rand_distr::uniform::SampleUniform,
    D: Device<E>,
    Config: DTModelConfig + 'static,
>: Clone
{
    type Action: Clone;
    const STATE_SIZE: usize;
    const ACTION_SIZE: usize;

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
    fn build_model() -> DTModelWrapper<E, D, Config, Self>
    where
        [(); Config::MAX_EPISODES_IN_GAME]: Sized,
        [(); Self::ACTION_SIZE]: Sized,
        [(); Self::STATE_SIZE]: Sized,
        [(); 3 * Config::SEQ_LEN]: Sized,
        [(); Config::HIDDEN_SIZE]: Sized,
        [(); Config::MLP_INNER]: Sized,
        [(); Config::MAX_EPISODES_IN_GAME]: Sized,
        [(); Config::NUM_ATTENTION_HEADS]: Sized,
        [(); Config::NUM_LAYERS]: Sized,
        [(); 3 * Config::HIDDEN_SIZE * Config::SEQ_LEN]: Sized,
        [(); Config::HIDDEN_SIZE * Config::SEQ_LEN]: Sized,
    {
        let dev: D = Default::default();
        let mut model =
            DTModel::<Config, { Self::STATE_SIZE }, { Self::ACTION_SIZE }, E, D>::build(&dev);
        model.reset_params();
        DTModelWrapper(model)
    }
}

pub trait GetOfflineData<
    E: Dtype + From<f32> + num_traits::Float + rand_distr::uniform::SampleUniform,
    D: Device<E> + ZerosTensor<usize> + CopySlice<usize>,
    Config: DTModelConfig + 'static,
>: DTState<E, D, Config>
{
    /// Required method
    fn play_one_game<R: rand::Rng + ?Sized>(rng: &mut R) -> (Vec<Self>, Vec<Self::Action>);

    /// Provided method
    fn get_batch<const B: usize, R: rand::Rng + ?Sized>(
        rng: &mut R,
        cap_from_game: Option<usize>
    ) -> (
        BatchedInput<B, { Self::STATE_SIZE }, { Self::ACTION_SIZE }, E, D, Config, NoneTape>,
        [Self::Action; B],
        //Tensor<(Const<B>, Const<{Config::SEQ_LEN}>, Const<{Config::HIDDEN_SIZE}>), E, D>
    )
    where
        [(); Config::MAX_EPISODES_IN_GAME]: Sized,
        [(); Config::SEQ_LEN]: Sized,
        [(); 3 * Config::SEQ_LEN]: Sized,
        [(); Config::HIDDEN_SIZE]: Sized,
        [(); Config::MLP_INNER]: Sized,
        [(); Config::NUM_ATTENTION_HEADS]: Sized,
    {
        get_batch_from_fn(rng, Self::play_one_game, cap_from_game)
    }
}

pub trait HumanEvaluatable<
    E: Dtype + From<f32> + num_traits::Float + rand_distr::uniform::SampleUniform,
    D: Device<E>,
    Config: DTModelConfig + 'static,
>: DTState<E, D, Config>
{
    fn print(&self);
    fn print_action(action: &Self::Action);
    fn is_still_playing(&self) -> bool;
}
