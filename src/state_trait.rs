use dfdx::prelude::*;

use crate::{
    dt_model::BatchedInput, trait_helpers::get_batch_from_fn, DTModel, DTModelConfig,
    DTModelWrapper,
};

/// DTState is the basic trait to implement for the decision transformer.
/// This state can represent a game state, or the like. Implementing all the methods goves access
/// to the primary benefit of DTState, which is the creation of a transformer model via '''build_model'''.
///
/// See [the snake_dt repo](https://github.com/JYudelson1/snake_dt/blob/main/src/dt_trait.rs) for examples.
pub trait DTState<
    E: Dtype + From<f32> + num_traits::Float + rand_distr::uniform::SampleUniform,
    D: Device<E>,
    Config: DTModelConfig + 'static,
>: Clone
{
    type Action: Clone;
    /// The total number of floats needed to represent the state
    const STATE_SIZE: usize;
    /// The total number of possible actions
    const ACTION_SIZE: usize;

    /// Required method
    ///
    /// Generate a new starting state. If the state is random, this should use the provided RNG.
    /// If determenistic, simply ignore rng.
    fn new_random<R: rand::Rng + ?Sized>(rng: &mut R) -> Self;

    /// Required method
    ///
    /// Apply the given action to the current state, consuming the action in the process.
    fn apply_action(&mut self, action: Self::Action);

    /// Required method
    ///
    /// If the given action were to be applied to the current state, what would eb the reward delta?
    /// Note that this should only give the additional reward gained or lost in this action,
    /// rather than the total score/reward.
    fn get_reward(&self, action: Self::Action) -> f32;

    /// Required method
    ///
    /// Convert the current state into a tensor that encodes the state.
    fn to_tensor(&self) -> Tensor<(Const<{ Self::STATE_SIZE }>,), E, D>;

    /// Required method
    ///
    /// Since we assume there are a finite number of discrete actions, we can map every action to a number in 0..ACTION_SIZE. This function should do that, and output a number in [0, ACTION_SIZE)
    fn action_to_index(action: &Self::Action) -> usize;

    /// Required method
    ///
    /// Since we assume there are a finite number of discrete actions, we can map every number in [0, ACTION_SIZE) to one action. This function should do that.
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

/// GetOfflineData gathers synthetic data to train the decision transformer model on.
pub trait GetOfflineData<
    E: Dtype + From<f32> + num_traits::Float + rand_distr::uniform::SampleUniform,
    D: Device<E> + ZerosTensor<usize> + CopySlice<usize>,
    Config: DTModelConfig + 'static,
>: DTState<E, D, Config>
{
    /// Required method
    ///
    /// Given an rng, play a full game. Note that, due to how a decision transformer works,
    /// the synthetic games don't need to be played very well, although they should ideally
    /// have high state coverage.
    ///
    /// Note that this should return the same number of states and actions, because the terminal state is not returned. See [the snake_dt repo](https://github.com/JYudelson1/snake_dt/blob/main/src/data.rs) for examples.
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

/// HumanEvaluable is for actually running a trained decision transformer model. Most DTStates should probably implement this.
pub trait HumanEvaluatable<
    E: Dtype + From<f32> + num_traits::Float + rand_distr::uniform::SampleUniform,
    D: Device<E>,
    Config: DTModelConfig + 'static,
>: DTState<E, D, Config>
{
    /// Required method
    ///
    /// Print the state.
    fn print(&self);
    
    /// Required method
    ///
    /// Print the given action.
    fn print_action(action: &Self::Action);
    
    /// Required method
    ///
    /// 
    fn is_still_playing(&self) -> bool;
}
