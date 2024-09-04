#![allow(type_alias_bounds)]

use dfdx::prelude::*;
use num_traits::Float;

use crate::{transformer::CustomTransformerDecoder, DTModelConfig};

type TimeEmbed<Config: DTModelConfig, E, D> =
    dfdx::nn::modules::Embedding<{ Config::MAX_EPISODES_IN_GAME }, { Config::HIDDEN_SIZE }, E, D>;
type StateHead<Config: DTModelConfig, const S: usize, E, D> =
    dfdx::nn::modules::Linear<S, { Config::HIDDEN_SIZE }, E, D>;
type ActionHead<Config: DTModelConfig, const A: usize, E, D> =
    dfdx::nn::modules::Linear<A, { Config::HIDDEN_SIZE }, E, D>;
type ReturnHead<Config: DTModelConfig, E, D> =
    dfdx::nn::modules::Linear<1, { Config::HIDDEN_SIZE }, E, D>;
type LN<Config: DTModelConfig> = dfdx::nn::builders::LayerNorm1D<{Config::HIDDEN_SIZE}>;

//type StatePredictor<const S: usize> = dfdx::nn::modules::Linear<{Config::HIDDEN_SIZE}, S, f32, STORAGE>;
type ActionPredictor<Config: DTModelConfig, const A: usize, E, D> =
    dfdx::nn::modules::Linear<{ Config::HIDDEN_SIZE }, A, E, D>;
//type ReturnPredictor = dfdx::nn::modules::Linear<{Config::HIDDEN_SIZE}, 1, f32, STORAGE>;

type DTTransformerInner<Config: DTModelConfig, E, D> = CustomTransformerDecoder<{Config::HIDDEN_SIZE}, {Config::MLP_INNER}, {Config::NUM_ATTENTION_HEADS}, {Config::NUM_LAYERS}, {3 * Config::SEQ_LEN}, E, D>;

pub type States<const EPISODES_IN_SEQ: usize, const S: usize, E, D, T = NoneTape> =
    Tensor<(Const<EPISODES_IN_SEQ>, Const<S>), E, D, T>;
pub type Actions<const EPISODES_IN_SEQ: usize, const A: usize, E, D, T = NoneTape> =
    Tensor<(Const<EPISODES_IN_SEQ>, Const<A>), E, D, T>;
pub type RewardsToGo<const EPISODES_IN_SEQ: usize, E, D, T = NoneTape> =
    Tensor<(Const<EPISODES_IN_SEQ>, Const<1>), E, D, T>;
pub type TimeSteps<const EPISODES_IN_SEQ: usize, D> = Tensor<(Const<EPISODES_IN_SEQ>,), usize, D>;
pub type Input<const S: usize, const A: usize, E, D, Config: DTModelConfig, T = NoneTape> = (
    States<{Config::SEQ_LEN}, S, E, D, T>,
    Actions<{Config::SEQ_LEN}, A, E, D, T>,
    RewardsToGo<{Config::SEQ_LEN}, E, D, T>,
    TimeSteps<{Config::SEQ_LEN}, D>,
);

pub type BatchedStates<
    const EPISODES_IN_SEQ: usize,
    const B: usize,
    const S: usize,
    E,
    D,
    T = NoneTape,
> = Tensor<(Const<B>, Const<EPISODES_IN_SEQ>, Const<S>), E, D, T>;
pub type BatchedActions<
    const EPISODES_IN_SEQ: usize,
    const B: usize,
    const A: usize,
    E,
    D,
    T = NoneTape,
> = Tensor<(Const<B>, Const<EPISODES_IN_SEQ>, Const<A>), E, D, T>;
pub type BatchedRewardsToGo<const EPISODES_IN_SEQ: usize, const B: usize, E, D, T = NoneTape> =
    Tensor<(Const<B>, Const<EPISODES_IN_SEQ>, Const<1>), E, D, T>;
pub type BatchedTimeSteps<const EPISODES_IN_SEQ: usize, const B: usize, D> =
    Tensor<(Const<B>, Const<EPISODES_IN_SEQ>), usize, D>;
pub type BatchedInput<
    const B: usize,
    const S: usize,
    const A: usize,
    E,
    D,
    Config: DTModelConfig,
    T = NoneTape,
> = (
    BatchedStates<{Config::SEQ_LEN}, B, S, E, D, T>,
    BatchedActions<{Config::SEQ_LEN}, B, A, E, D>,
    BatchedRewardsToGo<{Config::SEQ_LEN}, B, E, D>,
    BatchedTimeSteps<{Config::SEQ_LEN}, B, D>,
);

pub struct DTModel<
    Config: DTModelConfig,
    const STATE: usize,
    const ACTION: usize,
    E: Dtype,
    D: Device<E>,
> where
    [(); 3 * Config::SEQ_LEN]: Sized,
    [(); Config::MLP_INNER]: Sized,
    [(); Config::HIDDEN_SIZE]: Sized,
    [(); Config::MAX_EPISODES_IN_GAME]: Sized,
    [(); Config::NUM_ATTENTION_HEADS]: Sized,
    [(); Config::NUM_LAYERS]: Sized,
{
    pub transformer: DTTransformerInner<Config, E, D>,
    pub state_head: StateHead<Config, STATE, E, D>,
    pub action_head: ActionHead<Config, ACTION, E, D>,
    pub return_head: ReturnHead<Config, E, D>,
    pub predict_action: ActionPredictor<Config, ACTION, E, D>,
    pub time_embeddings: TimeEmbed<Config, E, D>,
}
// tensor Collections
impl<
        const S: usize,
        const A: usize,
        E: Dtype + num_traits::Float + rand_distr::uniform::SampleUniform,
        D: Device<E>,
        Config: DTModelConfig
    > TensorCollection<E, D> for DTModel<Config, S, A, E, D>
    where
    [(); 3 * Config::SEQ_LEN]: Sized,
    [(); Config::SEQ_LEN]: Sized,
    [(); Config::MLP_INNER]: Sized,
    [(); Config::MAX_EPISODES_IN_GAME]: Sized,
    [(); Config::NUM_ATTENTION_HEADS]: Sized,
    [(); Config::HIDDEN_SIZE]: Sized,
    [(); Config::NUM_LAYERS]: Sized,
{
    type To<E2: Dtype, D2: Device<E2>> = DTModel<Config, S, A, E2, D2>;

    fn iter_tensors<V: ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(
            (
                // Define name of each field and how to access it, using ModuleField for Modules,
                // and TensorField for Tensors.
                Self::module("state_head", |s| &s.state_head, |s| &mut s.state_head),
                Self::module("action_head", |s| &s.action_head, |s| &mut s.action_head),
                Self::module("return_head", |s| &s.return_head, |s| &mut s.return_head),
                Self::module(
                    "predict_action",
                    |s| &s.predict_action,
                    |s| &mut s.predict_action,
                ),
                Self::module("transformer", |s| &s.transformer, |s| &mut s.transformer),
                Self::module(
                    "time_embeddings",
                    |s| &s.time_embeddings,
                    |s| &mut s.time_embeddings,
                ),
            ),
            // Define how to construct the collection given its fields in the order they are given
            // above. This conversion is done using the ModuleFields trait.
            |(
                state_head,
                action_head,
                return_head,
                predict_action,
                transformer,
                time_embeddings,
            )| DTModel {
                time_embeddings,
                transformer,
                state_head,
                action_head,
                return_head,
                predict_action,
            },
        )
    }
}

// Module for one input
impl<
        const S: usize,
        const A: usize,
        E: Dtype + Float,
        D: Device<E> + DeviceBuildExt,
        Config: DTModelConfig
    > Module<Input<S, A, E, D, Config>> for DTModel<Config, S, A, E, D>
where
    [(); 3 * Config::SEQ_LEN]: Sized,
    [(); Config::SEQ_LEN]: Sized,
    [(); Config::MLP_INNER]: Sized,
    [(); Config::MAX_EPISODES_IN_GAME]: Sized,
    [(); Config::NUM_ATTENTION_HEADS]: Sized,
    [(); Config::HIDDEN_SIZE]: Sized,
    [(); Config::NUM_LAYERS]: Sized,
{
    type Output = Tensor<(Const<{Config::SEQ_LEN}>, Const<A>), E, D>;

    type Error = ();

    fn try_forward(
        &self,
        input: Input<S, A, E, D, Config>,
    ) -> Result<Self::Output, Self::Error> {
        let (states, actions, rewards, timesteps) = input;
        let dev: D = Default::default();

        let states = self.state_head.forward(states);
        let actions = self.action_head.forward(actions);
        let rewards = self.return_head.forward(rewards);

        let times = self.time_embeddings.forward(timesteps);

        let rewards = rewards + times.clone();
        let actions = actions + times.clone();
        let states = states + times;

        let stacked = [rewards, states, actions]
            .stack()
            .permute::<_, Axes3<1, 0, 2>>()
            .reshape::<(Const<{ 3 * Config::SEQ_LEN }>, Const<{Config::HIDDEN_SIZE}>)>();

        let input: Tensor<(Const<{ 3 * Config::SEQ_LEN }>, Const<{Config::HIDDEN_SIZE}>), E, D> =
            dev.build_module::<LN<Config>, E>().forward(stacked);

        // let out = self.transformer.forward(input);

        let out = input
            .reshape::<(Const<{Config::SEQ_LEN}>, Const<3>, Const<{Config::HIDDEN_SIZE}>)>()
            .permute::<_, Axes3<1, 0, 2>>();

        let actions = self.predict_action.forward(out.select(dev.tensor(2)));

        Ok(actions)
    }
}

// Batched Module
impl<
        const S: usize,
        const A: usize,
        const B: usize,
        E: Dtype + Float,
        D: Device<E> + DeviceBuildExt,
        Config: DTModelConfig
    > Module<BatchedInput<B, S, A, E, D, Config, NoneTape>>
    for DTModel<Config, S, A, E, D>
where
    [(); 3 * Config::SEQ_LEN]: Sized,
    [(); Config::SEQ_LEN]: Sized,
    [(); Config::MLP_INNER]: Sized,
    [(); Config::MAX_EPISODES_IN_GAME]: Sized,
    [(); Config::NUM_ATTENTION_HEADS]: Sized,
    [(); Config::HIDDEN_SIZE]: Sized,
    [(); Config::NUM_LAYERS]: Sized,
{
    type Output = Tensor<(Const<B>, Const<{Config::SEQ_LEN}>, Const<A>), E, D, NoneTape>;

    type Error = ();

    fn try_forward(
        &self,
        input: BatchedInput<B, S, A, E, D, Config, NoneTape>,
    ) -> Result<Self::Output, Self::Error> {
        let (states, actions, rewards, timesteps) = input;
        let dev: D = Default::default();

        let states = self.state_head.forward(states);
        let actions = self.action_head.forward(actions);
        let rewards = self.return_head.forward(rewards);

        let times = self.time_embeddings.forward(timesteps);

        let rewards = rewards + times.clone();
        
        let actions = actions + times.clone();
        let states = states + times;

        let stacked = [rewards, states, actions]
            .stack()
            .permute::<_, Axes4<0, 2, 1, 3>>()
            .reshape::<(Const<B>, Const<{ 3 * Config::SEQ_LEN }>, Const<{Config::HIDDEN_SIZE}>)>();

        let input: Tensor<(Const<B>, Const<{ 3 * Config::SEQ_LEN }>, Const<{Config::HIDDEN_SIZE}>), E, D> =
            dev.build_module::<LN<Config>, E>().forward(stacked);

        // let out = self.transformer.forward(input);

        let out = input
            .reshape::<(
                Const<B>,
                Const<{Config::SEQ_LEN}>,
                Const<3>,
                Const<{Config::HIDDEN_SIZE}>,
            )>()
            .permute::<_, Axes4<0, 2, 1, 3>>();

        let actions = self.predict_action.forward(out.select(dev.tensor([2; B]))); // TOOD: Check correctness

        Ok(actions)
    }
}


// Batched Module Mut
impl<
        const S: usize,
        const A: usize,
        const B: usize,
        T: Tape<E, D>,
        E: Dtype + Float,
        D: Device<E> + DeviceBuildExt,
        Config: DTModelConfig
    > ModuleMut<BatchedInput<B, S, A, E, D, Config, T>>
    for DTModel<Config, S, A, E, D>
where
    [(); 3 * Config::SEQ_LEN]: Sized,
    [(); Config::SEQ_LEN]: Sized,
    [(); Config::MLP_INNER]: Sized,
    [(); Config::MAX_EPISODES_IN_GAME]: Sized,
    [(); Config::NUM_ATTENTION_HEADS]: Sized,
    [(); Config::HIDDEN_SIZE]: Sized,
    [(); Config::NUM_LAYERS]: Sized,
{
    type Output = Tensor<(Const<B>, Const<{Config::SEQ_LEN}>, Const<A>), E, D, T>;

    type Error = ();

    fn try_forward_mut(
        &mut self,
        input: BatchedInput<B, S, A, E, D, Config, T>,
    ) -> Result<Self::Output, Self::Error> {
        let (states, actions, rewards, timesteps) = input;
        let dev: D = Default::default();

        let states = self.state_head.forward_mut(states);
        let (states, tape) = states.split_tape();
        let actions = self.action_head.forward_mut(actions.put_tape(tape));
        let (actions, tape) = actions.split_tape();
        let rewards = self.return_head.forward_mut(rewards.put_tape(tape));
        let (rewards, tape) = rewards.split_tape();

        let times = self.time_embeddings.forward(timesteps);

        let rewards = rewards + times.clone();
        let actions = actions + times.clone();
        let states = states + times;

        let stacked = [rewards, states, actions]
            .stack()
            .permute::<_, Axes4<0, 2, 1, 3>>()
            .reshape::<(Const<B>, Const<{ 3 * Config::SEQ_LEN }>, Const<{Config::HIDDEN_SIZE}>)>();
        let stacked = stacked.put_tape(tape);

        let input: Tensor<(Const<B>, Const<{ 3 * Config::SEQ_LEN }>, Const<{Config::HIDDEN_SIZE}>), E, D, T> =
            dev.build_module::<LN<Config>, E>().forward(stacked);

        // let out = self.transformer.forward(input);

        let out = input
            .reshape::<(
                Const<B>,
                Const<{Config::SEQ_LEN}>,
                Const<3>,
                Const<{Config::HIDDEN_SIZE}>,
            )>()
            .permute::<_, Axes4<0, 2, 1, 3>>();

        let actions = self
            .predict_action
            .forward_mut(out.select(dev.tensor([2; B]))); // TOOD: Check correctness

        Ok(actions)
    }
}
