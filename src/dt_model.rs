use dfdx::prelude::*;

use crate::transformer::CustomTransformerDecoder;

const HIDDEN: usize = 256;

type TimeEmbed<const MAX_EPISODES_IN_GAME: usize, E, D> =
    dfdx::nn::modules::Embedding<MAX_EPISODES_IN_GAME, HIDDEN, E, D>;
type StateHead<const S: usize, E, D> = dfdx::nn::modules::Linear<S, HIDDEN, E, D>;
type ActionHead<const A: usize, E, D> = dfdx::nn::modules::Linear<A, HIDDEN, E, D>;
type ReturnHead<E, D> = dfdx::nn::modules::Linear<1, HIDDEN, E, D>;
type LN = dfdx::nn::builders::LayerNorm1D<HIDDEN>;

//type StatePredictor<const S: usize> = dfdx::nn::modules::Linear<HIDDEN, S, f32, STORAGE>;
type ActionPredictor<const A: usize, E, D> = dfdx::nn::modules::Linear<HIDDEN, A, E, D>;
//type ReturnPredictor = dfdx::nn::modules::Linear<HIDDEN, 1, f32, STORAGE>;

type DTTransformerInner<const EPISODES_IN_SEQ: usize, E, D> = CustomTransformerDecoder<HIDDEN, {HIDDEN*4}, 12, 3, {3 * EPISODES_IN_SEQ}, E, D>;

pub type States<const EPISODES_IN_SEQ: usize, const S: usize, E, D, T = NoneTape> =
    Tensor<(Const<EPISODES_IN_SEQ>, Const<S>), E, D, T>;
pub type Actions<const EPISODES_IN_SEQ: usize, const A: usize, E, D, T = NoneTape> =
    Tensor<(Const<EPISODES_IN_SEQ>, Const<A>), E, D, T>;
pub type RewardsToGo<const EPISODES_IN_SEQ: usize, E, D, T = NoneTape> =
    Tensor<(Const<EPISODES_IN_SEQ>, Const<1>), E, D, T>;
pub type TimeSteps<const EPISODES_IN_SEQ: usize, D> = Tensor<(Const<EPISODES_IN_SEQ>,), usize, D>;
pub type Input<const EPISODES_IN_SEQ: usize, const S: usize, const A: usize, E, D, T = NoneTape> = (
    States<EPISODES_IN_SEQ, S, E, D, T>,
    Actions<EPISODES_IN_SEQ, A, E, D, T>,
    RewardsToGo<EPISODES_IN_SEQ, E, D, T>,
    TimeSteps<EPISODES_IN_SEQ, D>,
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
    const EPISODES_IN_SEQ: usize,
    const B: usize,
    const S: usize,
    const A: usize,
    E,
    D,
    T = NoneTape,
> = (
    BatchedStates<EPISODES_IN_SEQ, B, S, E, D, T>,
    BatchedActions<EPISODES_IN_SEQ, B, A, E, D, T>,
    BatchedRewardsToGo<EPISODES_IN_SEQ, B, E, D, T>,
    BatchedTimeSteps<EPISODES_IN_SEQ, B, D>,
);

pub struct DTModel<
    const MAX_EPISODES_IN_GAME: usize,
    const EPISODES_IN_SEQ: usize,
    const STATE: usize,
    const ACTION: usize,
    E: Dtype,
    D: Device<E>,
> where
    [(); 3 * EPISODES_IN_SEQ]: Sized,
{
    pub transformer: DTTransformerInner<EPISODES_IN_SEQ, E, D>,
    pub state_head: StateHead<STATE, E, D>,
    pub action_head: ActionHead<ACTION, E, D>,
    pub return_head: ReturnHead<E, D>,
    pub predict_action: ActionPredictor<ACTION, E, D>,
    pub time_embeddings: TimeEmbed<MAX_EPISODES_IN_GAME, E, D>,
}
// tensor Collections
impl<
        const MAX_EPISODES_IN_GAME: usize,
        const EPISODES_IN_SEQ: usize,
        const S: usize,
        const A: usize,
        E: Dtype + num_traits::Float + rand_distr::uniform::SampleUniform,
        D: Device<E>,
    > TensorCollection<E, D> for DTModel<MAX_EPISODES_IN_GAME, EPISODES_IN_SEQ, S, A, E, D>
    where [(); 3 * EPISODES_IN_SEQ]: Sized
{
    type To<E2: Dtype, D2: Device<E2>> = DTModel<MAX_EPISODES_IN_GAME, EPISODES_IN_SEQ, S, A, E2, D2>;

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
        const EPISODES_IN_SEQ: usize,
        const MAX_EPISODES_IN_GAME: usize,
        const S: usize,
        const A: usize,
        T: Tape<E, D>,
        E: Dtype,
        D: Device<E> + DeviceBuildExt,
    > Module<Input<EPISODES_IN_SEQ, S, A, E, D, T>> for DTModel<MAX_EPISODES_IN_GAME, EPISODES_IN_SEQ, S, A, E, D>
where
    [(); 3 * EPISODES_IN_SEQ]: Sized,
{
    type Output = Tensor<(Const<EPISODES_IN_SEQ>, Const<A>), E, D, T>;

    type Error = ();

    fn try_forward(
        &self,
        input: Input<EPISODES_IN_SEQ, S, A, E, D, T>,
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
            .reshape::<(Const<{ 3 * EPISODES_IN_SEQ }>, Const<HIDDEN>)>();

        let input: Tensor<(Const<{ 3 * EPISODES_IN_SEQ }>, Const<HIDDEN>), E, D, T> =
            dev.build_module::<LN, E>().forward(stacked);

        let out = self.transformer.forward(input);

        let out = out
            .reshape::<(Const<EPISODES_IN_SEQ>, Const<3>, Const<HIDDEN>)>()
            .permute::<_, Axes3<1, 0, 2>>();

        let actions = self.predict_action.forward(out.select(dev.tensor(2)));

        Ok(actions)
    }
}

// Batched Module
impl<
        const EPISODES_IN_SEQ: usize,
        const MAX_EPISODES_IN_GAME: usize,
        const S: usize,
        const A: usize,
        const B: usize,
        T: Tape<E, D>,
        E: Dtype,
        D: Device<E> + DeviceBuildExt,
    > Module<BatchedInput<EPISODES_IN_SEQ, B, S, A, E, D, T>>
    for DTModel<MAX_EPISODES_IN_GAME, EPISODES_IN_SEQ, S, A, E, D>
where
    [(); 3 * EPISODES_IN_SEQ]: Sized,
{
    type Output = Tensor<(Const<B>, Const<EPISODES_IN_SEQ>, Const<A>), E, D, T>;

    type Error = ();

    fn try_forward(
        &self,
        input: BatchedInput<EPISODES_IN_SEQ, B, S, A, E, D, T>,
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
            .reshape::<(Const<B>, Const<{ 3 * EPISODES_IN_SEQ }>, Const<HIDDEN>)>();

        let input: Tensor<(Const<B>, Const<{ 3 * EPISODES_IN_SEQ }>, Const<HIDDEN>), E, D, T> =
            dev.build_module::<LN, E>().forward(stacked);

        let out = self.transformer.forward(input);

        let out = out
            .reshape::<(
                Const<B>,
                Const<EPISODES_IN_SEQ>,
                Const<3>,
                Const<HIDDEN>,
            )>()
            .permute::<_, Axes4<0, 2, 1, 3>>();

        let actions = self.predict_action.forward(out.select(dev.tensor([2; B]))); // TOOD: Check correctness

        Ok(actions)
    }
}


// Batched Module Mut
impl<
        const EPISODES_IN_SEQ: usize,
        const MAX_EPISODES_IN_GAME: usize,
        const S: usize,
        const A: usize,
        const B: usize,
        T: Tape<E, D>,
        E: Dtype,
        D: Device<E> + DeviceBuildExt,
    > ModuleMut<BatchedInput<EPISODES_IN_SEQ, B, S, A, E, D, T>>
    for DTModel<MAX_EPISODES_IN_GAME, EPISODES_IN_SEQ, S, A, E, D>
where
    [(); 3 * EPISODES_IN_SEQ]: Sized,
{
    type Output = Tensor<(Const<B>, Const<EPISODES_IN_SEQ>, Const<A>), E, D, T>;

    type Error = ();

    fn try_forward_mut(
        &mut self,
        input: BatchedInput<EPISODES_IN_SEQ, B, S, A, E, D, T>,
    ) -> Result<Self::Output, Self::Error> {
        let (states, actions, rewards, timesteps) = input;
        let dev: D = Default::default();

        let states = self.state_head.forward_mut(states);
        let actions = self.action_head.forward_mut(actions);
        let rewards = self.return_head.forward_mut(rewards);

        let times = self.time_embeddings.forward(timesteps);

        let rewards = rewards + times.clone();
        let actions = actions + times.clone();
        let states = states + times;

        let stacked = [rewards, states, actions]
            .stack()
            .permute::<_, Axes4<0, 2, 1, 3>>()
            .reshape::<(Const<B>, Const<{ 3 * EPISODES_IN_SEQ }>, Const<HIDDEN>)>();

        let input: Tensor<(Const<B>, Const<{ 3 * EPISODES_IN_SEQ }>, Const<HIDDEN>), E, D, T> =
            dev.build_module::<LN, E>().forward(stacked);

        let out = self.transformer.forward_mut(input);

        let out = out
            .reshape::<(
                Const<B>,
                Const<EPISODES_IN_SEQ>,
                Const<3>,
                Const<HIDDEN>,
            )>()
            .permute::<_, Axes4<0, 2, 1, 3>>();

        let actions = self
            .predict_action
            .forward_mut(out.select(dev.tensor([2; B]))); // TOOD: Check correctness

        Ok(actions)
    }
}
