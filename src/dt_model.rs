//use dfdx::nn::builders::{AddInto, Embedding, LayerNorm1D, Linear, TransformerDecoder};
use dfdx::prelude::*;

const HIDDEN: usize = 256;

type TimeEmbed<const MAX_EPISODES_IN_SEQ: usize, E, D> =
    dfdx::nn::modules::Embedding<MAX_EPISODES_IN_SEQ, HIDDEN, E, D>;
type StateHead<const S: usize, E, D> = dfdx::nn::modules::Linear<S, HIDDEN, E, D>;
type ActionHead<const A: usize, E, D> = dfdx::nn::modules::Linear<A, HIDDEN, E, D>;
type ReturnHead<E, D> = dfdx::nn::modules::Linear<1, HIDDEN, E, D>;
type LN = dfdx::nn::builders::LayerNorm1D<HIDDEN>;

//type StatePredictor<const S: usize> = dfdx::nn::modules::Linear<HIDDEN, S, f32, STORAGE>;
type ActionPredictor<const A: usize, E, D> = dfdx::nn::modules::Linear<HIDDEN, A, E, D>;
//type ReturnPredictor = dfdx::nn::modules::Linear<HIDDEN, 1, f32, STORAGE>;

type DTTransformerInner<E, D> =
    dfdx::nn::modules::TransformerDecoder<HIDDEN, 12, { 4 * HIDDEN }, 12, E, D>;

pub type States<const MAX_EPISODES_IN_SEQ: usize, const S: usize, E, D, T = NoneTape> =
    Tensor<(Const<MAX_EPISODES_IN_SEQ>, Const<S>), E, D, T>;
pub type Actions<const MAX_EPISODES_IN_SEQ: usize, const A: usize, E, D, T = NoneTape> =
    Tensor<(Const<MAX_EPISODES_IN_SEQ>, Const<A>), E, D, T>;
pub type RewardsToGo<const MAX_EPISODES_IN_SEQ: usize, E, D, T = NoneTape> =
    Tensor<(Const<MAX_EPISODES_IN_SEQ>, Const<1>), E, D, T>;
pub type TimeSteps<const MAX_EPISODES_IN_SEQ: usize, D> =
    Tensor<(Const<MAX_EPISODES_IN_SEQ>,), usize, D>;
pub type Input<
    const MAX_EPISODES_IN_SEQ: usize,
    const S: usize,
    const A: usize,
    E,
    D,
    T = NoneTape,
> = (
    States<MAX_EPISODES_IN_SEQ, S, E, D, T>,
    Actions<MAX_EPISODES_IN_SEQ, A, E, D, T>,
    RewardsToGo<MAX_EPISODES_IN_SEQ, E, D, T>,
    TimeSteps<MAX_EPISODES_IN_SEQ, D>,
);

pub type BatchedStates<
    const MAX_EPISODES_IN_SEQ: usize,
    const B: usize,
    const S: usize,
    E,
    D,
    T = NoneTape,
> = Tensor<(Const<B>, Const<MAX_EPISODES_IN_SEQ>, Const<S>), E, D, T>;
pub type BatchedActions<
    const MAX_EPISODES_IN_SEQ: usize,
    const B: usize,
    const A: usize,
    E,
    D,
    T = NoneTape,
> = Tensor<(Const<B>, Const<MAX_EPISODES_IN_SEQ>, Const<A>), E, D, T>;
pub type BatchedRewardsToGo<const MAX_EPISODES_IN_SEQ: usize, const B: usize, E, D, T = NoneTape> =
    Tensor<(Const<B>, Const<MAX_EPISODES_IN_SEQ>, Const<1>), E, D, T>;
pub type BatchedTimeSteps<const MAX_EPISODES_IN_SEQ: usize, const B: usize, D> =
    Tensor<(Const<B>, Const<MAX_EPISODES_IN_SEQ>), usize, D>;
pub type BatchedInput<
    const MAX_EPISODES_IN_SEQ: usize,
    const B: usize,
    const S: usize,
    const A: usize,
    E,
    D,
    T = NoneTape,
> = (
    BatchedStates<MAX_EPISODES_IN_SEQ, B, S, E, D, T>,
    BatchedActions<MAX_EPISODES_IN_SEQ, B, A, E, D, T>,
    BatchedRewardsToGo<MAX_EPISODES_IN_SEQ, B, E, D, T>,
    BatchedTimeSteps<MAX_EPISODES_IN_SEQ, B, D>,
);

pub struct DTModel<
    const MAX_EPISODES_IN_SEQ: usize,
    const STATE: usize,
    const ACTION: usize,
    E: Dtype,
    D: Device<E>,
> {
    pub transformer: DTTransformerInner<E, D>,
    pub state_head: StateHead<STATE, E, D>,
    pub action_head: ActionHead<ACTION, E, D>,
    pub return_head: ReturnHead<E, D>,
    pub predict_action: ActionPredictor<ACTION, E, D>,
    pub time_embeddings: TimeEmbed<MAX_EPISODES_IN_SEQ, E, D>,
}
// tensor Collections
impl<
        const MAX_EPISODES_IN_SEQ: usize,
        const S: usize,
        const A: usize,
        E: Dtype + num_traits::Float + rand_distr::uniform::SampleUniform,
        D: Device<E>,
    > TensorCollection<E, D> for DTModel<MAX_EPISODES_IN_SEQ, S, A, E, D>
{
    type To<E2: Dtype, D2: Device<E2>> = DTModel<MAX_EPISODES_IN_SEQ, S, A, E2, D2>;

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

// ModuleMut for one input
impl<
        const MAX_EPISODES_IN_SEQ: usize,
        const S: usize,
        const A: usize,
        T: Tape<E, D>,
        E: Dtype,
        D: Device<E> + DeviceBuildExt,
    > ModuleMut<Input<MAX_EPISODES_IN_SEQ, S, A, E, D, T>>
    for DTModel<MAX_EPISODES_IN_SEQ, S, A, E, D>
where
    [(); 3 * MAX_EPISODES_IN_SEQ]: Sized,
{
    type Output = Tensor<(Const<MAX_EPISODES_IN_SEQ>, Const<A>), E, D, T>;

    type Error = ();

    fn try_forward_mut(
        &mut self,
        input: Input<MAX_EPISODES_IN_SEQ, S, A, E, D, T>,
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
            .permute::<_, Axes3<1, 0, 2>>()
            .reshape::<(Const<{ 3 * MAX_EPISODES_IN_SEQ }>, Const<HIDDEN>)>();

        let input: Tensor<(Const<{ 3 * MAX_EPISODES_IN_SEQ }>, Const<HIDDEN>), E, D, T> =
            dev.build_module::<LN, E>().forward(stacked);

        // let zeroes: Tensor<(Const<{ 3 * MAX_EPISODES_IN_SEQ }>, Const<256>), E, D, _> = dev.zeros_like(&input);
        // let t_out = self.transformer.forward_mut((zeroes, input));

        let out = input
            .reshape::<(Const<MAX_EPISODES_IN_SEQ>, Const<3>, Const<HIDDEN>)>()
            .permute::<_, Axes3<1, 0, 2>>();

        let actions = self.predict_action.forward_mut(out.select(dev.tensor(2)));

        Ok(actions)
    }
}
// Module for one input
impl<
        const MAX_EPISODES_IN_SEQ: usize,
        const S: usize,
        const A: usize,
        T: Tape<E, D>,
        E: Dtype,
        D: Device<E> + DeviceBuildExt,
    > Module<Input<MAX_EPISODES_IN_SEQ, S, A, E, D, T>> for DTModel<MAX_EPISODES_IN_SEQ, S, A, E, D>
where
    [(); 3 * MAX_EPISODES_IN_SEQ]: Sized,
{
    type Output = Tensor<(Const<MAX_EPISODES_IN_SEQ>, Const<A>), E, D, T>;

    type Error = ();

    fn try_forward(
        &self,
        input: Input<MAX_EPISODES_IN_SEQ, S, A, E, D, T>,
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
            .reshape::<(Const<{ 3 * MAX_EPISODES_IN_SEQ }>, Const<HIDDEN>)>();

        let input: Tensor<(Const<{ 3 * MAX_EPISODES_IN_SEQ }>, Const<HIDDEN>), E, D, T> =
            dev.build_module::<LN, E>().forward(stacked);

        // let zeroes: Tensor<(Const<{ 3 * MAX_EPISODES_IN_SEQ }>, Const<256>), E, D, _> = dev.zeros_like(&input);
        // let t_out = self.transformer.forward_mut((zeroes, input));

        let out = input
            .reshape::<(Const<MAX_EPISODES_IN_SEQ>, Const<3>, Const<HIDDEN>)>()
            .permute::<_, Axes3<1, 0, 2>>();

        let actions = self.predict_action.forward(out.select(dev.tensor(2)));

        Ok(actions)
    }
}

// Batched Module Mut
impl<
        const MAX_EPISODES_IN_SEQ: usize,
        const S: usize,
        const A: usize,
        const B: usize,
        T: Tape<E, D>,
        E: Dtype,
        D: Device<E> + DeviceBuildExt,
    > ModuleMut<BatchedInput<MAX_EPISODES_IN_SEQ, B, S, A, E, D, T>>
    for DTModel<MAX_EPISODES_IN_SEQ, S, A, E, D>
where
    [(); 3 * MAX_EPISODES_IN_SEQ]: Sized,
{
    type Output = Tensor<(Const<B>, Const<MAX_EPISODES_IN_SEQ>, Const<A>), E, D, T>;

    type Error = ();

    fn try_forward_mut(
        &mut self,
        input: BatchedInput<MAX_EPISODES_IN_SEQ, B, S, A, E, D, T>,
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
            .reshape::<(Const<B>, Const<{ 3 * MAX_EPISODES_IN_SEQ }>, Const<HIDDEN>)>();

        let input: Tensor<(Const<B>, Const<{ 3 * MAX_EPISODES_IN_SEQ }>, Const<HIDDEN>), E, D, T> =
            dev.build_module::<LN, E>().forward(stacked);

        // let zeroes: Tensor<(Const<{ 3 * MAX_EPISODES_IN_SEQ }>, Const<256>), E, D, _> = dev.zeros_like(&input);
        // let t_out = self.transformer.forward_mut((zeroes, input));

        let out = input
            .reshape::<(
                Const<B>,
                Const<MAX_EPISODES_IN_SEQ>,
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
