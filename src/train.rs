use dfdx::prelude::*;

use crate::{dt_model::BatchedInput, DTModel, DTModelConfig, DTModelWrapper, DTState};

pub fn loss<const B: usize, const A: usize, E: Dtype, D: Device<E>, T: Tape<E, D>>(
    action_pred: Tensor<(Const<B>, Const<A>), E, D, T>,
    action_actual: Tensor<(Const<B>, Const<A>), E, D, NoneTape>,
) -> Tensor<Rank0, E, D, T> {
    mse_loss(action_pred, action_actual)
}

impl<
        E: Dtype + From<f32> + num_traits::Float + rand_distr::uniform::SampleUniform,
        D: Device<E> + DeviceBuildExt + dfdx::tensor::ZerosTensor<usize>,
        Config: DTModelConfig + 'static,
        Game: DTState<E, D, Config>,
    > DTModelWrapper<E, D, Config, Game>
where
    [(); Config::MAX_EPISODES_IN_GAME]: Sized,
    [(); Config::SEQ_LEN]: Sized,
    [(); 3 * Config::SEQ_LEN]: Sized,
    [(); Config::HIDDEN_SIZE]: Sized,
    [(); Config::MLP_INNER]: Sized,
    [(); Config::NUM_ATTENTION_HEADS]: Sized,
    [(); Game::ACTION_SIZE]: Sized,
    [(); Game::STATE_SIZE]: Sized,
    [(); Config::NUM_LAYERS]: Sized,
    [(); Config::HIDDEN_SIZE / Config::NUM_ATTENTION_HEADS]: Sized,
    [(); 3 * Config::HIDDEN_SIZE * Config::SEQ_LEN]: Sized,
{
    pub fn train_on_batch<
        const B: usize,
        O: Optimizer<DTModel<Config, { Game::STATE_SIZE }, { Game::ACTION_SIZE }, E, D>, D, E>,
    >(
        &mut self,
        batch: BatchedInput<B, { Game::STATE_SIZE }, { Game::ACTION_SIZE }, E, D, Config>,
        actions: [Game::Action; B],
        optimizer: &mut O,
    ) -> E
    where
        [(); 3 * { Config::SEQ_LEN }]: Sized,
    {
        // Using the [dfdx dox](https://docs.rs/dfdx/latest/dfdx/index.html)
        let mut grads = self.0.alloc_grads();
        // Trace gradients through forward pass
        let batch: BatchedInput<
            B,
            { Game::STATE_SIZE },
            { Game::ACTION_SIZE },
            E,
            D,
            Config,
            OwnedTape<E, D>,
        > = (batch.0.traced(grads), batch.1, batch.2, batch.3);
        let y = self.0.forward_mut(batch);

        // compute loss & run backpropagation
        let actual = actions
            .map(|action| Game::action_to_tensor(&action))
            .stack();

        let pred = y;
        let loss = loss(pred, actual);
        let loss_value = loss.as_vec()[0];
        grads = loss.backward();

        // apply gradients
        optimizer.update(&mut self.0, &grads).unwrap();

        // Zero grads
        self.0.zero_grads(&mut grads);

        loss_value
    }
}
