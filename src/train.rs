use dfdx::{optim::Adam, prelude::*};

use crate::{dt_model::BatchedInput, DTModel, DTModelWrapper, DTState};

pub fn loss<const B: usize, const A: usize, E: Dtype, D: Device<E>, T: Tape<E, D>>(
    action_pred: Tensor<(Const<B>, Const<A>), E, D, T>,
    action_actual: Tensor<(Const<B>, Const<A>), E, D, NoneTape>,
) -> Tensor<Rank0, E, D, T> {
    mse_loss(action_pred, action_actual)
}

impl<
        E: Dtype + From<f32> + num_traits::Float + rand_distr::uniform::SampleUniform,
        D: Device<E> + DeviceBuildExt,
        Game: DTState<E, D>,
    > DTModelWrapper<E, D, Game>
    where [(); Game::MAX_EPISODES_IN_GAME]: Sized,
    [(); Game::ACTION_SIZE]: Sized,
    [(); Game::STATE_SIZE]: Sized,
{
    pub fn train_on_batch<const B: usize>(
    &mut self, 
    batch: BatchedInput<
        { Game::EPISODES_IN_SEQ },
        B,
        { Game::STATE_SIZE },
        { Game::ACTION_SIZE },
        E,
        D,
    >,
    actions: [Game::Action; B],
    optimizer: &mut Adam<
        DTModel<{ Game::MAX_EPISODES_IN_GAME }, { Game::STATE_SIZE }, { Game::ACTION_SIZE }, E, D>,
        E,
        D,
    >,
    dev: &D,
) -> E where
    [(); 3 * { Game::EPISODES_IN_SEQ }]: Sized,
{
    // Using the [dfdx dox](https://docs.rs/dfdx/latest/dfdx/index.html)

    // Trace gradients through forward pass
    let batch: BatchedInput<
        { Game::EPISODES_IN_SEQ },
        B,
        { Game::STATE_SIZE },
        { Game::ACTION_SIZE },
        E,
        D,
        OwnedTape<E, D>,
    > = (
        batch.0.retaped(),
        batch.1.retaped(),
        batch.2.retaped(),
        batch.3,
    );
    let y = self.0.forward_mut(batch);

    // compute loss & run backpropagation
    let actual = actions
        .map(|action| Game::action_to_tensor(&action))
        .stack();
    let pred_index = dev.tensor([Game::EPISODES_IN_SEQ - 1; B]);
    let pred: Tensor<(Const<B>, Const<{ Game::ACTION_SIZE }>), E, D, OwnedTape<E, D>> =
    y.select(pred_index);
    let loss = loss(pred, actual);
    let loss_value = loss.as_vec()[0];
    let grads = loss.backward();

    // apply gradients
    optimizer.update(&mut self.0, &grads).expect("unused grads");

    loss_value
    }
}
