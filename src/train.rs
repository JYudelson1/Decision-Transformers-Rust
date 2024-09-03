use dfdx::prelude::*;

pub fn loss<const A: usize, E: Dtype, D: Device<E>>(
    action_pred: Tensor<(Const<A>,), E, D>,
    action_actual: Tensor<(Const<A>,), E, D>,
) -> Tensor<Rank0, E, D> {
    mse_loss(action_pred, action_actual)
}
