use dfdx::prelude::*;

use crate::DTModelConfig;

pub fn stack_usize<E: Dtype, D: Device<E>, Config: DTModelConfig>(
    tensors: [Tensor<(), usize, D>; Config::SEQ_LEN],
    dev: &D,
) -> Tensor<(Const<{ Config::SEQ_LEN }>,), usize, D> {
    let mut data: [usize; Config::SEQ_LEN] = [0; Config::SEQ_LEN];

    for i in 0..Config::SEQ_LEN {
        data[i] = tensors[i].as_vec()[0];
    }

    dev.tensor(data)
}

pub fn stack_usize_batched<
    E: Dtype,
    D: Device<E> + CopySlice<usize>,
    Config: DTModelConfig,
    const B: usize,
>(
    tensors: [Tensor<(Const<{ Config::SEQ_LEN }>,), usize, D>; B],
    dev: &D,
) -> Tensor<(Const<B>, Const<{ Config::SEQ_LEN }>), usize, D> {
    let mut data: [[usize; Config::SEQ_LEN]; B] = [[0; Config::SEQ_LEN]; B];

    for i in 0..Config::SEQ_LEN {
        tensors[i].copy_into(&mut data[i]);
    }

    dev.tensor(data)
    //dev.tensor(tensors)
}
