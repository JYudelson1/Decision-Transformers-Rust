use std::vec;

use dfdx::{
    nn::{
        modules::{LayerNorm1D, Linear, MultiHeadAttention, ReLU, Residual},
        Module, ModuleField, ModuleFields, ModuleMut, ModuleVisitor, TensorCollection,
    },
    prelude::Device,
    shapes::{Const, Dtype},
    tensor::{Tape, Tensor},
};

type Mlp<const HIDDEN: usize, const INNER: usize, E, D> = (
    Linear<HIDDEN, INNER, E, D>,
    ReLU,
    Linear<HIDDEN, INNER, E, D>,
);

type MlpSegment<const HIDDEN: usize, const INNER: usize, E, D> = (
    Residual<Mlp<HIDDEN, INNER, E, D>>,
    LayerNorm1D<HIDDEN, E, D>,
);

type MHASegment<const HIDDEN: usize, const NUM_HEADS: usize, E, D> = (
    Residual<MultiHeadAttention<HIDDEN, NUM_HEADS, HIDDEN, HIDDEN, E, D>>,
    LayerNorm1D<HIDDEN, E, D>,
);

type DecoderBlock<const HIDDEN: usize, const MLP_INNER: usize, const NUM_HEADS: usize, E, D> = (
    MHASegment<HIDDEN, NUM_HEADS, E, D>,
    MlpSegment<HIDDEN, MLP_INNER, E, D>,
);

pub struct CustomTransformerDecoder<
    const HIDDEN: usize,
    const MLP_INNER: usize,
    const NUM_HEADS: usize,
    const NUM_LAYERS: usize,
    const SEQ_LEN: usize,
    E: Dtype,
    D: Device<E>,
> {
    pub all_blocks: [DecoderBlock<HIDDEN, MLP_INNER, NUM_HEADS, E, D>; NUM_LAYERS],
}

impl<
        const HIDDEN: usize,
        const MLP_INNER: usize,
        const NUM_HEADS: usize,
        const NUM_LAYERS: usize,
        const SEQ_LEN: usize,
        E: Dtype,
        D: Device<E>,
    > CustomTransformerDecoder<HIDDEN, MLP_INNER, NUM_HEADS, NUM_LAYERS, SEQ_LEN, E, D>
{
    pub fn get_layer(&self, i: usize) -> &DecoderBlock<HIDDEN, MLP_INNER, NUM_HEADS, E, D> {
        &self.all_blocks[i]
    }

    pub fn get_layer_mut(
        &mut self,
        i: usize,
    ) -> &mut DecoderBlock<HIDDEN, MLP_INNER, NUM_HEADS, E, D> {
        &mut self.all_blocks[i]
    }

    pub fn from_blocks<E2: Dtype, D2: Device<E2>>(
        modules: Vec<(
            (
                Residual<MultiHeadAttention<HIDDEN, NUM_HEADS, HIDDEN, HIDDEN, E2, D2>>,
                LayerNorm1D<HIDDEN, E2, D2>,
            ),
            (
                Residual<(
                    Linear<HIDDEN, MLP_INNER, E2, D2>,
                    ReLU,
                    Linear<HIDDEN, MLP_INNER, E2, D2>,
                )>,
                LayerNorm1D<HIDDEN, E2, D2>,
            ),
        )>,
    ) -> CustomTransformerDecoder<HIDDEN, MLP_INNER, NUM_HEADS, NUM_LAYERS, SEQ_LEN, E2, D2> {
        CustomTransformerDecoder {
            all_blocks: std::array::from_fn(|i| modules[i].clone()),
        }
    }
}

impl<
        const HIDDEN: usize,
        const MLP_INNER: usize,
        const NUM_HEADS: usize,
        const NUM_LAYERS: usize,
        const SEQ_LEN: usize,
        E: Dtype,
        D: Device<E>,
    > TensorCollection<E, D>
    for CustomTransformerDecoder<HIDDEN, MLP_INNER, NUM_HEADS, NUM_LAYERS, SEQ_LEN, E, D>
where
    DecoderBlock<HIDDEN, MLP_INNER, NUM_HEADS, E, D>: TensorCollection<E, D>,
{
    type To<E2: Dtype, D2: Device<E2>> =
        CustomTransformerDecoder<HIDDEN, MLP_INNER, NUM_HEADS, NUM_LAYERS, SEQ_LEN, E2, D2>;

    fn iter_tensors<V: ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        todo!()
        // let fields: [_; NUM_LAYERS] = std::array::from_fn(|i| {
        //     Self::module(
        //         &format!("Layer_{i}"),
        //         |s| s.get_layer(i),
        //         |s| s.get_layer_mut(i),
        //     )
        // });
        // let fields = Vec::from(fields);
        // visitor.visit_fields(
        //     fields,
        //     // Define how to construct the collection given its fields in the order they are given
        //     // above. This conversion is done using the ModuleFields trait.
        //     |modules: Vec<_>| {
        //         Self::from_blocks::<
        //             <V as dfdx::nn::ModuleVisitor<Self, E, D>>::E2,
        //             <V as dfdx::nn::ModuleVisitor<Self, E, D>>::D2,
        //         >(modules)
        //     },
        // )
    }
}

impl<
        E: Dtype,
        D: Device<E>,
        T: Tape<E, D>,
        const HIDDEN: usize,
        const MLP_INNER: usize,
        const NUM_HEADS: usize,
        const NUM_LAYERS: usize,
        const SEQ_LEN: usize,
    > Module<Tensor<(Const<SEQ_LEN>, Const<HIDDEN>), E, D, T>>
    for CustomTransformerDecoder<HIDDEN, MLP_INNER, NUM_HEADS, NUM_LAYERS, SEQ_LEN, E, D>
{
    type Output = Tensor<(Const<SEQ_LEN>, Const<HIDDEN>), E, D, T>;

    type Error = ();

    fn try_forward(
        &self,
        mut input: Tensor<(Const<SEQ_LEN>, Const<HIDDEN>), E, D, T>,
    ) -> Result<Self::Output, Self::Error> {
        // for layer in self.all_blocks {
        //     input = layer.forward(input);
        // }
        // Ok(input)
        todo!()
    }
}

impl<
        E: Dtype,
        D: Device<E>,
        T: Tape<E, D>,
        const HIDDEN: usize,
        const MLP_INNER: usize,
        const NUM_HEADS: usize,
        const NUM_LAYERS: usize,
        const SEQ_LEN: usize,
        const B: usize,
    > Module<Tensor<(Const<B>, Const<SEQ_LEN>, Const<HIDDEN>), E, D, T>>
    for CustomTransformerDecoder<HIDDEN, MLP_INNER, NUM_HEADS, NUM_LAYERS, SEQ_LEN, E, D>
{
    type Output = Tensor<(Const<B>, Const<SEQ_LEN>, Const<HIDDEN>), E, D, T>;

    type Error = ();

    fn try_forward(
        &self,
        mut input: Tensor<(Const<B>, Const<SEQ_LEN>, Const<HIDDEN>), E, D, T>,
    ) -> Result<Self::Output, Self::Error> {
        // for layer in self.all_blocks {
        //     input = layer.forward(input);
        // }
        // Ok(input)
        todo!()
    }
}

impl<
        E: Dtype,
        D: Device<E>,
        T: Tape<E, D>,
        const HIDDEN: usize,
        const MLP_INNER: usize,
        const NUM_HEADS: usize,
        const NUM_LAYERS: usize,
        const SEQ_LEN: usize,
        const B: usize,
    > ModuleMut<Tensor<(Const<B>, Const<SEQ_LEN>, Const<HIDDEN>), E, D, T>>
    for CustomTransformerDecoder<HIDDEN, MLP_INNER, NUM_HEADS, NUM_LAYERS, SEQ_LEN, E, D>
{
    type Output = Tensor<(Const<B>, Const<SEQ_LEN>, Const<HIDDEN>), E, D, T>;

    type Error = ();

    fn try_forward_mut(
        &mut self,
        mut input: Tensor<(Const<B>, Const<SEQ_LEN>, Const<HIDDEN>), E, D, T>,
    ) -> Result<Self::Output, Self::Error> {
        // for layer in self.all_blocks {
        //     input = layer.forward(input);
        // }
        // Ok(input)
        todo!()
    }
}
