use dfdx::{
    nn::{
        modules::{LayerNorm1D, Linear, MultiHeadAttention, ReLU, Residual},
        Module, ModuleMut, ModuleVisitor, TensorCollection,
    },
    prelude::Device,
    shapes::{Const, Dtype},
    tensor::{PutTape, SplitTape, Tape, Tensor},
};
use num_traits::Float;
use rand_distr::uniform::SampleUniform;

type Mlp<const HIDDEN: usize, const INNER: usize, E, D> = (
    Linear<HIDDEN, INNER, E, D>,
    ReLU,
    Linear<INNER, HIDDEN, E, D>,
);

type MlpSegment<const HIDDEN: usize, const INNER: usize, E, D> = (
    Residual<Mlp<HIDDEN, INNER, E, D>>,
    LayerNorm1D<HIDDEN, E, D>,
);

#[derive(Debug, Clone)]
pub struct SelfAttention<const HIDDEN: usize, const NUM_HEADS: usize, E: Dtype, D: Device<E>>(
    pub MultiHeadAttention<HIDDEN, NUM_HEADS, HIDDEN, HIDDEN, E, D>,
);

impl<
        const HIDDEN: usize,
        const NUM_HEADS: usize,
        E: Dtype + num_traits::Float + rand_distr::uniform::SampleUniform,
        D: Device<E>,
    > TensorCollection<E, D> for SelfAttention<HIDDEN, NUM_HEADS, E, D>
{
    type To<E2: Dtype, D2: Device<E2>> = SelfAttention<HIDDEN, NUM_HEADS, E2, D2>;

    fn iter_tensors<V: ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(
            (
                // Define name of each field and how to access it, using ModuleField for Modules,
                // and TensorField for Tensors.
                Self::module("mhs", |s| &s.0, |s| &mut s.0),
            ),
            // Define how to construct the collection given its fields in the order they are given
            // above. This conversion is done using the ModuleFields trait.
            |mha| SelfAttention(mha.0),
        )
    }
}

// Single Module
impl<
        const SEQ_LEN: usize,
        const HIDDEN: usize,
        const NUM_HEADS: usize,
        E: Dtype + Float,
        D: Device<E>,
        T: Tape<E, D>,
    > Module<Tensor<(Const<SEQ_LEN>, Const<HIDDEN>), E, D, T>>
    for SelfAttention<HIDDEN, NUM_HEADS, E, D>
{
    type Output = Tensor<(Const<SEQ_LEN>, Const<HIDDEN>), E, D, T>;

    type Error = D::Err;

    fn try_forward(
        &self,
        input: Tensor<(Const<SEQ_LEN>, Const<HIDDEN>), E, D, T>,
    ) -> Result<Self::Output, Self::Error> {
        let (input, tape) = input.split_tape();
        let input = (input.clone().put_tape(tape), input.clone(), input);
        let out = self.0.forward(input);

        Ok(out)
    }
}

// Batched Mut
impl<
        const SEQ_LEN: usize,
        const HIDDEN: usize,
        const NUM_HEADS: usize,
        const B: usize,
        E: Dtype + Float,
        D: Device<E>,
        T: Tape<E, D>,
    > Module<Tensor<(Const<B>, Const<SEQ_LEN>, Const<HIDDEN>), E, D, T>>
    for SelfAttention<HIDDEN, NUM_HEADS, E, D>
{
    type Output = Tensor<(Const<B>, Const<SEQ_LEN>, Const<HIDDEN>), E, D, T>;

    type Error = D::Err;

    fn try_forward(
        &self,
        input: Tensor<(Const<B>, Const<SEQ_LEN>, Const<HIDDEN>), E, D, T>,
    ) -> Result<Self::Output, Self::Error> {
        let (input, tape) = input.split_tape();
        let input = (input.clone().put_tape(tape), input.clone(), input);
        let out = self.0.forward(input);

        Ok(out)
    }
}

// Batched ModuleMut
impl<
        const SEQ_LEN: usize,
        const HIDDEN: usize,
        const NUM_HEADS: usize,
        const B: usize,
        E: Dtype + Float,
        D: Device<E>,
        T: Tape<E, D>,
    > ModuleMut<Tensor<(Const<B>, Const<SEQ_LEN>, Const<HIDDEN>), E, D, T>>
    for SelfAttention<HIDDEN, NUM_HEADS, E, D>
{
    type Output = Tensor<(Const<B>, Const<SEQ_LEN>, Const<HIDDEN>), E, D, T>;

    type Error = D::Err;

    fn try_forward_mut(
        &mut self,
        input: Tensor<(Const<B>, Const<SEQ_LEN>, Const<HIDDEN>), E, D, T>,
    ) -> Result<Self::Output, Self::Error> {
        let (input, tape) = input.split_tape();
        let input = (input.clone().put_tape(tape), input.clone(), input);
        let out = self.0.forward(input);

        Ok(out)
    }
}

type MHASegment<const HIDDEN: usize, const SEQ_LEN: usize, const NUM_HEADS: usize, E, D> = (
    Residual<SelfAttention<HIDDEN, NUM_HEADS, E, D>>,
    LayerNorm1D<HIDDEN, E, D>,
);

struct DecoderBlock<
    const HIDDEN: usize,
    const SEQ_LEN: usize,
    const MLP_INNER: usize,
    const NUM_HEADS: usize,
    E: Dtype,
    D: Device<E>,
> {
    self_attn: SelfAttention<HIDDEN, NUM_HEADS, E, D>,
    ln1: LayerNorm1D<HIDDEN, E, D>,
    mlp_1: Linear<HIDDEN, MLP_INNER, E, D>,
    relu: ReLU,
    mlp_2: Linear<MLP_INNER, HIDDEN, E, D>,
    ln2: LayerNorm1D<HIDDEN, E, D>,
}

impl<
        const HIDDEN: usize,
        const SEQ_LEN: usize,
        const MLP_INNER: usize,
        const NUM_HEADS: usize,
        E: Dtype + Float + SampleUniform,
        D: Device<E>,
    > TensorCollection<E, D> for DecoderBlock<HIDDEN, SEQ_LEN, MLP_INNER, NUM_HEADS, E, D>
{
    type To<E2: Dtype, D2: Device<E2>> =
        DecoderBlock<HIDDEN, SEQ_LEN, MLP_INNER, NUM_HEADS, E2, D2>;

    fn iter_tensors<V: ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        visitor.visit_fields(
            (
                // Define name of each field and how to access it, using ModuleField for Modules,
                // and TensorField for Tensors.
                Self::module("self_attn", |s| &s.self_attn, |s| &mut s.self_attn),
                Self::module("ln1", |s| &s.ln1, |s| &mut s.ln1),
                Self::module("mlp1", |s| &s.mlp_1, |s| &mut s.mlp_1),
                Self::module("relu", |s| &s.relu, |s| &mut s.relu),
                Self::module("mlp2", |s| &s.mlp_2, |s| &mut s.mlp_2),
                Self::module("ln2", |s| &s.ln2, |s| &mut s.ln2),
            ),
            // Define how to construct the collection given its fields in the order they are given
            // above. This conversion is done using the ModuleFields trait.
            |(self_attn, ln1, mlp_1, relu, mlp_2, ln2)| DecoderBlock {
                self_attn,
                ln1,
                mlp_1,
                relu,
                mlp_2,
                ln2,
            },
        )
    }
}

// Single Module
impl<
        const SEQ_LEN: usize,
        const HIDDEN: usize,
        const NUM_HEADS: usize,
        const MLP_INNER: usize,
        E: Dtype + Float,
        D: Device<E>,
        T: Tape<E, D>,
    > Module<Tensor<(Const<SEQ_LEN>, Const<HIDDEN>), E, D, T>>
    for DecoderBlock<HIDDEN, SEQ_LEN, MLP_INNER, NUM_HEADS, E, D>
{
    type Output = Tensor<(Const<SEQ_LEN>, Const<HIDDEN>), E, D, T>;

    type Error = D::Err;

    fn try_forward(
        &self,
        input: Tensor<(Const<SEQ_LEN>, Const<HIDDEN>), E, D, T>,
    ) -> Result<Self::Output, Self::Error> {
        todo!();
    }
}

// Batched Mut
impl<
        const SEQ_LEN: usize,
        const HIDDEN: usize,
        const NUM_HEADS: usize,
        const MLP_INNER: usize,
        const B: usize,
        E: Dtype + Float,
        D: Device<E>,
        T: Tape<E, D>,
    > Module<Tensor<(Const<B>, Const<SEQ_LEN>, Const<HIDDEN>), E, D, T>>
    for DecoderBlock<HIDDEN, SEQ_LEN, MLP_INNER, NUM_HEADS, E, D>
{
    type Output = Tensor<(Const<B>, Const<SEQ_LEN>, Const<HIDDEN>), E, D, T>;

    type Error = D::Err;

    fn try_forward(
        &self,
        input: Tensor<(Const<B>, Const<SEQ_LEN>, Const<HIDDEN>), E, D, T>,
    ) -> Result<Self::Output, Self::Error> {
        todo!()
    }
}

// Batched ModuleMut
impl<
        const SEQ_LEN: usize,
        const HIDDEN: usize,
        const NUM_HEADS: usize,
        const B: usize,
        const MLP_INNER: usize,
        E: Dtype + Float,
        D: Device<E>,
        T: Tape<E, D>,
    > ModuleMut<Tensor<(Const<B>, Const<SEQ_LEN>, Const<HIDDEN>), E, D, T>>
    for DecoderBlock<HIDDEN, SEQ_LEN, MLP_INNER, NUM_HEADS, E, D>
{
    type Output = Tensor<(Const<B>, Const<SEQ_LEN>, Const<HIDDEN>), E, D, T>;

    type Error = D::Err;

    fn try_forward_mut(
        &mut self,
        input: Tensor<(Const<B>, Const<SEQ_LEN>, Const<HIDDEN>), E, D, T>,
    ) -> Result<Self::Output, Self::Error> {
        todo!()
    }
}

pub struct CustomTransformerDecoder<
    const HIDDEN: usize,
    const MLP_INNER: usize,
    const NUM_HEADS: usize,
    const NUM_LAYERS: usize,
    const SEQ_LEN: usize,
    E: Dtype,
    D: Device<E>,
> {
    pub all_blocks: Vec<DecoderBlock<HIDDEN, SEQ_LEN, MLP_INNER, NUM_HEADS, E, D>>,
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
    pub fn get_layer(
        &self,
        i: usize,
    ) -> &DecoderBlock<HIDDEN, SEQ_LEN, MLP_INNER, NUM_HEADS, E, D> {
        &self.all_blocks[i]
    }

    pub fn get_layer_mut(
        &mut self,
        i: usize,
    ) -> &mut DecoderBlock<HIDDEN, SEQ_LEN, MLP_INNER, NUM_HEADS, E, D> {
        &mut self.all_blocks[i]
    }

    // pub fn from_blocks<
    //     V: ModuleVisitor<
    //         CustomTransformerDecoder<HIDDEN, MLP_INNER, NUM_HEADS, NUM_LAYERS, SEQ_LEN, E, D>,
    //         E,
    //         D,
    //     >,
    // >(
    //     blocks: Vec<<DecoderBlock<HIDDEN, SEQ_LEN, MLP_INNER, NUM_HEADS, E, D> as TensorCollection<E, D>>::To<V::E2, V::D2>>,
    // ) -> CustomTransformerDecoder<HIDDEN, MLP_INNER, NUM_HEADS, NUM_LAYERS, SEQ_LEN, V::E2, V::D2>
    // where
    //     DecoderBlock<HIDDEN, SEQ_LEN, MLP_INNER, NUM_HEADS, E, D>: TensorCollection<E, D>,
    //     CustomTransformerDecoder<HIDDEN, MLP_INNER, NUM_HEADS, NUM_LAYERS, SEQ_LEN, E, D>:
    //         TensorCollection<E, D>,
    //     V::E2: Dtype,
    //     V::D2: Device<V::E2>,
    // {
    //     let layers: Vec<DecoderBlock<HIDDEN, SEQ_LEN, MLP_INNER, NUM_HEADS, V::E2, V::D2>> = blocks.into_iter().map(|block: <DecoderBlock<HIDDEN, SEQ_LEN, MLP_INNER, NUM_HEADS, E, D> as TensorCollection<E, D>>::To<V::E2, V::D2>| block.into()).collect();
    //     CustomTransformerDecoder { all_blocks: layers }
    // }
}

impl<
        const HIDDEN: usize,
        const MLP_INNER: usize,
        const NUM_HEADS: usize,
        const NUM_LAYERS: usize,
        const SEQ_LEN: usize,
        E: Dtype + Float + SampleUniform,
        D: Device<E>,
    > TensorCollection<E, D>
    for CustomTransformerDecoder<HIDDEN, MLP_INNER, NUM_HEADS, NUM_LAYERS, SEQ_LEN, E, D>
{
    type To<E2: Dtype, D2: Device<E2>> =
        CustomTransformerDecoder<HIDDEN, MLP_INNER, NUM_HEADS, NUM_LAYERS, SEQ_LEN, E2, D2>;

    fn iter_tensors<V: ModuleVisitor<Self, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        let fields: [_; NUM_LAYERS] = std::array::from_fn(|i| {
            Self::module(
                "layer_i",
                move |s| s.get_layer(i),
                move |s| s.get_layer_mut(i),
            )
        });
        let fields = Vec::from(fields);
        visitor.visit_fields(
            fields,
            // Define how to construct the collection given its fields in the order they are given
            // above. This conversion is done using the ModuleFields trait.
            |modules| CustomTransformerDecoder {
                all_blocks: modules,
            },
        )
    }
}

// Single Module
impl<
        E: Dtype + Float,
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

    type Error = D::Err;

    fn try_forward(
        &self,
        mut input: Tensor<(Const<SEQ_LEN>, Const<HIDDEN>), E, D, T>,
    ) -> Result<Self::Output, Self::Error> {
        for layer in &self.all_blocks {
            input = layer.forward(input);
        }
        Ok(input)
    }
}

// Batched Module
impl<
        E: Dtype + Float,
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

    type Error = D::Err;

    fn try_forward(
        &self,
        mut input: Tensor<(Const<B>, Const<SEQ_LEN>, Const<HIDDEN>), E, D, T>,
    ) -> Result<Self::Output, Self::Error> {
        for layer in &self.all_blocks {
            input = layer.forward(input);
        }
        Ok(input)
    }
}

// Batched ModuleMut
impl<
        E: Dtype + Float,
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

    type Error = D::Err;

    fn try_forward_mut(
        &mut self,
        mut input: Tensor<(Const<B>, Const<SEQ_LEN>, Const<HIDDEN>), E, D, T>,
    ) -> Result<Self::Output, Self::Error> {
        for layer in &self.all_blocks {
            input = layer.forward(input);
        }
        Ok(input)
    }
}
