use crate::self_attn::SelfAttention;
use dfdx::{
    nn::{
        modules::{FastGeLU, LayerNorm1D, Linear},
        Module, ModuleMut, ModuleVisitor, TensorCollection,
    },
    prelude::*,
    shapes::{Const, Dtype},
    tensor::{Tape, Tensor, WithEmptyTape},
};
use num_traits::Float;
use rand_distr::uniform::SampleUniform;

struct DecoderBlock<
    const HIDDEN: usize,
    const SEQ_LEN: usize,
    const MLP_INNER: usize,
    const NUM_HEADS: usize,
    E: Dtype,
    D: Device<E>,
> {
    self_attn: SelfAttention<HIDDEN, NUM_HEADS, E, D>,
    ln: LayerNorm1D<HIDDEN, E, D>,
    mlp_1: Linear<HIDDEN, MLP_INNER, E, D>,
    gelu: FastGeLU,
    mlp_2: Linear<MLP_INNER, HIDDEN, E, D>,
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
                Self::module("ln", |s| &s.ln, |s| &mut s.ln),
                Self::module("mlp1", |s| &s.mlp_1, |s| &mut s.mlp_1),
                Self::module("gelu", |s| &s.gelu, |s| &mut s.gelu),
                Self::module("mlp2", |s| &s.mlp_2, |s| &mut s.mlp_2),
            ),
            // Define how to construct the collection given its fields in the order they are given
            // above. This conversion is done using the ModuleFields trait.
            |(self_attn, ln, mlp_1, gelu, mlp_2)| DecoderBlock {
                self_attn,
                ln,
                mlp_1,
                gelu,
                mlp_2,
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
    > Module<Tensor<(Const<SEQ_LEN>, Const<HIDDEN>), E, D>>
    for DecoderBlock<HIDDEN, SEQ_LEN, MLP_INNER, NUM_HEADS, E, D>
where
    [(); HIDDEN / NUM_HEADS]: Sized,
{
    type Output = Tensor<(Const<SEQ_LEN>, Const<HIDDEN>), E, D>;

    type Error = D::Err;

    fn try_forward(
        &self,
        input: Tensor<(Const<SEQ_LEN>, Const<HIDDEN>), E, D>,
    ) -> Result<Self::Output, Self::Error> {
        // Skip connection
        let skip = input.clone();

        // Self attention
        let mha_out = self.self_attn.forward(input);
        let residual_out = skip + mha_out;

        // Layer norm
        let ln_out = self.ln.forward(residual_out);

        // Second skip
        let mlp_skip = ln_out.clone();

        // Mlp
        let y = self.mlp_1.forward(ln_out);
        let y = self.gelu.forward(y);
        let y = self.mlp_2.forward(y);
        let y = self.gelu.forward(y);

        let mlp_out = y + mlp_skip;

        // Norm
        let out = self.ln.forward(mlp_out);

        Ok(out)
    }
}

// Batched Module
impl<
        const SEQ_LEN: usize,
        const HIDDEN: usize,
        const NUM_HEADS: usize,
        const MLP_INNER: usize,
        const B: usize,
        E: Dtype + Float,
        D: Device<E>,
    > Module<Tensor<(Const<B>, Const<SEQ_LEN>, Const<HIDDEN>), E, D>>
    for DecoderBlock<HIDDEN, SEQ_LEN, MLP_INNER, NUM_HEADS, E, D>
where
    [(); HIDDEN / NUM_HEADS]: Sized,
{
    type Output = Tensor<(Const<B>, Const<SEQ_LEN>, Const<HIDDEN>), E, D>;

    type Error = D::Err;

    fn try_forward(
        &self,
        input: Tensor<(Const<B>, Const<SEQ_LEN>, Const<HIDDEN>), E, D>,
    ) -> Result<Self::Output, Self::Error> {
        // Skip connection
        let skip = input.clone();

        // Self attention
        let mha_out = self.self_attn.forward(input);
        let residual_out = skip + mha_out;

        // Layer norm
        let ln_out = self.ln.forward(residual_out);

        // Second skip
        let mlp_skip = ln_out.clone();

        // Mlp
        let y = self.mlp_1.forward(ln_out);
        let y = self.gelu.forward(y);
        let y = self.mlp_2.forward(y);
        let y = self.gelu.forward(y);

        let mlp_out = y + mlp_skip;

        // Norm
        let out = self.ln.forward(mlp_out);

        Ok(out)
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
where
    [(); HIDDEN / NUM_HEADS]: Sized,
{
    type Output = Tensor<(Const<B>, Const<SEQ_LEN>, Const<HIDDEN>), E, D, T>;

    type Error = D::Err;

    fn try_forward_mut(
        &mut self,
        input: Tensor<(Const<B>, Const<SEQ_LEN>, Const<HIDDEN>), E, D, T>,
    ) -> Result<Self::Output, Self::Error> {
        // Skip connection
        let skip = input.with_empty_tape();

        // Self attention
        let mha_out = self.self_attn.forward_mut(self.ln.forward(input));
        let residual_out = skip + mha_out;

        // Layer norm
        //let ln_out = self.ln.forward_mut(residual_out);

        // Second skip
        let mlp_skip = residual_out.with_empty_tape();

        // Mlp
        let y = self.mlp_1.forward_mut(self.ln.forward(residual_out));
        let y = self.gelu.forward_mut(y);
        let y = self.mlp_2.forward_mut(y);
        let y = self.gelu.forward_mut(y);

        let mlp_out = y + mlp_skip;

        // Norm
        let out = self.ln.forward_mut(mlp_out);

        Ok(out)
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
    all_blocks: Vec<DecoderBlock<HIDDEN, SEQ_LEN, MLP_INNER, NUM_HEADS, E, D>>,
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
    fn get_layer(&self, i: usize) -> &DecoderBlock<HIDDEN, SEQ_LEN, MLP_INNER, NUM_HEADS, E, D> {
        &self.all_blocks[i]
    }

    fn get_layer_mut(
        &mut self,
        i: usize,
    ) -> &mut DecoderBlock<HIDDEN, SEQ_LEN, MLP_INNER, NUM_HEADS, E, D> {
        &mut self.all_blocks[i]
    }
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
        const HIDDEN: usize,
        const MLP_INNER: usize,
        const NUM_HEADS: usize,
        const NUM_LAYERS: usize,
        const SEQ_LEN: usize,
    > Module<Tensor<(Const<SEQ_LEN>, Const<HIDDEN>), E, D>>
    for CustomTransformerDecoder<HIDDEN, MLP_INNER, NUM_HEADS, NUM_LAYERS, SEQ_LEN, E, D>
where
    [(); HIDDEN / NUM_HEADS]: Sized,
{
    type Output = Tensor<(Const<SEQ_LEN>, Const<HIDDEN>), E, D>;

    type Error = D::Err;

    fn try_forward(
        &self,
        mut input: Tensor<(Const<SEQ_LEN>, Const<HIDDEN>), E, D>,
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
        const HIDDEN: usize,
        const MLP_INNER: usize,
        const NUM_HEADS: usize,
        const NUM_LAYERS: usize,
        const SEQ_LEN: usize,
        const B: usize,
    > Module<Tensor<(Const<B>, Const<SEQ_LEN>, Const<HIDDEN>), E, D>>
    for CustomTransformerDecoder<HIDDEN, MLP_INNER, NUM_HEADS, NUM_LAYERS, SEQ_LEN, E, D>
where
    [(); HIDDEN / NUM_HEADS]: Sized,
{
    type Output = Tensor<(Const<B>, Const<SEQ_LEN>, Const<HIDDEN>), E, D>;

    type Error = D::Err;

    fn try_forward(
        &self,
        mut input: Tensor<(Const<B>, Const<SEQ_LEN>, Const<HIDDEN>), E, D>,
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
where
    [(); HIDDEN / NUM_HEADS]: Sized,
{
    type Output = Tensor<(Const<B>, Const<SEQ_LEN>, Const<HIDDEN>), E, D, T>;

    type Error = D::Err;

    fn try_forward_mut(
        &mut self,
        mut input: Tensor<(Const<B>, Const<SEQ_LEN>, Const<HIDDEN>), E, D, T>,
    ) -> Result<Self::Output, Self::Error> {
        
        for layer in &mut self.all_blocks {
            input = layer.forward_mut(input);
        }
        Ok(input)
    }
}

#[allow(dead_code)]
fn build_mask<const SEQ_LEN: usize, const HIDDEN: usize, E: Dtype, D: Device<E>>(
) -> Tensor<(Const<SEQ_LEN>, Const<HIDDEN>), E, D> {
    let dev = D::default();

    let mask: [Tensor<(Const<HIDDEN>,), E, D>; SEQ_LEN] = std::array::from_fn(|i| {
        if i == SEQ_LEN - 1 {
            dev.zeros()
        } else {
            dev.ones()
        }
    });

    mask.stack()
}
