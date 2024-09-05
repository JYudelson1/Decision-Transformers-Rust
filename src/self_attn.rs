use dfdx::prelude::*;

use dfdx::nn::modules::Linear;
use num_traits::Float;

#[derive(Debug, Clone)]
pub struct SelfAttention<const HIDDEN: usize, const NUM_HEADS: usize, E: Dtype, D: Device<E>> {
    pub w_q: Linear<HIDDEN, HIDDEN, E, D>,
    pub w_k: Linear<HIDDEN, HIDDEN, E, D>,
    pub w_v: Linear<HIDDEN, HIDDEN, E, D>,
    pub w_o: Linear<HIDDEN, HIDDEN, E, D>,
}

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
                Self::module("w_q", |s| &s.w_q, |s| &mut s.w_q),
                Self::module("w_k", |s| &s.w_k, |s| &mut s.w_k),
                Self::module("w_v", |s| &s.w_v, |s| &mut s.w_v),
                Self::module("w_o", |s| &s.w_o, |s| &mut s.w_o),
            ),
            // Define how to construct the collection given its fields in the order they are given
            // above. This conversion is done using the ModuleFields trait.
            |(w_q, w_k, w_v, w_o)| SelfAttention { w_q, w_k, w_v, w_o },
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
    > Module<Tensor<(Const<SEQ_LEN>, Const<HIDDEN>), E, D>>
    for SelfAttention<HIDDEN, NUM_HEADS, E, D>
where
    [(); HIDDEN / NUM_HEADS]: Sized,
{
    type Output = Tensor<(Const<SEQ_LEN>, Const<HIDDEN>), E, D>;

    type Error = D::Err;

    fn try_forward(
        &self,
        input: Tensor<(Const<SEQ_LEN>, Const<HIDDEN>), E, D>,
    ) -> Result<Self::Output, Self::Error> {
        let v = self
            .w_v
            .forward(input.clone())
            .try_reshape::<(
                Const<SEQ_LEN>,
                Const<NUM_HEADS>,
                Const<{ HIDDEN / NUM_HEADS }>,
            )>()?
            .try_permute::<_, Axes3<1, 0, 2>>()?;

        let k = self
            .w_k
            .forward(input.clone())
            .try_reshape::<(
                Const<SEQ_LEN>,
                Const<NUM_HEADS>,
                Const<{ HIDDEN / NUM_HEADS }>,
            )>()?
            .try_permute::<_, Axes3<1, 2, 0>>()?;

        let q = self
            .w_q
            .forward(input)
            .try_reshape::<(
                Const<SEQ_LEN>,
                Const<NUM_HEADS>,
                Const<{ HIDDEN / NUM_HEADS }>,
            )>()?
            .try_permute::<_, Axes3<1, 0, 2>>()?;

        let scalar: E = E::from_f64(1.0 / ((HIDDEN / NUM_HEADS) as f64).sqrt()).unwrap();
        let qk = q.try_matmul(k)?.try_mul(scalar)?;
        let qk = qk.softmax::<Axis<2>>();

        let tokens = qk.try_matmul(v)?;
        let tokens = tokens.try_permute::<_, Axes3<1, 0, 2>>()?;
        let tokens = tokens.try_reshape::<(Const<SEQ_LEN>, Const<HIDDEN>)>()?;

        self.w_v.try_forward(tokens)
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
    > Module<Tensor<(Const<B>, Const<SEQ_LEN>, Const<HIDDEN>), E, D>>
    for SelfAttention<HIDDEN, NUM_HEADS, E, D>
where
    [(); HIDDEN / NUM_HEADS]: Sized,
{
    type Output = Tensor<(Const<B>, Const<SEQ_LEN>, Const<HIDDEN>), E, D>;

    type Error = D::Err;

    fn try_forward(
        &self,
        input: Tensor<(Const<B>, Const<SEQ_LEN>, Const<HIDDEN>), E, D>,
    ) -> Result<Self::Output, Self::Error> {
        let v = self
            .w_v
            .forward(input.clone())
            .try_reshape::<(
                Const<B>,
                Const<SEQ_LEN>,
                Const<NUM_HEADS>,
                Const<{ HIDDEN / NUM_HEADS }>,
            )>()?
            .try_permute::<_, Axes4<0, 2, 1, 3>>()?;

        let k = self
            .w_k
            .forward(input.clone())
            .try_reshape::<(
                Const<B>,
                Const<SEQ_LEN>,
                Const<NUM_HEADS>,
                Const<{ HIDDEN / NUM_HEADS }>,
            )>()?
            .try_permute::<_, Axes4<0, 2, 3, 1>>()?;

        let q = self
            .w_q
            .forward(input)
            .try_reshape::<(
                Const<B>,
                Const<SEQ_LEN>,
                Const<NUM_HEADS>,
                Const<{ HIDDEN / NUM_HEADS }>,
            )>()?
            .try_permute::<_, Axes4<0, 2, 1, 3>>()?;

        let scalar: E = E::from_f64(1.0 / ((HIDDEN / NUM_HEADS) as f64).sqrt()).unwrap();
        let qk = q.try_matmul(k)?.try_mul(scalar)?;
        let qk = qk.softmax::<Axis<3>>();

        let tokens = qk.try_matmul(v)?;
        let tokens = tokens.try_permute::<_, Axes4<0, 2, 1, 3>>()?;
        let tokens = tokens.try_reshape::<(Const<B>, Const<SEQ_LEN>, Const<HIDDEN>)>()?;

        self.w_v.try_forward(tokens)
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
where
    [(); HIDDEN / NUM_HEADS]: Sized,
{
    type Output = Tensor<(Const<B>, Const<SEQ_LEN>, Const<HIDDEN>), E, D, T>;

    type Error = D::Err;

    fn try_forward_mut(
        &mut self,
        input: Tensor<(Const<B>, Const<SEQ_LEN>, Const<HIDDEN>), E, D, T>,
    ) -> Result<Self::Output, Self::Error> {
        let (input, tape) = input.split_tape();
        let v = self
            .w_v
            .forward_mut(input.clone().put_tape(tape))
            .try_reshape::<(
                Const<B>,
                Const<SEQ_LEN>,
                Const<NUM_HEADS>,
                Const<{ HIDDEN / NUM_HEADS }>,
            )>()?
            .try_permute::<_, Axes4<0, 2, 1, 3>>()?;
        let (v, tape) = v.split_tape();

        let k = self
            .w_k
            .forward(input.clone().put_tape(tape))
            .try_reshape::<(
                Const<B>,
                Const<SEQ_LEN>,
                Const<NUM_HEADS>,
                Const<{ HIDDEN / NUM_HEADS }>,
            )>()?
            .try_permute::<_, Axes4<0, 2, 3, 1>>()?;
        let (k, tape) = k.split_tape();

        let q = self
            .w_q
            .forward(input.put_tape(tape))
            .try_reshape::<(
                Const<B>,
                Const<SEQ_LEN>,
                Const<NUM_HEADS>,
                Const<{ HIDDEN / NUM_HEADS }>,
            )>()?
            .try_permute::<_, Axes4<0, 2, 1, 3>>()?;

        let scalar: E = E::from_f64(1.0 / ((HIDDEN / NUM_HEADS) as f64).sqrt()).unwrap();
        let qk = q.try_matmul(k)?.try_mul(scalar)?;
        let qk = qk.softmax::<Axis<3>>();

        let tokens = qk.try_matmul(v)?;
        let tokens = tokens.try_permute::<_, Axes4<0, 2, 1, 3>>()?;
        let tokens = tokens.try_reshape::<(Const<B>, Const<SEQ_LEN>, Const<HIDDEN>)>()?;

        self.w_v.try_forward(tokens)
    }
}
