pub trait DTModelConfig {
    const NUM_ATTENTION_HEADS: usize;
    const HIDDEN_SIZE: usize;
    const MLP_INNER: usize;
    const SEQ_LEN: usize;
    const MAX_EPISODES_IN_GAME: usize;
    const NUM_LAYERS: usize;
}
