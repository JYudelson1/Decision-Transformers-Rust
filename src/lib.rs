#![allow(incomplete_features)]
#![feature(generic_const_exprs)]

mod dt_model;
mod state_trait;
mod train;
mod trait_helpers;

pub use dt_model::DTModel;
pub use state_trait::{DTState, GetOfflineData};
pub use trait_helpers::DTModelWrapper;
