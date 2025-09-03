use crate::types::{OpSignature, PreparedOp, TensorAnyRef};


/// Trait to implement for each Op
pub trait Op: Send + Sync {
    /// Full signature
    fn signature(&self) -> &OpSignature;

    /// Given typed tensors, produce the GPU task(s)
    fn prepare(
        &self,
        inputs: &[TensorAnyRef],
        outputs: &[TensorAnyRef]
    ) -> PreparedOp;

    /// For a simple GPU kernel, return WGSL source + entry point
    fn shader_template(&self) -> (&'static str, &'static str);
}


/// Wrapper for op factory functions
pub struct OpFactory {
    pub name: &'static str,
    pub factory: fn() -> Box<dyn Op>,
}

// Collect all registered ops
inventory::collect!(OpFactory);