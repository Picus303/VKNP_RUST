use bytemuck::{Pod, Zeroable};
use std::fmt;

include!("generated_data_types.rs");

/// Type alias for a buffer identifier
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct BufferId(pub u64);
impl fmt::Display for BufferId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "BufferId({})", self.0)
    }
}

/// Maximum number of dimensions for a view descriptor
pub const MAX_DIMS: usize = 8; // (B, C, H, W, D, T) + 2 should be enough

/// Descriptor for a view into a buffer
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, PartialEq, Eq)]
pub struct ViewDescriptor {
    pub offset:  u32,
    pub ndim:    u32,
    pub shape:   [u32; MAX_DIMS],
    pub strides: [u32; MAX_DIMS],
}