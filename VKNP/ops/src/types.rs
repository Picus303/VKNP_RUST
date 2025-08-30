use core_types::{BufferId, DataType, ViewDescriptor};
use derive_more::From;
use tensor::Tensor;

include!("generated_tensor_any.rs");

/// The full signature of an operation:
/// - `name`
/// - number of tensor inputs
/// - allowed DataTypes per tensor input
/// - expected output DataTypes
#[derive(Debug, Clone)]
pub struct OpSignature {
    pub name:           &'static str,
    pub num_inputs:     usize,
    pub num_outputs:    usize,
    pub input_dtypes:   Vec<Vec<DataType>>,
    pub output_dtypes:  Vec<Vec<DataType>>,
}

/// A GPU “kernel” ready to bind & dispatch
#[derive(Debug, Clone)]
pub struct GpuTask {
    pub pipeline_source:    String,
    pub entry_point:        String,
    pub input_descs:        Vec<ViewDescriptor>,
    pub output_descs:       Vec<ViewDescriptor>,
    pub input_types:        Vec<DataType>,
    pub output_types:       Vec<DataType>,
    pub input_ids:          Vec<BufferId>,
    pub output_ids:         Vec<BufferId>,
}

/// Result of preparing an Op: either a single GPU kernel
/// or a sequence of sub-ops (for composites like FFT)
#[derive(Debug, Clone)]
pub enum PreparedOp {
    Gpu(GpuTask),
    Composite(Vec<PreparedOp>),
}

/// Errors during signature validation
#[derive(Debug)]
pub enum OpError {
    UnknownOp(String),
    ArityMismatch { op: String, expected: usize, found: usize },
    DtypeMismatch  { op: String, index: usize, expected: Vec<DataType>, found: DataType },
}

/// Trait to implement for each Op to work with inventory
pub trait RegistrationInfo {
    /// Unique name for the operation
    const NAME: &'static str;
}