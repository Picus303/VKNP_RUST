/// Dynamically-typed Tensor: wraps `Tensor<T>` for various T
#[derive(From)]
pub enum TensorAny {
    F32(Tensor<f32>),
    I32(Tensor<i32>),
    U32(Tensor<u32>),
}

impl TensorAny {
    /// Query its DataType
    pub fn dtype(&self) -> DataType {
        match self {
            TensorAny::F32(_) => DataType::F32,
            TensorAny::I32(_) => DataType::I32,
            TensorAny::U32(_) => DataType::U32,
        }
    }

    /// Immutable access to its ViewDescriptor
    pub fn view(&self) -> &ViewDescriptor {
        match self {
            TensorAny::F32(t) => t.view(),
            TensorAny::I32(t) => t.view(),
            TensorAny::U32(t) => t.view(),
        }
    }
}