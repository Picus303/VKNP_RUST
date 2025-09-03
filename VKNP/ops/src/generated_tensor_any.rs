/// Dynamically-typed Tensor: wraps `Tensor<T>` for various T
pub enum TensorAnyRef<'a> {
    F32(&'a Tensor<f32>),
    I32(&'a Tensor<i32>),
    U32(&'a Tensor<u32>),
}

impl<'a> TensorAnyRef<'a> {
    pub fn dtype(&self) -> DataType {
        match self {
            TensorAnyRef::F32(_) => DataType::F32,
            TensorAnyRef::I32(_) => DataType::I32,
            TensorAnyRef::U32(_) => DataType::U32,
        }
    }

    pub fn view(&self) -> &ViewDescriptor {
        match self {
            TensorAnyRef::F32(t) => t.view(),
            TensorAnyRef::I32(t) => t.view(),
            TensorAnyRef::U32(t) => t.view(),
        }
    }
}


impl<'a> From<&'a Tensor<f32>> for TensorAnyRef<'a> {
    fn from(t: &'a Tensor<f32>) -> Self {
        TensorAnyRef::F32(t)
    }
}
impl<'a> From<&'a Tensor<i32>> for TensorAnyRef<'a> {
    fn from(t: &'a Tensor<i32>) -> Self {
        TensorAnyRef::I32(t)
    }
}
impl<'a> From<&'a Tensor<u32>> for TensorAnyRef<'a> {
    fn from(t: &'a Tensor<u32>) -> Self {
        TensorAnyRef::U32(t)
    }
}