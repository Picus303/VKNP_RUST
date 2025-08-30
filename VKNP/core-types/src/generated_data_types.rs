/// Supported element types
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DataType {
    F32,
    I32,
    U32,
}

impl DataType {
    /// Size of one element, in bytes
    pub fn size_in_bytes(self) -> usize {
        match self {
            DataType::F32 => std::mem::size_of::<f32>(),
            DataType::I32 => std::mem::size_of::<i32>(),
            DataType::U32 => std::mem::size_of::<u32>(),
        }
    }
}

/// Marker‐trait so we can go from T to DataType
pub trait Element: bytemuck::Pod {
    const DTYPE: DataType;
}

impl Element for f32 { const DTYPE: DataType = DataType::F32; }

impl Element for i32 { const DTYPE: DataType = DataType::I32; }

impl Element for u32 { const DTYPE: DataType = DataType::U32; }