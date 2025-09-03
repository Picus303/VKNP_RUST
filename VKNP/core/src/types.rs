use std::sync::Arc;
use wgpu::{Buffer, BufferUsages, BindGroupLayout, ComputePipeline};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferKind {
    Main,
    Upload,
    Download,
}
impl From<BufferKind> for BufferUsages {
    fn from(kind: BufferKind) -> Self {
        match kind {
            BufferKind::Main => BufferUsages::STORAGE | BufferUsages::COPY_SRC | BufferUsages::COPY_DST,
            BufferKind::Upload => BufferUsages::MAP_WRITE | BufferUsages::COPY_SRC,
            BufferKind::Download => BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        }
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct AbstractBuffer(pub(crate) Buffer);
impl AbstractBuffer {
    pub(crate) fn raw(&self) -> &wgpu::Buffer {
        &self.0
    }

    pub fn size(&self) -> u64 {
        self.0.size()
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct AbstractBindGroupLayout(pub(crate) BindGroupLayout);
impl AbstractBindGroupLayout {
    pub(crate) fn raw(&self) -> &wgpu::BindGroupLayout {
        &self.0
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct AbstractComputePipeline(pub(crate) ComputePipeline);
impl AbstractComputePipeline {
    pub(crate) fn raw(&self) -> &wgpu::ComputePipeline {
        &self.0
    }
}

#[derive(Clone)]
pub struct BufferHandle(Arc<AbstractBuffer>);
impl BufferHandle {
    pub fn new(inner: Arc<AbstractBuffer>) -> Self { BufferHandle(inner) }
    pub fn as_raw(&self) -> &AbstractBuffer { &self.0 }
    pub fn strong_count(&self) -> usize { Arc::strong_count(&self.0) }
}

#[derive(Clone)]
pub struct BufferToken(Arc<AbstractBuffer>);
impl BufferToken {
    pub fn new(inner: Arc<AbstractBuffer>) -> Self { BufferToken(inner) }
}