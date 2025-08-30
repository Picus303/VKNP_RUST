mod utils;

use bytemuck::Zeroable;
use core_types::{BufferId, DataType, Element, ViewDescriptor};
use memory::MemoryManager;
use utils::compute_strides;
use std::marker::PhantomData;

/// Lightweight handle: (BufferId, ViewDescriptor, device_id, dtype)
pub struct Tensor<T: Element> {
    buffer_id: BufferId,
    device_id: usize,
    view:      ViewDescriptor,
    dtype:     DataType,
    _marker:   PhantomData<T>,
}

impl<T: Element> Tensor<T> {
    /* --------------------------------------------------------------------- */
    /* Constructors                                                          */
    /* --------------------------------------------------------------------- */

    /// Allocate an **uninitialised** tensor on the given device.
    pub fn empty(
        mgr:       &mut MemoryManager,
        shape:     &[usize],
        device_id: usize,
    ) -> Self {
        let elem_count = shape.iter().product::<usize>();
        let bytes      = elem_count * T::DTYPE.size_in_bytes();
        let buf_id     = mgr.allocate_raw(bytes).unwrap();

        let mut vd = ViewDescriptor::zeroed();
        vd.ndim = shape.len() as u32;
        let strides = compute_strides(shape);
        for (i, &d) in shape.iter().enumerate() {
            vd.shape[i]   = d as u32;
            vd.strides[i] = strides[i] as u32;
        }

        Tensor {
            buffer_id: buf_id,
            device_id,
            view:      vd,
            dtype:     T::DTYPE,
            _marker:   PhantomData,
        }
    }

    /// Construct a tensor by uploading a CPU slice into GPU
    pub fn from_vec(
        mgr:       &mut MemoryManager,
        data:      &[T],
        shape:     &[usize],
        device_id: usize,
    ) -> Self {
        // 1) allocate
        let elem_count = shape.iter().product::<usize>();
        let bytes      = elem_count * T::DTYPE.size_in_bytes();
        let buf_id     = mgr.allocate_raw(bytes).unwrap();
        // 2) write
        mgr.write_to_buffer(buf_id, data).unwrap();
        // 3) build the view descriptor
        let mut vd = ViewDescriptor::zeroed();
        vd.ndim = shape.len() as u32;
        let strides = compute_strides(shape);
        for (i, &d) in shape.iter().enumerate() {
            vd.shape[i]   = d as u32;
            vd.strides[i] = strides[i] as u32;
        }

        Tensor {
            buffer_id: buf_id,
            device_id,
            view:      vd,
            dtype:     T::DTYPE,
            _marker:   PhantomData,
        }
    }

    /// Download a tensor from GPU to CPU into a `Vec<T>`.
    pub fn to_vec(&self, mgr: &mut MemoryManager) -> Vec<T> {
        mgr.download_raw(self.buffer_id).unwrap()
    }

    /* --------------------------------------------------------------------- */
    /* Accessors                                                             */
    /* --------------------------------------------------------------------- */

    /// The view descriptor (shape, strides, offset)
    pub fn view(&self) -> &ViewDescriptor {
        &self.view
    }

    /// The internal BufferId
    pub fn buffer_id(&self) -> BufferId {
        self.buffer_id
    }

    /// The device index this tensor lives on
    pub fn device_id(&self) -> usize {
        self.device_id
    }

    /// The tensor’s DataType
    pub fn dtype(&self) -> DataType {
        self.dtype
    }
}

/* ------------------------------------------------------------------------- */
/*                                     Tests                                 */
/* ------------------------------------------------------------------------- */
#[cfg(test)]
mod tests {
    use super::*;
    use pollster::block_on;
    use vknp_core::GpuContext;
    use core_types::MAX_DIMS;

    #[test]
    fn test_empty_tensor_dtype_and_view() {
        let ctx = block_on(GpuContext::new()).unwrap();
        let mut mm = MemoryManager::new(ctx);

        let shape = [2, 3, 4];
        let t: Tensor<f32> = Tensor::empty(&mut mm, &shape, 0);

        // dtype must be f32
        assert_eq!(t.dtype(), DataType::F32);

        // shape padded to MAX_DIMS
        let mut expect_shape = [0u32; MAX_DIMS];
        for i in 0..shape.len() {
            expect_shape[i] = shape[i] as u32;
        }
        assert_eq!(t.view().shape, expect_shape);

        // strides for [2,3,4] row-major = [12,4,1]
        let mut expect_strides = [0u32; MAX_DIMS];
        expect_strides[..3].copy_from_slice(&[12, 4, 1]);
        assert_eq!(t.view().strides, expect_strides);

        assert_eq!(t.device_id(), 0);
    }

    #[test]
    fn test_from_vec_and_to_vec_preserves_data_and_dtype() {
        let ctx = block_on(GpuContext::new()).unwrap();
        let mut mm = MemoryManager::new(ctx);

        let shape = [2, 2];
        let data  = vec![1u32, 2, 3, 4];
        let t     = Tensor::from_vec(&mut mm, &data, &shape, 0);

        // data round-trip
        assert_eq!(t.to_vec(&mut mm), data);
        // dtype correct
        assert_eq!(t.dtype(), DataType::U32);

        // view correctness (shape only)
        let mut expect_shape = [0u32; MAX_DIMS];
        for i in 0..shape.len() { expect_shape[i] = shape[i] as u32; }
        assert_eq!(t.view().shape, expect_shape);
    }
}
