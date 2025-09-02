mod pool;

use anyhow::Result;
use bytemuck::{cast_slice, Pod};
use core_types::BufferId;
use pool::BufferPool;
use vknp_core::{GpuContext, types::BufferKind, types::AbstractBuffer};

/// Manages three buffer pools on **one** GPU device:
/// - `main_pool`         : STORAGE buffers that hold tensor data
/// - `staging_upload`    : MAP_WRITE + COPY_SRC  (CPU → GPU)
/// - `staging_download`  : MAP_READ  + COPY_DST  (GPU → CPU)
pub struct MemoryManager {
    ctx:              GpuContext,
    main_pool:        BufferPool,
    staging_upload:   BufferPool,
    staging_download: BufferPool,
}

impl MemoryManager {
    pub fn new(ctx: GpuContext) -> Self {
        // Create buffer pools for different usage types
        let main_pool        = BufferPool::new(ctx.clone(), BufferKind::Main);
        let staging_upload   = BufferPool::new(ctx.clone(), BufferKind::Upload);
        let staging_download = BufferPool::new(ctx.clone(), BufferKind::Download);

        Self { ctx, main_pool, staging_upload, staging_download }
    }

    /// Raw allocation
    pub fn allocate_raw(&mut self, size_bytes: usize) -> Result<BufferId> {
        Ok(self.main_pool.get_buffer(size_bytes)?)
    }

    /// Raw deallocation
    pub fn release(&mut self, id: BufferId) {
        self.main_pool.release_buffer(id);
    }

    /// Raw upload: CPU → GPU.
    pub fn write_to_buffer<T: Pod>(
        &mut self,
        dest_id: BufferId,
        data: &[T],
    ) -> Result<()> {
        let bytes = cast_slice(data);

        // 1) staging_upload: write via GpuContext
        let sid = self.staging_upload.get_buffer(bytes.len())?;
        {
            let buf = self.staging_upload.get(sid).unwrap();
            // delegate mapping + write + unmap
            self.ctx.write_buffer(buf, bytes);
        }

        // 2) copy staging_upload → main_pool[dest_id]
        let dst = self.main_pool.get(dest_id).unwrap();
        let src = self.staging_upload.get(sid).unwrap();
        self.ctx.copy_buffer_to_buffer(src, dst, bytes.len() as u64);

        // 3) cleanup staging
        self.staging_upload.release_buffer(sid);

        Ok(())
    }

    /// Raw download: GPU → CPU into a `Vec<T>`
    pub fn download_raw<T: Pod>(&mut self, id: BufferId) -> Result<Vec<T>> {
        // 1) Copy main → staging_download
        let src_buf = self.main_pool.get(id).unwrap();
        let size = src_buf.size();
        let sid = self.staging_download.get_buffer(size as usize)?;
        {
            let dst_buf = self.staging_download.get(sid).unwrap();
            self.ctx.copy_buffer_to_buffer(src_buf, dst_buf, size);
        }

        // 2) read entire staging buffer via GpuContext
        let dst_buf = self.staging_download.get(sid).unwrap();
        let bytes = self.ctx.read_buffer(dst_buf);

        // 3) cleanup staging
        self.staging_download.release_buffer(sid);

        // 4) cast to Vec<T>
        let vec = cast_slice::<u8, T>(&bytes).to_vec();
        Ok(vec)
    }

    /// Get a reference to a buffer in the main pool.
    pub fn get_ref(&self, id: BufferId) -> Option<&AbstractBuffer> {
        self.main_pool.get(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pollster::block_on;

    #[test]
    fn test_allocate_and_free() {
        let ctx  = block_on(GpuContext::new()).unwrap();
        let mut mm = MemoryManager::new(ctx);
        let id = mm.allocate_raw(256).unwrap();
        assert!(mm.get_ref(id).is_some());
        mm.release(id);
        assert!(mm.get_ref(id).is_none());
    }

    #[test]
    fn test_upload_download_roundtrip() {
        let ctx  = block_on(GpuContext::new()).unwrap();
        let mut mm = MemoryManager::new(ctx);
        let data  = vec![10u32, 20, 30, 40];

        let id = mm.allocate_raw(data.len() * std::mem::size_of::<u32>()).unwrap();
        mm.write_to_buffer(id, &data).unwrap();
        let back: Vec<u32> = mm.download_raw(id).unwrap();
        assert_eq!(data, back);
        mm.release(id);
    }
}
