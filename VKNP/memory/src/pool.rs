use std::collections::HashMap;
use std::sync::{Arc, atomic::{AtomicU64, Ordering}};
use parking_lot::Mutex;
use anyhow::Result;

use vknp_core::GpuContext;
use vknp_core::types::{AbstractBuffer, BufferKind};
use core_types::BufferId;

struct BufferEntry {
    buffer: Arc<AbstractBuffer>,
    size: usize,
}

/// thread-safe pool of GPU buffers
pub struct BufferPool {
    ctx: GpuContext,
    usage: BufferKind,
    next_id: AtomicU64,
    entries: Mutex<HashMap<BufferId, BufferEntry>>,
}

impl BufferPool {
    pub fn new(ctx: GpuContext, usage: BufferKind) -> Self {
        Self {
            ctx,
            usage,
            next_id: AtomicU64::new(0),
            entries: Mutex::new(HashMap::new()),
        }
    }

    /// Allocate (or recycle) a buffer of `size_bytes`, returning a unique ID
    pub fn get_buffer(&self, size_bytes: usize) -> Result<BufferId> {
        // MVP: alloc à chaque demande (recyclage à venir)
        let raw = self.ctx.create_buffer(size_bytes as u64, self.usage);
        let id = BufferId(self.next_id.fetch_add(1, Ordering::Relaxed));
        let handle = Arc::new(raw);

        self.entries.lock().insert(id, BufferEntry {
            buffer: handle,
            size: size_bytes,
        });
        Ok(id)
    }

    /// Retrieve a clonable handle to the buffer for a given ID
    pub fn get(&self, id: BufferId) -> Option<Arc<AbstractBuffer>> {
        self.entries.lock().get(&id).map(|e| e.buffer.clone())
    }

    pub fn get_buffer_size(&self, id: BufferId) -> Option<usize> {
        self.entries.lock().get(&id).map(|e| e.size)
    }

    /// Explicitly release a buffer by its ID
    pub fn release_buffer(&self, id: BufferId) {
        self.entries.lock().remove(&id);
    }

    pub fn usage(&self) -> BufferKind { self.usage }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pollster::block_on;

    #[test]
    fn test_buffer_pool_allocation() {
        // Allocate a buffer and check if it was created successfully
        // Also check if the size is correct and if it can be retrieved
        let ctx = block_on(GpuContext::new()).expect("Failed to create GPU context");
        let usage = BufferKind::Main;
        let pool = BufferPool::new(ctx, usage);

        let id = pool.get_buffer(1024).expect("Failed to allocate buffer");
        assert!(pool.get(id).is_some(), "Buffer should be allocated");

        let size = pool.get_buffer_size(id).expect("Buffer size should be available");
        assert_eq!(size, 1024, "Expected buffer size to be 1024 bytes");

        assert_eq!(pool.usage(), usage, "Usage bitmap should match");

        println!("Allocated buffer ID: {}", id);
        println!("Buffer size: {}", size);

        // Release the buffer
        pool.release_buffer(id);
        assert!(pool.get(id).is_none(), "Buffer should be released");
    }
}