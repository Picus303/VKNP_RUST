use std::collections::HashMap;
use std::sync::{Arc, atomic::{AtomicU64, Ordering}};
use parking_lot::Mutex;
use anyhow::Result;

use vknp_core::GpuContext;
use vknp_core::types::{AbstractBuffer, BufferKind, BufferHandle};
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

    /// Allocate (or recycle) a buffer of `size_bytes`, returning a unique ID and Arc<AbstractBuffer>
    pub fn create_buffer(&self, size_bytes: usize) -> Result<(BufferId, BufferHandle)> {
        // MVP: alloc à chaque demande (recyclage à venir)
        let raw = self.ctx.create_buffer(size_bytes as u64, self.usage);
        let id = BufferId(self.next_id.fetch_add(1, Ordering::Relaxed));
        let handle = Arc::new(raw);

        self.entries.lock().insert(id, BufferEntry {
            buffer: handle.clone(),
            size: size_bytes,
        });
        Ok((id, handle))
    }

    /// Retrieve a clonable handle to the buffer for a given ID
    pub fn get(&self, id: BufferId) -> Option<BufferHandle> {
        self.entries.lock().get(&id).map(|e| e.buffer.clone())
    }

    pub fn get_buffer_size(&self, id: BufferId) -> Option<usize> {
        self.entries.lock().get(&id).map(|e| e.size)
    }

    /// Explicitly release a buffer by its ID
    pub fn release_buffer(&self, id: BufferId) {
        self.entries.lock().remove(&id);
    }

    /// Clear entries with only one reference (the one in the pool)
    pub fn clear_unused(&self) {
        self.entries.lock().retain(|_, entry| Arc::strong_count(&entry.buffer) > 1);
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

        let (id, _) = pool.create_buffer(1024).expect("Failed to allocate buffer");
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