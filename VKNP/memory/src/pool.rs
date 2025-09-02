use std::collections::HashMap;
use vknp_core::GpuContext;
use core_types::BufferId;
use anyhow::Error;

use vknp_core::types::{AbstractBuffer, BufferKind};

/// Internal handle for storing the buffer and its meta-info
struct BufferEntry {
    buffer: AbstractBuffer,
    size: usize,
    ref_count: usize,
}

/// thread-safe pool of GPU buffers
pub struct BufferPool {
    ctx: GpuContext,
    next_id: u64,
    entries: HashMap<BufferId, BufferEntry>,
    usage: BufferKind,
}

impl BufferPool {
    /// Create a new BufferPool with the provided GpuContext and BufferUsages
    pub fn new(ctx: GpuContext, usage: BufferKind) -> Self {
        Self {
            ctx,
            next_id: 0,
            entries: HashMap::new(),
            usage,
        }
    }

    /// Allocate (or recycle) a buffer of `size_bytes`, returning a unique ID
    pub fn get_buffer(&mut self, size_bytes: usize) -> Result<BufferId, Error> {
        // for now always allocate a fresh buffer; future: reuse existing of adequate size
        let buffer = self.ctx.create_buffer(size_bytes as u64, self.usage);

        let id = BufferId(self.next_id);
        self.next_id = self.next_id.wrapping_add(1);
        self.entries.insert(id, BufferEntry {
            buffer,
            size: size_bytes,
            ref_count: 1,
        });
        Ok(id)
    }

    /// Retrieve a reference to the buffer for a given ID
    pub fn get(&self, id: BufferId) -> Option<&AbstractBuffer> {
        self.entries.get(&id).map(|e| &e.buffer)
    }

    /// Increase the reference count for a buffer
    pub fn inc_ref(&mut self, id: BufferId) -> Result<(), Error> {
        match self.entries.get_mut(&id) {
            Some(entry) => {
                entry.ref_count += 1;
                Ok(())
            },
            None => Err(anyhow::anyhow!("Buffer ID {} not found", id)),
        }
    }

    /// Decrease the reference count for a buffer, releasing it if count reaches zero
    pub fn dec_ref(&mut self, id: BufferId) -> Result<(), Error> {
        if let Some(entry) = self.entries.get_mut(&id) {
            // careful not to underflow
            if entry.ref_count == 0 {
                self.entries.remove(&id);
                Err(anyhow::anyhow!("Buffer ID {} already has zero reference count", id))
            } else {
                // decrease ref count and remove if it reaches zero
                entry.ref_count -= 1;
                if entry.ref_count == 0 {
                    self.entries.remove(&id);
                }
                Ok(())
            }
        } else {
            Err(anyhow::anyhow!("Buffer ID {} not found", id))
        }
    }

    /// Get the size of the buffer for a given ID
    pub fn get_buffer_size(&self, id: BufferId) -> Option<usize> {
        self.entries.get(&id).map(|e| e.size)
    }

    /// Explicitly release a buffer by its ID
    pub fn release_buffer(&mut self, id: BufferId) {
        self.entries.remove(&id);
    }

    /// Get the usage bitmap for this pool
    pub fn usage(&self) -> BufferKind {
        self.usage
    }
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
        let mut pool = BufferPool::new(ctx, usage);

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