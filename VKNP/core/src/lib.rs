pub mod types;

use anyhow::Result;
use std::sync::Arc;
use wgpu::{
    util::DeviceExt, BindGroupLayoutDescriptor, BindGroupLayoutEntry, ShaderStages,
    CommandEncoder, CommandEncoderDescriptor, Device, Instance, PollType, ComputePipelineDescriptor,
    PipelineLayoutDescriptor, Queue, ShaderModule, ShaderModuleDescriptor, ShaderSource,
    PipelineCompilationOptions, BindGroup, BindGroupEntry, BindGroupDescriptor, ComputePassDescriptor,
};

use types::{AbstractBuffer, AbstractBindGroupLayout, AbstractComputePipeline, BufferKind, BufferHandle};

/// Context for GPU operations
#[derive(Clone)]
pub struct GpuContext {
    pub device: Arc<Device>,
    pub queue:  Arc<Queue>,
}

impl GpuContext {
    /* ------------------------------------------------------------------ */
    /* Construction                                                       */
    /* ------------------------------------------------------------------ */
    pub async fn new() -> Result<Self> {
        let instance = Instance::default();
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .map_err(|e| anyhow::anyhow!("No suitable adapter found: {}", e))?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await?;

        Ok(Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
        })
    }

    /* ------------------------------------------------------------------ */
    /* Buffers                                                            */
    /* ------------------------------------------------------------------ */

    /// Allocate an uninitialised GPU buffer.
    pub fn create_buffer(&self, size: u64, usage: BufferKind) -> AbstractBuffer {
        AbstractBuffer(self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: usage.into(),
            mapped_at_creation: false,
        }))
    }

    /// Allocate and initialise a GPU buffer from host data.
    pub fn create_buffer_with_data(&self, data: &[u8], usage: BufferKind) -> AbstractBuffer {
        AbstractBuffer(self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: data,
            usage: usage.into(),
        }))
    }

    /// Blocking write: map-write, copy `data`, unmap.
    pub fn write_buffer(&self, buffer: &AbstractBuffer, data: &[u8]) {
        // simple blocking write (MAP_WRITE)
        let wgpu_buffer = buffer.raw();
        let slice = wgpu_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Write, |_| ());
        // wait
        self.device.poll(PollType::Wait).expect("Device poll failed");
        slice.get_mapped_range_mut().copy_from_slice(data);
        wgpu_buffer.unmap();
    }

    /// Blocking read: map-read entire buffer, return Vec<u8>.
    pub fn read_buffer(&self, buffer: &AbstractBuffer) -> Vec<u8> {
        let wgpu_buffer = buffer.raw();
        let slice = wgpu_buffer.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| ());
        self.device.poll(PollType::Wait).expect("Device poll failed");
        let data = slice.get_mapped_range().to_vec();
        wgpu_buffer.unmap();
        data
    }

    /* ------------------------------------------------------------------ */
    /* Encoder helpers                                                    */
    /* ------------------------------------------------------------------ */
    fn create_encoder(&self, label: &str) -> CommandEncoder {
        self.device
            .create_command_encoder(&CommandEncoderDescriptor { label: Some(label) })
    }

    fn submit_encoder(&self, encoder: CommandEncoder) {
        self.queue.submit(Some(encoder.finish()));
    }

    pub fn copy_buffer_to_buffer(&self, src: &AbstractBuffer, dst: &AbstractBuffer, size: u64) {
        let mut enc = self.create_encoder("copy-b2b");
        enc.copy_buffer_to_buffer(src.raw(), 0, dst.raw(), 0, size);
        self.submit_encoder(enc);
    }

    /* ------------------------------------------------------------------ */
    /* Shaders Preprocessing                                              */
    /* ------------------------------------------------------------------ */

    /// Create a storage buffer layout for a compute shader.
    pub fn create_storage_layout(&self, n_in: usize, n_out: usize) -> Arc<AbstractBindGroupLayout> {
        let total = n_in + n_out;
        let mut entries: Vec<BindGroupLayoutEntry> = Vec::with_capacity(total);

        // Input buffers
        for i in 0..n_in {
            entries.push(BindGroupLayoutEntry {
                binding: i as u32,
                visibility: ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }

        // Output buffers
        for i in 0..n_out {
            entries.push(BindGroupLayoutEntry {
                binding: (n_in + i) as u32,
                visibility: ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }

        let bgl = self.device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("storage-layout"),
            entries: &entries,
        });
        Arc::new(AbstractBindGroupLayout(bgl))
    }

    /// Create a compute pipeline from WGSL source code.
    pub fn create_compute_pipeline(
        &self,
        src: &str,
        entry: &str,
        layout: &AbstractBindGroupLayout,
    ) -> Arc<AbstractComputePipeline> {
        // Create shader module
        let module: ShaderModule = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("wgsl-module"),
            source: ShaderSource::Wgsl(src.into()),
        });
        // Create pipeline layout
        let pipeline_layout = self.device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("compute-pl-layout"),
            bind_group_layouts: &[&layout.0],
            push_constant_ranges: &[],
        });
        // Create compute pipeline
        let pipeline = self.device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("compute-pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some(entry),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });
        Arc::new(AbstractComputePipeline(pipeline))
    }

    /* ------------------------------------------------------------------ */
    /* Dispatch                                                           */
    /* ------------------------------------------------------------------ */

    fn create_storage_bind_group(
        &self,
        layout: &AbstractBindGroupLayout,
        inputs: &[&AbstractBuffer],
        outputs: &[&AbstractBuffer],
    ) -> BindGroup {
        let mut entries: Vec<BindGroupEntry> = Vec::with_capacity(inputs.len() + outputs.len());
        for (i, b) in inputs.iter().enumerate() {
            entries.push(BindGroupEntry {
                binding: i as u32,
                resource: b.0.as_entire_binding(),
            });
        }
        let off = inputs.len();
        for (i, b) in outputs.iter().enumerate() {
            entries.push(BindGroupEntry {
                binding: (off + i) as u32,
                resource: b.0.as_entire_binding(),
            });
        }
        self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("storage-bg"),
            layout: &layout.0,
            entries: &entries,
        })
    }

    pub fn dispatch_compute_1d(
        &self,
        pipeline: &AbstractComputePipeline,
        layout: &AbstractBindGroupLayout,
        inputs: &[BufferHandle],
        outputs: &[BufferHandle],
        total_elems: u32,
        workgroup_size: u32,
    ) {
        let input_refs: Vec<&AbstractBuffer> = inputs.iter().map(|arc| arc.as_ref()).collect();
        let output_refs: Vec<&AbstractBuffer> = outputs.iter().map(|arc| arc.as_ref()).collect();

        let bg = self.create_storage_bind_group(layout, &input_refs, &output_refs);
        let (x, _, _) = self.dispatch_size_1d(total_elems, workgroup_size);

        let mut enc = self.create_encoder("dispatch-1d");
        {
            let mut pass = enc.begin_compute_pass(&ComputePassDescriptor::default());
            pass.set_pipeline(&pipeline.0);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(x, 1, 1);
        }
        self.submit_encoder(enc);
    }

    /* ------------------------------------------------------------------ */
    /* Misc utils                                                         */
    /* ------------------------------------------------------------------ */

    /// Block until GPU idle / or PollType::Poll for non-blocking.
    fn device_poll(&self, mode: PollType) {
        self.device.poll(mode).expect("Device poll failed");
    }

    /// Register an uncaptured-error callback (helpful for debugging shaders).
    fn set_uncaptured_error_callback<F>(&self, cb: F)
    where
        F: wgpu::UncapturedErrorHandler + 'static,
    {
        self.device.on_uncaptured_error(Box::new(cb));
    }

    /// Helper: compute `(x,1,1)` for 1-D dispatch with `workgroup_size`.
    pub fn dispatch_size_1d(&self, total: u32, workgroup_size: u32) -> (u32, u32, u32) {
        ((total + workgroup_size - 1) / workgroup_size, 1, 1)
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use pollster::block_on;

    #[test]
    fn test_gpu_context_creation() {
        // Create a GPU context and check if it was created successfully
        let ctx = block_on(GpuContext::new()).expect("Failed to create GPU context");
        let limits = ctx.device.limits();

        // We always expect at least 1 invocation per workgroup
        assert!(limits.max_compute_invocations_per_workgroup > 0,
                "Expected max_compute_invocations_per_workgroup > 0, got {}", 
                limits.max_compute_invocations_per_workgroup);

        println!("Device created successfully: {:?}", ctx.device);
        println!("Device limits: {:?}", limits);
    }
}
