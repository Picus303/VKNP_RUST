use std::{
    collections::HashMap,
    sync::Arc,
};
use parking_lot::Mutex;

use vknp_core::{GpuContext, types::AbstractBindGroupLayout, types::AbstractComputePipeline};
use core_types::DataType;

/// Signature of a specialized kernel: shader + dtypes
#[derive(Clone, PartialEq, Eq, Hash)]
struct KernelKey {
    src:  Arc<str>,
    ent:  Arc<str>,
    t_in: Vec<DataType>,
    t_out: Vec<DataType>,
    p_len: usize,
}

struct PipelineBundle {
    pipeline: Arc<AbstractComputePipeline>,
    layout:   Arc<AbstractBindGroupLayout>,
}

/// Structure for managing kernel pipelines and layouts.
pub struct KernelManager {
    ctx:   GpuContext,
    cache: Mutex<HashMap<KernelKey, Arc<PipelineBundle>>>,
}

impl KernelManager {
    pub fn new(ctx: GpuContext) -> Self {
        Self { ctx, cache: Mutex::new(HashMap::new()) }
    }

    pub fn get(
        &self,
        src: &str,
        entry: &str,
        t_in: Vec<DataType>,
        t_out: Vec<DataType>,
        p_len: usize,
    ) -> Result<(Arc<AbstractComputePipeline>, Arc<AbstractBindGroupLayout>), String> {
        let key = KernelKey {
            src:  Arc::from(src),
            ent:  Arc::from(entry),
            t_in,
            t_out,
            p_len,
        };

        // cache lookup
        if let Some(b) = self.cache.lock().get(&key) {
            return Ok((b.pipeline.clone(), b.layout.clone()));
        }

        // create layout + pipeline via GpuContext
        let n_in = key.t_in.len() + key.p_len;
        let n_out = key.t_out.len();
        let layout   = self.ctx.create_storage_layout(n_in, n_out);
        let pipeline = self.ctx.create_compute_pipeline(src, entry, &layout);

        let bundle = Arc::new(PipelineBundle { pipeline: pipeline.clone(), layout: layout.clone() });
        self.cache.lock().insert(key, bundle);

        Ok((pipeline, layout))
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use pollster::block_on;
    use vknp_core::GpuContext;

    #[test]
    fn compile_and_retrieve() {
        // Create GPU context and kernel manager
        let ctx = block_on(GpuContext::new()).unwrap();
        let manager = KernelManager::new(ctx);

        // Define kernel source and parameters
        let src = r#"
            @group(0) @binding(0) var<storage, read>  A: array<f32>;
            @group(0) @binding(1) var<storage, read>  B: array<f32>;
            @group(0) @binding(2) var<storage, read_write> C: array<f32>;
            @compute @workgroup_size(64)
            fn add_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
                let i = gid.x;
                C[i] = A[i] + B[i];
            }
        "#;
        let entry = "add_kernel";
        let t_in = vec![DataType::F32, DataType::F32];
        let t_out = vec![DataType::F32];

        // Compile the kernel
        let (pipeline, layout) = manager.get(&src, entry, t_in.clone(), t_out.clone(), 0)
            .expect("shader compilation failed");

        // Retrieve and compare
        let (pipeline2, layout2) = manager.get(&src, entry, t_in.clone(), t_out.clone(), 0)
            .expect("shader compilation failed");

        assert_eq!(pipeline, pipeline2);
        assert_eq!(layout, layout2);
        }
    }