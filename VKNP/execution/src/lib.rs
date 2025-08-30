mod kernel_manager;

use vknp_core::GpuContext;
use vknp_core::types::AbstractBuffer;
use memory::MemoryManager;
use vknp_ops::types::{GpuTask, PreparedOp};

use kernel_manager::KernelManager;

/*

pub struct ExecutionEngine {
    device:  Arc<wgpu::Device>,
    queue:   Arc<wgpu::Queue>,
    kernels: KernelManager,
    binder:  BindGroupBuilder,
}

impl ExecutionEngine {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let kernels = KernelManager::new(device.clone());
        let binder  = BindGroupBuilder::new(device.clone());
        Self { device, queue, kernels, binder }
    }

    /* --------------------------------------------------------------------- */
    /* Public API                                                            */
    /* --------------------------------------------------------------------- */

    pub fn run_prepared(
        &mut self,
        p: PreparedOp,
        mm: &mut MemoryManager,
    ) -> Result<Vec<TensorAny>, ExecutionError> {
        match p {
            PreparedOp::Gpu(task)        => self.run_gpu_task(task, mm),
            PreparedOp::Composite(list)  => {
                let mut last_outputs = Vec::new();
                for sub in list {
                    last_outputs = self.run_prepared(sub, mm)?;
                }
                Ok(last_outputs)
            }
        }
    }

    /* --------------------------------------------------------------------- */
    /* Private helpers                                                       */
    /* --------------------------------------------------------------------- */

    fn run_gpu_task(
        &mut self,
        task: GpuTask,
        mm: &mut MemoryManager,
    ) -> Result<Vec<TensorAny>, ExecutionError> {
        // ------- 1. allocate outputs --------------------------------------
        let mut output_buffers: Vec<Arc<wgpu::Buffer>> = Vec::new();
        let mut output_tensors: Vec<TensorAny> = Vec::new();

        for vd in &task.output_descs {
            let elem_count = (0..vd.ndim as usize)
                .map(|i| vd.shape[i] as usize)
                .product::<usize>();
            let bytes = elem_count * DataType::F32.size_in_bytes(); // TODO use dtype per-output
            let id = mm.allocate_raw(bytes)?;                       // returns BufferId
            let buf = Arc::clone(mm.main_pool().get(id).unwrap()); // hypothetical getter
            output_buffers.push(buf);

            // wrap BufferId + vd in a TensorAny (only F32 for now)
            output_tensors.push(match DataType::F32 {
                DataType::F32 => TensorAny::F32(
                    tensor::Tensor::<f32>::from_raw(id, vd.clone(), 0) // device_id=0 mono-GPU
                ),
                _ => return Err(ExecutionError::UnsupportedDtype(DataType::F32)),
            });
        }

        // ------- 2. collect input buffers ---------------------------------
        let input_buffers: Vec<_> = task
            .input_descs
            .iter()
            .map(|vd| {
                // Ici on suppose que MemoryManager sait retrouver le buffer
                // à partir du ViewDescriptor (sinon il faut passer BufferId
                // dans GpuTask directement).
                Arc::clone(mm.buffer_from_view(vd).unwrap())
            })
            .collect();

        // ------- 3. pipeline & bind group ---------------------------------
        let layout    = self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("dyn_layout0"),
            entries: &(0..(input_buffers.len() + output_buffers.len()))
                .map(|i| wgpu::BindGroupLayoutEntry {
                    binding: i as u32,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: if i < input_buffers.len() {
                            wgpu::BufferBindingType::Storage { read_only: true }
                        } else {
                            wgpu::BufferBindingType::Storage { read_only: false }
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect::<Vec<_>>(),
        });

        let pipeline = self.kernels.get_pipeline(
            &task.pipeline_source,
            &task.entry_point,
            &layout,
        );

        let bg = self.binder.build(
            &layout,
            &task.input_descs,
            &output_buffers,
            &input_buffers,
        );

        // ------- 4. dispatch ----------------------------------------------
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bg, &[]);

            // TODO : derive a better dispatch size
            let total_elements = task.output_descs[0].shape[0];
            let wg_size = 64;
            let workgroups = (total_elements + wg_size - 1) / wg_size;
            pass.dispatch_workgroups(workgroups, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));

        Ok(output_tensors)
    }
}
*/


pub struct ExecutionEngine {
    ctx:     GpuContext,
    kernels: KernelManager,
}

impl ExecutionEngine {
    pub fn new(ctx: GpuContext) -> Self {
        Self { kernels: KernelManager::new(ctx.clone()), ctx }
    }

    fn run_gpu_task(&self, task: GpuTask, mm: &mut MemoryManager) -> anyhow::Result<()> {
        // 1) buffers opaques
        let inputs: Vec<&AbstractBuffer> = task.input_ids.iter()
            .map(|&id| mm.get_ref(id).expect("missing input buffer"))
            .collect();
        let outputs: Vec<&AbstractBuffer> = task.output_ids.iter()
            .map(|&id| mm.get_ref(id).expect("missing output buffer"))
            .collect();

        // 2) pipeline + layout (cache côté KernelManager)
        let (pipeline, layout) = self.kernels.get(
            &task.pipeline_source,
            &task.entry_point,
            task.input_types,
            task.output_types,
        ).expect("failed to get kernel");

        // 3) dispatch 1D opaque (bind-group éphémère créé en interne)
        let total: u32 = {
            let vd = &task.output_descs[0];
            (0..vd.ndim as usize).map(|i| vd.shape[i]).product()
        };
        self.ctx.dispatch_compute_1d(&pipeline, &layout, &inputs, &outputs, total, 64);

        Ok(())
    }

    pub fn run_prepared(
        &self,
        prepared: PreparedOp,
        mm: &mut MemoryManager,
    ) -> anyhow::Result<()> {
        match prepared {
            PreparedOp::Gpu(task) => {
                // Only check success of one operation
                self.run_gpu_task(task, mm)?;
                Ok(())
            }
            PreparedOp::Composite(ops) => {
                // Check success of all sub-operations
                for sub_op in ops {
                    self.run_prepared(sub_op, mm)?;
                }
                Ok(())
            }
        }
    }
}


/* ------------------------------------------------------------------------- */
/*                                  Tests                                    */
/* ------------------------------------------------------------------------- */
#[cfg(test)]
mod tests {
    use super::*;
    use vknp_ops::OpRegistry;
    use pollster::block_on;
    use vknp_core::GpuContext;
    use core_types::BufferId;
    use tensor::Tensor;

    #[test]
    fn run_add_op() {
        // --- init gpu + memory + engine ----------------------------------
        let ctx = block_on(GpuContext::new()).unwrap();
        let mut mm = MemoryManager::new(ctx.clone());

        let engine = ExecutionEngine::new(ctx.clone());

        // --- tensors -----------------------------------------------------
        let a = Tensor::<f32>::from_vec(&mut mm, &[1.0, 2.0, 3.0, 4.0], &[4], 0);
        let b = Tensor::<f32>::from_vec(&mut mm, &[5.0, 6.0, 7.0, 8.0], &[4], 0);
        let c = Tensor::<f32>::empty(&mut mm, &[4], 0);

        // --- registry & prepare -----------------------------------------
        let mut reg = OpRegistry::new();
        reg.collect_inventory();

        let op = reg
            .check_and_prepare("add", vec![a.into(), b.into()], vec![c.into()])
            .unwrap();

        // --- run ---------------------------------------------------------------
        let out_ids: Vec<BufferId> = match &op {
            PreparedOp::Gpu(task) => task.output_ids.clone(),
            PreparedOp::Composite(_) => unreachable!("test simple"),
        };
        engine.run_prepared(op, &mut mm).unwrap();

        // --- check results ------------------------------------------------
        let result: Vec<f32> = mm.download_raw(out_ids[0]).unwrap();
        assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
    }
}