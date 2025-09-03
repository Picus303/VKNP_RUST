mod kernel_manager;

use std::sync::Arc;

use vknp_core::GpuContext;
use vknp_core::types::AbstractBuffer;
use memory::MemoryManager;
use vknp_ops::types::{GpuTask, PreparedOp};

use kernel_manager::KernelManager;


/// Execution engine for running GPU tasks.
pub struct ExecutionEngine {
    ctx:     GpuContext,
    kernels: KernelManager,
}

impl ExecutionEngine {
    pub fn new(ctx: GpuContext) -> Self {
        Self { kernels: KernelManager::new(ctx.clone()), ctx }
    }

    fn run_gpu_task(&self, task: GpuTask, mm: &MemoryManager) -> anyhow::Result<()> {
        // 1) Allouer/écrire les buffers de paramètres (emprunt mutable)
        let mut param_ids = Vec::with_capacity(task.params.len());
        for p in &task.params {
            let id = mm.allocate_raw(p.bytes.len())?;
            mm.write_to_buffer(id, &p.bytes)?;
            param_ids.push(id);
        }

        // 2) Pipeline + layout
        let (pipeline, layout) = self.kernels
            .get(&task.pipeline_source, &task.entry_point, task.input_types, task.output_types, task.params.len())
            .map_err(|e| anyhow::anyhow!("failed to get kernel: {e}"))?;

        // 3) Total à partir du 1er output
        let total: u32 = {
            let vd = &task.output_descs[0];
            (0..vd.ndim as usize).map(|i| vd.shape[i]).product()
        };

        // 4) Créer les prêts immuables et dispatcher dans un *scope court*
        {
            let inputs: Vec<Arc<AbstractBuffer>> = task.input_ids.iter()
                .map(|&id| mm.get_ref(id).ok_or_else(|| anyhow::anyhow!("missing input buffer: {:?}", id)))
                .collect::<Result<_, _>>()?;

            let outputs: Vec<Arc<AbstractBuffer>> = task.output_ids.iter()
                .map(|&id| mm.get_ref(id).ok_or_else(|| anyhow::anyhow!("missing output buffer: {:?}", id)))
                .collect::<Result<_, _>>()?;

            let param_bufs: Vec<Arc<AbstractBuffer>> = param_ids.iter()
                .map(|&id| mm.get_ref(id).ok_or_else(|| anyhow::anyhow!("param buffer missing: {:?}", id)))
                .collect::<Result<_, _>>()?;

            let all_inputs: Vec<Arc<AbstractBuffer>> =
                inputs.iter().cloned().chain(param_bufs.iter().cloned()).collect();

            let all_inputs_refs: Vec<&AbstractBuffer> = all_inputs.iter().map(|arc| arc.as_ref()).collect();
            let outputs_refs: Vec<&AbstractBuffer> = outputs.iter().map(|arc| arc.as_ref()).collect();

            self.ctx.dispatch_compute_1d(&pipeline, &layout, &all_inputs_refs, &outputs_refs, total, 64);
        }

        // 5) Maintenant on peut ré-emprunter mutablement pour libérer
        for id in param_ids {
            mm.release(id);
        }

        Ok(())
    }

    pub fn run_prepared(
        &self,
        prepared: PreparedOp,
        mm: &MemoryManager,
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
        let mm = MemoryManager::new(ctx.clone());

        let engine = ExecutionEngine::new(ctx.clone());

        // --- tensors -----------------------------------------------------
        let a = Tensor::<f32>::from_vec(&mm, &[1.0, 2.0, 3.0, 4.0], &[4], 0);
        let b = Tensor::<f32>::from_vec(&mm, &[5.0, 6.0, 7.0, 8.0], &[1, 4], 0);
        let c = Tensor::<f32>::empty(&mm, &[4], 0);

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
        engine.run_prepared(op, &mm).unwrap();

        // --- check results ------------------------------------------------
        let result: Vec<f32> = mm.download_raw(out_ids[0]).unwrap();
        assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
    }
}