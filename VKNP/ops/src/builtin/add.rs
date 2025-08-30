use core_types::DataType;

use crate::op::Op;
use crate::register_op;
use crate::types::{OpSignature, GpuTask, PreparedOp, TensorAny, RegistrationInfo};


/// “add” f32+f32 → f32 (1 output)
pub struct AddOp {
    sig: OpSignature,
}

impl AddOp {
    pub fn new() -> Self {
        let dt = DataType::F32;
        Self {
            sig: OpSignature {
                name:          "add",
                num_inputs:    2,
                num_outputs:   1,
                input_dtypes:  vec![ vec![dt], vec![dt] ],
                output_dtypes: vec![ vec![dt] ],
            },
        }
    }
}

impl RegistrationInfo for AddOp {
    const NAME: &'static str = "add";
}

impl Op for AddOp {
    fn signature(&self) -> &OpSignature { &self.sig }

    fn prepare(
        &self,
        inputs:  &[TensorAny],
        outputs: &[TensorAny],
    ) -> PreparedOp {
        let a = match &inputs[0]  { TensorAny::F32(t) => t, _ => unreachable!() };
        let b = match &inputs[1]  { TensorAny::F32(t) => t, _ => unreachable!() };
        let c = match &outputs[0] { TensorAny::F32(t) => t, _ => unreachable!() };

        let (src, entry) = self.shader_template();
        let task = GpuTask {
            pipeline_source: src.to_string(),
            entry_point:     entry.to_string(),
            input_descs:     vec![ a.view().clone(), b.view().clone() ],
            output_descs:    vec![ c.view().clone() ],
            input_types:     vec![ a.dtype(), b.dtype() ],
            output_types:    vec![ c.dtype() ],
            input_ids:       vec![ a.buffer_id(), b.buffer_id() ],
            output_ids:      vec![ c.buffer_id() ],
        };
        PreparedOp::Gpu(task)
    }

    fn shader_template(&self) -> (&'static str, &'static str) {
        (r#"
            @group(0) @binding(0) var<storage, read>  A: array<f32>;
            @group(0) @binding(1) var<storage, read>  B: array<f32>;
            @group(0) @binding(2) var<storage, read_write> C: array<f32>;
            @compute @workgroup_size(64)
            fn add_kernel(@builtin(global_invocation_id) gid: vec3<u32>) {
                let i = gid.x;
                C[i] = A[i] + B[i];
            }
        "#, "add_kernel")
    }
}

register_op!(AddOp);