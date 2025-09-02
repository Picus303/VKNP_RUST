use bytemuck::{Pod, Zeroable};
use core_types::{DataType, ViewDescriptor, MAX_DIMS};

use crate::op::Op;
use crate::register_op;
use crate::types::{OpSignature, ParamBuffer, GpuTask, PreparedOp, TensorAny, RegistrationInfo};


#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct ViewU {
    offset:  u32,
    ndim:    u32,
    _pad0:   [u32; 2],
    shape:   [u32; MAX_DIMS],
    strides: [u32; MAX_DIMS],
}

#[repr(C)]
#[derive(Clone, Copy, Zeroable, Pod)]
struct MetaU {
    a: ViewU,
    b: ViewU,
    c: ViewU,
    total_elems: u32,
    _pad1: [u32; 3],
    _tail_pad: [u32; 4],
}

fn descriptor_to_uniform(v: &ViewDescriptor) -> ViewU {
    ViewU { offset: v.offset, ndim: v.ndim, _pad0: [0;2], shape: v.shape, strides: v.strides }
}

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
        // Convert to universal tensor
        let a = match &inputs[0]  { TensorAny::F32(t) => t, _ => unreachable!() };
        let b = match &inputs[1]  { TensorAny::F32(t) => t, _ => unreachable!() };
        let c = match &outputs[0] { TensorAny::F32(t) => t, _ => unreachable!() };

        // Create metadata
        let mut total = 1u32;
        for d in 0..(c.view().ndim as usize) { total *= c.view().shape[d]; }

        // param buffer (storage RO) : MetaU
        let meta = MetaU {
            a: descriptor_to_uniform(a.view()),
            b: descriptor_to_uniform(b.view()),
            c: descriptor_to_uniform(c.view()),
            total_elems: total,
            _pad1: [0;3],
            _tail_pad: [0;4],
        };
        let param = ParamBuffer { bytes: bytemuck::bytes_of(&meta).to_vec() };

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
            params:          vec![param],
        };
        PreparedOp::Gpu(task)
    }

    fn shader_template(&self) -> (&'static str, &'static str) {
        (ADD_WGSL, "add_strided")
    }
}

register_op!(AddOp);

const ADD_WGSL: &str = r#"
const MAX_DIMS : u32 = 8u;

struct View {
  offset  : u32,
  ndim    : u32,
  _pad0   : vec2<u32>,
  shape   : array<u32, MAX_DIMS>,
  strides : array<u32, MAX_DIMS>,
};

struct Meta {
  a           : View,
  b           : View,
  c           : View,
  total_elems : u32,
  _pad1       : vec3<u32>,
};

@group(0) @binding(0) var<storage, read>  A    : array<f32>;
@group(0) @binding(1) var<storage, read>  B    : array<f32>;
@group(0) @binding(2) var<storage, read>  M    : Meta;
@group(0) @binding(3) var<storage, read_write> C    : array<f32>;

fn linear_to_offsets(i: u32, v: View) -> u32 {
  var idx = i;
  var off = v.offset;
  var d: i32 = i32(v.ndim) - 1;
  loop {
    if (d < 0) { break; }
    let du : u32 = u32(d);
    let dim = v.shape[du];
    let coord = idx % dim;
    idx = idx / dim;
    off = off + coord * v.strides[du]; // stride 0 -> broadcast
    d = d - 1;
  }
  return off;
}

@compute @workgroup_size(64)
fn add_strided(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= M.total_elems) { return; }

  let ai = linear_to_offsets(i, M.a);
  let bi = linear_to_offsets(i, M.b);
  let ci = linear_to_offsets(i, M.c);

  C[ci] = A[ai] + B[bi];
}
"#;