pub mod op;
pub mod types;
pub mod builtin;

use std::collections::HashMap;
use types::{PreparedOp, TensorAny, OpError, RegistrationInfo};
use op::{Op, OpFactory};


/// Register an operation with the inventory system
#[macro_export]
macro_rules! register_op {
    ($op_type:ident) => {
        inventory::submit! {
            $crate::OpFactory {
                name: <$op_type as $crate::RegistrationInfo>::NAME,
                factory: || Box::new($op_type::new()),
            }
        }
    };
}


/// Holds all registered ops, validates signature & dtypes, then calls prepare()
pub struct OpRegistry {
    map: HashMap<&'static str, Box<dyn Op>>,
}

impl OpRegistry {
    pub fn new() -> Self {
        Self { map: HashMap::new() }
    }

    pub fn collect_inventory(&mut self) {
        for factory in inventory::iter::<OpFactory> {
            let op = (factory.factory)();
            self.register_boxed(factory.name, op);
        }
    }

    /// Register a new Op under its signature name
    pub fn register<O: Op + 'static>(&mut self, op: O) {
        let name = op.signature().name;
        self.map.insert(name, Box::new(op));
    }

    /// Register a boxed Op with an explicit name
    pub fn register_boxed(&mut self, name: &'static str, op: Box<dyn Op>) {
        self.map.insert(name, op);
    }

    /// Lookup + validate arity & dtypes + prepare in one call
    pub fn check_and_prepare(
        &self,
        name:    &str,
        inputs:  Vec<TensorAny>,
        outputs: Vec<TensorAny>,
    ) -> Result<PreparedOp, OpError> {
        let op = self.map.get(name)
            .ok_or(OpError::UnknownOp(name.to_string()))?;
        let sig = op.signature();

        // inputs
        if inputs.len() != sig.num_inputs {
            return Err(OpError::ArityMismatch {
                op: name.to_string(),
                expected: sig.num_inputs,
                found: inputs.len(),
            });
        }
        for (i, t) in inputs.iter().enumerate() {
            let dt = t.dtype();
            if !sig.input_dtypes[i].contains(&dt) {
                return Err(OpError::DtypeMismatch {
                    op: name.to_string(),
                    index: i,
                    expected: sig.input_dtypes[i].clone(),
                    found: dt,
                });
            }
        }

        // outputs
        if outputs.len() != sig.num_outputs {
            return Err(OpError::ArityMismatch {
                op: name.to_string(),
                expected: sig.num_outputs,
                found: outputs.len(),
            });
        }
        for (i, t) in outputs.iter().enumerate() {
            let dt = t.dtype();
            if !sig.output_dtypes[i].contains(&dt) {
                return Err(OpError::DtypeMismatch {
                    op: name.to_string(),
                    index: i,
                    expected: sig.output_dtypes[i].clone(),
                    found: dt,
                });
            }
        }

        // prepare the operation
        Ok(op.prepare(&inputs, &outputs))
    }

    /// lookup sans validation
    pub fn get(&self, name: &str) -> Option<&dyn Op> {
        self.map.get(name).map(|b| b.as_ref())
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use vknp_core::GpuContext;
    use memory::MemoryManager;
    use pollster::block_on;
    use tensor::Tensor;

    #[test]
    fn registry_and_addop() {
        // setup GPU & memory manager
        let ctx = block_on(GpuContext::new()).unwrap();
        let mut mm = MemoryManager::new(ctx);

        // register all available ops
        let mut reg = OpRegistry::new();
        reg.collect_inventory();

        // print the number and name of registered ops
        println!("Registered {} ops", reg.map.len());
        for name in reg.map.keys() {
            println!(" - {}", name);
        }

        // prepare with three f32 tensors
        let shape = [4];
        let t1 = Tensor::<f32>::empty(&mut mm, &shape, 0);
        let t2 = Tensor::<f32>::empty(&mut mm, &shape, 0);
        let t3 = Tensor::<f32>::empty(&mut mm, &shape, 0);

        let inputs = vec![ TensorAny::F32(t1), TensorAny::F32(t2) ];
        let outputs = vec![ TensorAny::F32(t3) ];

        // check and prepare the "add" op
        let prepared = reg.check_and_prepare("add", inputs, outputs).unwrap();
        match prepared {
            PreparedOp::Gpu(task) => {
                // should have one output descriptor
                assert_eq!(task.output_descs.len(), 1);
                assert_eq!(task.entry_point, "add_kernel");
            }
            _ => panic!("AddOp should produce a single GpuTask"),
        }

        // requesting unknown op errors
        let err = reg.check_and_prepare("extremely_strange_op", vec![], vec![]).unwrap_err();
        match err {
            OpError::UnknownOp(name) => assert_eq!(name, "extremely_strange_op"),
            _ => panic!("expected UnknownOp"),
        }
    }
}