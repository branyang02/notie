use std::collections::HashMap;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct EquationMapping(HashMap<String, EquationMeta>);

#[wasm_bindgen]
impl EquationMapping {
    // Constructor
    #[wasm_bindgen(constructor)]
    pub fn new() -> EquationMapping {
        EquationMapping(HashMap::new())
    }

    // Add an equation
    pub fn add_equation(&mut self, key: String, equation_number: String, equation_string: String) {
        let details = EquationMeta {
            equation_number,
            equation_string,
        };
        self.0.insert(key, details);
    }

    // Get an equation by key
    pub fn get_equation(&self, key: &str) -> Option<EquationMeta> {
        self.0.get(key).cloned()
    }

    // Get the number of equations
    pub fn len(&self) -> usize {
        self.0.len()
    }

    // Check if the mapping is empty
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

#[wasm_bindgen]
#[derive(Debug, Clone)]
pub struct EquationMeta {
    equation_number: String,
    equation_string: String,
}

#[wasm_bindgen]
impl EquationMeta {
    // Constructor
    #[wasm_bindgen(constructor)]
    pub fn new(equation_number: String, equation_string: String) -> EquationMeta {
        EquationMeta {
            equation_number,
            equation_string,
        }
    }

    // Get equation number
    pub fn get_equation_number(&self) -> String {
        self.equation_number.clone()
    }

    // Get equation string
    pub fn get_equation_string(&self) -> String {
        self.equation_string.clone()
    }
}
