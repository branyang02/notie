extern crate cfg_if;
extern crate wasm_bindgen;

mod equation_mapping;
mod utils;

use crate::equation_mapping::EquationMapping;
use cfg_if::cfg_if;
use utils::set_panic_hook;
use wasm_bindgen::prelude::*;

cfg_if! {
    if #[cfg(feature = "wee_alloc")] {
        extern crate wee_alloc;
        #[global_allocator]
        static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;
    }
}

#[wasm_bindgen]
pub struct RustMarkdownProcessor {
    markdown_content: String,
    equation_mapping: EquationMapping,
}

#[wasm_bindgen]
impl RustMarkdownProcessor {
    // Constructor
    #[wasm_bindgen(constructor)]
    pub fn new(markdown_content: String) -> RustMarkdownProcessor {
        set_panic_hook();
        RustMarkdownProcessor {
            markdown_content,
            equation_mapping: EquationMapping::new(),
        }
    }

    pub fn process(&mut self) {
        // print processing
        println!("Processing markdown content...");
    }

    pub fn get_equation_mapping(&self) -> EquationMapping {
        self.equation_mapping.clone()
    }

    pub fn get_markdown_content(&self) -> String {
        self.markdown_content.clone()
    }
}
