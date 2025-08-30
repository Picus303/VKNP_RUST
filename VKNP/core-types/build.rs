use minijinja::{Environment, context};
use std::fs;
use std::path::Path;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, Serialize)]
struct TypeInfo {
    name: String,
    rust: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct TypeList {
    types: Vec<TypeInfo>,
}

fn main() {
    // Read the yaml file
    let yaml_path = Path::new("../supported_types.yaml");
    let yaml_str = fs::read_to_string(yaml_path)
        .expect("Unable to read supported_types.yaml");
    let type_list: TypeList = serde_yaml::from_str(&yaml_str)
        .expect("Failed to parse YAML");

    // Load the template from a file
    let template_path = Path::new("templates/data_types.jinja");
    let template_source = fs::read_to_string(template_path)
        .expect("Unable to read template file");

    let env = Environment::new();
    let tmpl = env.template_from_str(&template_source).unwrap();

    let rendered = tmpl.render(context! { types => type_list.types }).unwrap();

    fs::write("src/generated_data_types.rs", rendered)
        .expect("Unable to write generated file");
    
    // Tell cargo to rerun if files change
    println!("cargo:rerun-if-changed=../supported_types.yaml");
    println!("cargo:rerun-if-changed=templates/data_types.jinja");
}
