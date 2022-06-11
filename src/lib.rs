mod gravity;

use crate::gravity::Vec2;
use wasm_bindgen::prelude::*;


#[wasm_bindgen]
pub struct Simulation(gravity::Simulation);

#[wasm_bindgen]
impl Simulation {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Simulation {
        Simulation{0: gravity::Simulation::new()}
    }
    #[wasm_bindgen]
    pub fn wow(&mut self) -> f32 {
        self.0.add(&Vec2{x: 0.0, y: -5.0}, &Vec2{x: 0.1, y: 0.0}, 0.1);
        self.0.add(&Vec2{x: 0.0, y: 5.0}, &Vec2{x: -0.1, y: 5.0}, 0.1);
        5.5
    }
}


#[wasm_bindgen]
pub fn step(simulation: &mut Simulation, dt: f32) {
    gravity::step(&mut simulation.0, dt);
}

/*

#[wasm_bindgen]
extern {
    pub fn alert(s: &str);
}
#[wasm_bindgen]
pub fn greet(name: &str) {
    alert(&format!("Hello, {}!", name));
}
*/
