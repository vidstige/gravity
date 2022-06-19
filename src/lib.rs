mod gravity;

use crate::gravity::Vec2;
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Simulation(gravity::Simulation);

#[wasm_bindgen]
impl Simulation {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Simulation {
        let mut simulation = gravity::Simulation::new();
        simulation.G = 0.025;
        Simulation{0: simulation}
    }
    #[wasm_bindgen]
    pub fn add(&mut self, x: f32, y: f32, vx: f32, vy: f32, mass: f32) {
        self.0.add(&Vec2{x: x, y: y}, &Vec2{x: vx, y: vy}, mass);
    }
    
    #[wasm_bindgen(method, getter)]
    pub fn g(&mut self) -> f32 { self.0.G }

    #[wasm_bindgen(method, setter)]
    pub fn set_g(&mut self, G: f32) {
        self.0.G = G;
    }
}

#[wasm_bindgen]
pub fn step(simulation: &mut Simulation, dt: f32) {
    gravity::step(&mut simulation.0, dt);
}

#[wasm_bindgen]
pub fn render(target: String, simulation: &Simulation) {
    let document = web_sys::window().unwrap().document().unwrap();
    let canvas = document.get_element_by_id(&target).unwrap();
    let canvas: web_sys::HtmlCanvasElement = canvas
        .dyn_into::<web_sys::HtmlCanvasElement>()
        .map_err(|_| ())
        .unwrap();

    let ctx = canvas
        .get_context("2d")
        .unwrap()
        .unwrap()
        .dyn_into::<web_sys::CanvasRenderingContext2d>()
        .unwrap();

    ctx.reset_transform();
    ctx.clear_rect(0.0, 0.0, canvas.width() as f64, canvas.height() as f64);
    ctx.translate(0.5 * canvas.width() as f64, 0.5 * canvas.height() as f64);
    
    ctx.begin_path();
    for p in simulation.0.positions() {
        ctx.arc(p.x as f64, p.y as f64, 5.0, 0.0, std::f64::consts::TAU);
    }
    ctx.fill();
    
/*
    context.begin_path();

    // Draw the outer circle.
    context
        .arc(75.0, 75.0, 50.0, 0.0, std::f64::consts::TAU)
        .unwrap();

    // Draw the mouth.
    context.move_to(110.0, 75.0);
    context.arc(75.0, 75.0, 35.0, 0.0, 0.5 * std::f64::consts::TAU).unwrap();

    // Draw the left eye.
    context.move_to(65.0, 65.0);
    context
        .arc(60.0, 65.0, 5.0, 0.0, std::f64::consts::TAU)
        .unwrap();

    // Draw the right eye.
    context.move_to(95.0, 65.0);
    context
        .arc(90.0, 65.0, 5.0, 0.0, std::f64::consts::TAU)
        .unwrap();

    context.stroke();*/
}